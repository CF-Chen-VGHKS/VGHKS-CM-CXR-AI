# predict.py
# Prediction module: Responsible for loading models, executing predictions, extracting confidence values, etc.
# Image preprocessing: Resize image to 512 (maintain aspect ratio), then apply CLAHE
# model1 (Pattern) + model1-Nod: Perform detection on both 512-origin and 512-CLAHE. model1-Nod performs Bbox screening first.
# Total 4 results (model1/origin, model1/CLAHE, model1-Nod/origin, model1-Nod/CLAHE) integrated using NMS
# Added functionality to output YOLO format annotation + confidence txt files
# Final results are drawn on the original size image and saved

import cv2
from PIL import Image
import os
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors
from ultralytics.engine.results import Results


class ModelManager:
    """Model Manager: Responsible for loading and managing all YOLO models"""
    
    def __init__(self, model_paths):
        """Initialize model manager
        
        Args:
            model_paths: Dictionary containing paths for each model
                - 'model1': Pattern detection model path
                - 'model1-Nod': Pattern detection model with bbox screening for nodule
        """
        self.models = {}
        self.model_paths = model_paths
        
    def load_all_models(self):
        """Load all models"""
        print("Loading AI models...")
        
        try:
            self.models['model1'] = YOLO(self.model_paths['model1'])
            self.models['model1-Nod'] = YOLO(self.model_paths['model1-Nod'])
            
            print("✓ All models loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to load models: {str(e)}")
            return False
    
    def get_model(self, model_name):
        """Get the specified model"""
        return self.models.get(model_name)


def extract_detection_confidence(results, target_classes):
    """Extract confidence values from object detection model
    
    Args:
        results: YOLO model prediction results
        target_classes: List of target class names to focus on
    
    Returns:
        dict: Mapping of class names to a list of confidence values
    """
    if results.boxes is None or len(results.boxes) == 0:
        return {}
    
    class_names = results.names
    confidence_dict = {}
    
    for i, box in enumerate(results.boxes):
        class_id = int(box.cls[0])
        class_name = class_names[class_id]
        confidence = float(box.conf[0])
        
        if class_name.lower() in [tc.lower() for tc in target_classes]:
            if class_name not in confidence_dict:
                confidence_dict[class_name] = []
            confidence_dict[class_name].append(confidence)
    
    return confidence_dict


def extract_classification_confidence(results, target_classes):
    """Extract confidence values from classification model
    
    Args:
        results: YOLO classification model prediction results
        target_classes: List of target class names to focus on
    
    Returns:
        dict: Mapping of class names to confidence values
    """
    if results.probs is None:
        return {}
    
    class_names = results.names
    probs = results.probs.data
    confidence_dict = {}
    
    for class_id, prob in enumerate(probs):
        class_name = class_names[class_id]
        confidence = float(prob)
        
        if class_name.lower() in [tc.lower() for tc in target_classes]:
            confidence_dict[class_name] = confidence
    
    return confidence_dict


def check_trigger_condition(model1_confidences, trigger_classes, trigger_threshold):
    """Check if any target class confidence value is greater than the threshold
    
    Args:
        model1_confidences: Prediction result confidence dictionary of Model 1
        trigger_classes: List of trigger classes
        trigger_threshold: Trigger threshold value
    
    Returns:
        tuple: (Is triggered, List of triggered classes)
    """
    triggered_classes = []
    
    for class_name in trigger_classes:
        if class_name in model1_confidences:
            confidences = model1_confidences[class_name]
            max_conf = max(confidences) if isinstance(confidences, list) else confidences
            
            if max_conf > trigger_threshold:
                triggered_classes.append((class_name, max_conf))
    
    return len(triggered_classes) > 0, triggered_classes


def get_confidence_value(confidences_dict, class_name, default_value=0.0):
    """Safely get confidence value
    
    Args:
        confidences_dict: Confidence dictionary
        class_name: Class name
        default_value: Default value
    
    Returns:
        float: Confidence value
    """
    if not confidences_dict:
        return default_value
    
    value = confidences_dict.get(class_name)
    
    if value is None:
        return default_value
    elif isinstance(value, list):
        return max(value) if value else default_value
    else:
        return float(value)


def _find_nodule_class_id(class_names, nodule_class_name="nodule"):
    """Find the ID of the nodule class"""
    for class_id, name in class_names.items():
        if name.lower() == nodule_class_name.lower():
            return class_id
    return None


def apply_bbox_screening(
    results,
    img_width: int,
    img_height: int,
    nodule_class_id: int | None = None,
    nodule_class_name: str = "nodule",
    size_threshold_ratio: float = 1/64
):
    """Apply Bbox screening to YOLO prediction results: remove excessively large nodule bboxes
    
    Reference YOLOBatchPredictor: Remove when nodule bbox area > image area * size_threshold_ratio
    
    Args:
        results: YOLO prediction results
        img_width: Image width
        img_height: Image height
        nodule_class_name: Nodule class name
        size_threshold_ratio: Threshold ratio of bbox area to image area (default 1/64)
    
    Returns:
        Results-like object containing filtered boxes
    """
    if results.boxes is None or len(results.boxes) == 0:
        return results
    
    img_area = img_width * img_height
    max_bbox_area = img_area * size_threshold_ratio
    class_names = results.names
    if nodule_class_id is None:
        nodule_class_id = _find_nodule_class_id(class_names, nodule_class_name)
    
    filtered_boxes = []
    filtered_conf = []
    filtered_cls = []
    
    for i in range(len(results.boxes)):
        box = results.boxes[i]
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu()
        
        x1, y1, x2, y2 = xyxy.numpy()
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # Remove if it is a nodule and area exceeds threshold
        if nodule_class_id is not None and cls_id == nodule_class_id and bbox_area > max_bbox_area:
            continue
        
        filtered_boxes.append(xyxy.unsqueeze(0))
        filtered_conf.append(torch.tensor([conf], device=xyxy.device))
        filtered_cls.append(torch.tensor([cls_id], device=xyxy.device, dtype=torch.long))
    
    if len(filtered_boxes) == 0:
        # No boxes kept, return empty Results-like object
        return _create_empty_filtered_results(results)
    
    filtered_boxes_t = torch.cat(filtered_boxes, dim=0)
    filtered_conf_t = torch.cat(filtered_conf, dim=0)
    filtered_cls_t = torch.cat(filtered_cls, dim=0)
    
    return _create_filtered_results(
        filtered_boxes_t, filtered_conf_t, filtered_cls_t,
        results
    )

def rewrite_class_id_fixed(source_results, reference_results, src_class_id: int, dst_class_id: int):
    """Rewrite class id with fixed rules (not searching by name), align class ids of different models before merging.
    
    Example: model1-Nod's nodule=0, rewritten to model1's nodule=5.
    """
    if source_results.boxes is None or len(source_results.boxes) == 0:
        return source_results
    
    boxes_xyxy = source_results.boxes.xyxy.clone()
    conf = source_results.boxes.conf.clone()
    cls = source_results.boxes.cls
    
    # cls might be [N] or [N,1], flatten to [N] uniformly
    cls_out = cls.view(-1).long().clone()
    cls_out[cls_out == int(src_class_id)] = int(dst_class_id)
    
    return _create_filtered_results(
        boxes_xyxy=boxes_xyxy,
        conf=conf.view(-1).clone(),
        cls=cls_out,
        base_result=reference_results,
        orig_shape=getattr(source_results, 'orig_shape', None)
    )


def _create_empty_filtered_results(base_result):
    """Create an empty Results-like object"""
    device = base_result.boxes.xyxy.device if base_result.boxes is not None else torch.device('cpu')
    empty_boxes = torch.zeros((0, 4), device=device)
    empty_conf = torch.zeros((0,), device=device)
    empty_cls = torch.zeros((0,), device=device, dtype=torch.long)
    return _create_filtered_results(empty_boxes, empty_conf, empty_cls, base_result)


def _create_filtered_results(boxes_xyxy, conf, cls, base_result, orig_shape=None):
    """Create Results-like object from box tensors"""
    class BoxItem:
        def __init__(self, xyxy_tensor, cls_tensor, conf_tensor):
            if xyxy_tensor.numel() == 4:
                self.xyxy = xyxy_tensor.unsqueeze(0) if len(xyxy_tensor.shape) == 1 else xyxy_tensor
            else:
                self.xyxy = xyxy_tensor.unsqueeze(0) if len(xyxy_tensor.shape) == 1 else xyxy_tensor
            self.cls = cls_tensor.unsqueeze(0) if len(cls_tensor.shape) == 0 else (cls_tensor[0:1] if cls_tensor.shape[0] > 0 else cls_tensor.unsqueeze(0))
            self.conf = conf_tensor.unsqueeze(0) if len(conf_tensor.shape) == 0 else (conf_tensor[0:1] if conf_tensor.shape[0] > 0 else conf_tensor.unsqueeze(0))
    
    class MergedBoxes:
        def __init__(self, boxes_xyxy, conf, cls):
            self.xyxy = boxes_xyxy
            self.conf = conf
            self.cls = cls
            self.xywh = self.xyxyn = self.xywhn = None
            self._box_items = [BoxItem(boxes_xyxy[i], cls[i], conf[i]) for i in range(len(boxes_xyxy))]
        def __len__(self):
            return len(self._box_items)
        def __iter__(self):
            return iter(self._box_items)
        def __getitem__(self, index):
            return self._box_items[index]
    
    class FilteredResults:
        def __init__(self, boxes_xyxy, conf, cls, names, orig_img=None, path=None, orig_shape=None):
            self.boxes = MergedBoxes(boxes_xyxy, conf, cls)
            self.names = names
            self.orig_img = orig_img
            self.path = path
            self.orig_shape = orig_shape if orig_shape is not None else (0, 0)
    
    orig_img = base_result.orig_img if hasattr(base_result, 'orig_img') else None
    path = base_result.path if hasattr(base_result, 'path') else None
    if orig_shape is None:
        orig_shape = getattr(base_result, 'orig_shape', None) or (orig_img.shape[:2] if orig_img is not None else (0, 0))
    
    return FilteredResults(
        boxes_xyxy=boxes_xyxy, conf=conf, cls=cls,
        names=base_result.names,
        orig_img=orig_img, path=path, orig_shape=orig_shape
    )


def merge_and_apply_nms_multi(results_list, iou_threshold=0.5, orig_shape=None):
    """Merge multiple YOLO prediction results and apply Non-Maximum Suppression (NMS)
    
    Args:
        results_list: List of YOLO prediction results (can be model1/origin, model1/CLAHE, model1-Nod/origin, model1-Nod/CLAHE, etc.)
        iou_threshold: IoU threshold for NMS
        orig_shape: Original image dimensions (height, width), inferred from the first result if None
    
    Returns:
        Results: New result object after merging and applying NMS
    """
    all_boxes = []
    all_confidences = []
    all_class_ids = []
    base_result = results_list[0] if results_list else None
    
    for result in results_list:
        if result.boxes is not None and len(result.boxes) > 0:
            all_boxes.append(result.boxes.xyxy.cpu())
            all_confidences.append(result.boxes.conf.cpu())
            all_class_ids.append(result.boxes.cls.cpu())
    
    if len(all_boxes) == 0:
        for result in results_list:
            if result.boxes is not None and len(result.boxes) > 0:
                return result
        return base_result if base_result else results_list[0]
    
    # Merge all boxes
    merged_boxes = torch.cat(all_boxes, dim=0)
    merged_conf = torch.cat(all_confidences, dim=0)
    merged_cls = torch.cat(all_class_ids, dim=0)
    
    # Process NMS grouped by class (because NMS should be performed within the same class)
    unique_classes = torch.unique(merged_cls)
    final_boxes = []
    final_conf = []
    final_cls = []
    
    for cls_id in unique_classes:
        # Get all boxes for this class
        cls_mask = (merged_cls == cls_id)
        cls_boxes = merged_boxes[cls_mask]
        cls_conf = merged_conf[cls_mask]
        cls_ids = merged_cls[cls_mask]
        
        if len(cls_boxes) == 0:
            continue
        
        # Apply NMS (using torchvision's NMS)
        # torchvision.ops.nms requires boxes format to be [x1, y1, x2, y2]
        # and requires score and boxes tensors to be on the same device
        import torchvision
        keep_indices = torchvision.ops.nms(cls_boxes, cls_conf, iou_threshold)
        
        # Keep boxes after NMS
        final_boxes.append(cls_boxes[keep_indices])
        final_conf.append(cls_conf[keep_indices])
        final_cls.append(cls_ids[keep_indices])
    
    # If no boxes are kept, return the first result
    if len(final_boxes) == 0:
        return base_result
    
    # Merge results of all classes
    final_boxes = torch.cat(final_boxes, dim=0)
    final_conf = torch.cat(final_conf, dim=0)
    final_cls = torch.cat(final_cls, dim=0)
    
    # Sort by confidence (from high to low)
    sorted_indices = torch.argsort(final_conf, descending=True)
    final_boxes = final_boxes[sorted_indices]
    final_conf = final_conf[sorted_indices]
    final_cls = final_cls[sorted_indices]
    
    # Create new Results object
    # We need to copy necessary attributes from base_result
    # Since Results object structure is complex, we create a new result object
    # but keep the original boxes structure
    
    # Use orig_img and path from base_result
    orig_img = base_result.orig_img if hasattr(base_result, 'orig_img') else None
    path = base_result.path if hasattr(base_result, 'path') else None
    
    # Create a new Results object
    # Since ultralytics' Results object structure is complex, we need to build it manually
    # The simplest way is to create a result object containing all necessary attributes
    
    # Transfer results to the same device as original results
    device = torch.device('cpu')
    for result in results_list:
        if result.boxes is not None and len(result.boxes) > 0:
            device = result.boxes.xyxy.device
            break
    
    final_boxes = final_boxes.to(device)
    final_conf = final_conf.to(device)
    final_cls = final_cls.to(device).long()
    
    # Create a new Results object
    # Since we cannot directly modify the Results object, we create a wrapper class to simulate Results structure
    # Need to ensure boxes can be iterated, and each box has cls, conf, and xyxy attributes
    
    class BoxItem:
        """Single box item, simulating ultralytics' Box structure"""
        def __init__(self, xyxy_tensor, cls_tensor, conf_tensor):
            # Ensure xyxy_tensor is a tensor of shape [4], then make it [1, 4]
            # So box.xyxy[0] returns a tensor of shape [4], which can be unpacked to x1, y1, x2, y2
            if xyxy_tensor.numel() == 4:
                # Ensure it's a 1D tensor, then add batch dimension
                if len(xyxy_tensor.shape) == 0:
                    # Scalar (should not happen, but for safety)
                    self.xyxy = xyxy_tensor.unsqueeze(0).unsqueeze(0)
                elif len(xyxy_tensor.shape) == 1:
                    # [4] -> [1, 4]
                    self.xyxy = xyxy_tensor.unsqueeze(0)
                elif len(xyxy_tensor.shape) == 2 and xyxy_tensor.shape[0] == 1:
                    # Already [1, 4]
                    self.xyxy = xyxy_tensor
                else:
                    # Try to reshape to [1, 4]
                    self.xyxy = xyxy_tensor.view(1, 4)
            else:
                # If not 4 elements, keep as is (will cause error, but at least won't crash)
                self.xyxy = xyxy_tensor.unsqueeze(0) if len(xyxy_tensor.shape) == 1 else xyxy_tensor
            
            # cls and conf should be scalars, make them [1] shape to support [0] indexing
            if len(cls_tensor.shape) == 0:  # Scalar
                self.cls = cls_tensor.unsqueeze(0)  # Scalar -> [1]
            elif len(cls_tensor.shape) == 1:
                # Ensure it's [1] shape
                if cls_tensor.shape[0] == 1:
                    self.cls = cls_tensor
                else:
                    # Take first element and reshape to [1]
                    self.cls = cls_tensor[0:1] if cls_tensor.shape[0] > 0 else cls_tensor.unsqueeze(0)
            else:
                # Try to reshape to [1]
                if cls_tensor.numel() == 1:
                    self.cls = cls_tensor.view(1)
                else:
                    self.cls = cls_tensor.flatten()[0:1] if cls_tensor.numel() > 0 else cls_tensor.unsqueeze(0)
                
            if len(conf_tensor.shape) == 0:  # Scalar
                self.conf = conf_tensor.unsqueeze(0)  # Scalar -> [1]
            elif len(conf_tensor.shape) == 1:
                # Ensure it's [1] shape
                if conf_tensor.shape[0] == 1:
                    self.conf = conf_tensor
                else:
                    # Take first element and reshape to [1]
                    self.conf = conf_tensor[0:1] if conf_tensor.shape[0] > 0 else conf_tensor.unsqueeze(0)
            else:
                # Try to reshape to [1]
                if conf_tensor.numel() == 1:
                    self.conf = conf_tensor.view(1)
                else:
                    self.conf = conf_tensor.flatten()[0:1] if conf_tensor.numel() > 0 else conf_tensor.unsqueeze(0)
    
    class MergedBoxes:
        """Merged Boxes object, simulating ultralytics' Boxes structure"""
        def __init__(self, boxes_xyxy, conf, cls):
            self.xyxy = boxes_xyxy
            self.conf = conf
            self.cls = cls
            
            # Add some extra attributes for compatibility
            self.xywh = None
            self.xyxyn = None
            self.xywhn = None
            
            # Store box item list for iteration
            self._box_items = []
            for i in range(len(boxes_xyxy)):
                # boxes_xyxy[i] shape is [4], cls[i] and conf[i] are scalars
                self._box_items.append(BoxItem(boxes_xyxy[i], cls[i], conf[i]))
        
        def __len__(self):
            return len(self._box_items)
        
        def __iter__(self):
            return iter(self._box_items)
        
        def __getitem__(self, index):
            return self._box_items[index]
    
    class MergedResults:
        """Merged Results object, simulating YOLO Results structure"""
        def __init__(self, boxes_xyxy, conf, cls, names, orig_img=None, path=None, orig_shape=None):
            self.boxes = MergedBoxes(boxes_xyxy, conf, cls)
            self.names = names
            self.orig_img = orig_img
            self.path = path
            self.orig_shape = orig_shape if orig_shape is not None else (0, 0)
    
    # Get orig_shape (prefer passed parameter, otherwise infer from base_result)
    if orig_shape is None:
        if hasattr(base_result, 'orig_shape'):
            orig_shape = base_result.orig_shape
        elif orig_img is not None:
            # If no orig_shape attribute, infer from orig_img
            orig_shape = orig_img.shape[:2]  # (height, width)
        elif hasattr(base_result, 'orig_img') and base_result.orig_img is not None:
            orig_shape = base_result.orig_img.shape[:2]  # (height, width)
        else:
            orig_shape = (0, 0)  # Default value
    
    merged_result = MergedResults(
        boxes_xyxy=final_boxes,
        conf=final_conf,
        cls=final_cls,
        names=base_result.names,
        orig_img=orig_img,
        path=path,
        orig_shape=orig_shape
    )
    
    return merged_result


def merge_and_apply_nms(results1, results2, iou_threshold=0.5, orig_shape=None):
    """Merge two YOLO prediction results and apply NMS (for backward compatibility)"""
    return merge_and_apply_nms_multi(
        [results1, results2], iou_threshold=iou_threshold, orig_shape=orig_shape
    )


def run_prediction(image, model_manager, config):
    """Execute the complete prediction process
    
    Args:
        image: Input image (numpy array)
        model_manager: Model manager instance
        config: Configuration parameters dictionary
    
    Returns:
        dict: Dictionary containing all prediction results, original image, and scale factor information
    """
    conf_set = config.get('conf_set', 0.01)
    iou_set = config.get('iou_set', 0.05)
    trigger_threshold = config.get('trigger_threshold', 0.07)
    trigger_classes = config.get('trigger_classes', ['Opacity', 'Effusion', 'Fibrosis'])
    target_classes = config.get('target_classes', {})
    
    # Save original image dimensions
    original_height, original_width = image.shape[:2]
    original_image = image.copy()
    
    # Resize image to 512 pixels width (maintain aspect ratio)
    target_width = 512
    scale_factor = target_width / original_width
    new_height = int(original_height * scale_factor)
    
    pil_image = Image.fromarray(image).convert("RGB")  # numpy -> PIL, ensure RGB
    resized_pil = pil_image.resize((target_width, new_height), Image.Resampling.LANCZOS)
    resized_image = np.array(resized_pil)  # PIL -> numpy, for subsequent cv2 use
    
    # ========== Step 1: Determine if image is PA or Lat ==========
    model_pa_lat = model_manager.get_model('model_PA-Lat')
    pa_lat_result = model_pa_lat(resized_image)[0]
    
    # Extract PA/Lat classification result
    is_pa = False
    is_lat = False
    pa_confidence = 0.0
    lat_confidence = 0.0
    
    if pa_lat_result.probs is not None:
        class_names = pa_lat_result.names
        probs = pa_lat_result.probs.data
        
        for class_id, prob in enumerate(probs):
            class_name = class_names[class_id]
            confidence = float(prob)
            
            if class_name.lower() in ['lat']:
                lat_confidence = confidence
                pa_confidence = 1- confidence
                if confidence > 0.5:
                    is_lat = True
                    is_pa = False
                else: 
                    is_pa = True
                    is_lat = False
    
    # Initialize results dictionary
    results = {
        'is_pa': is_pa,
        'is_lat': is_lat,
        'pa_confidence': pa_confidence,
        'lat_confidence': lat_confidence,
        'model1_results': None,
        'model1_confidences': {},
        'model1_Nod_results': None,
        'model1_Nod_confidences': {},
        'model2_confidences': {},
        'model3_confidences': {},
        'model4_confidences': {},
        'model5_confidences': {},
        'model_COPD-Lat_confidences': {},
        'trigger_activated': False,
        'triggered_classes': [],
        'original_image': original_image,
        'original_size': (original_width, original_height),
        'scale_factor': scale_factor,
        'resized_image': resized_image,
        'processed_image': None  # Will be set if PA or Lat
    }
    
    # ========== If Lat, only execute model_COPD-Lat ==========
    if is_lat:
        print(f"Lateral image detected (Lat confidence: {lat_confidence:.3f}), executing COPD-Lat prediction...")
        
        # Apply CLAHE processing (same as PA flow)
        if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_image = clahe.apply(gray_image)
            processed_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed_image = clahe.apply(resized_image)
            if len(processed_image.shape) == 2:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        
        results['processed_image'] = processed_image
        
        # Execute model_COPD-Lat prediction
        model_copd_lat = model_manager.get_model('model_COPD-Lat')
        copd_lat_results = model_copd_lat(processed_image)[0]
        results['model_COPD-Lat_confidences'] = extract_classification_confidence(
            copd_lat_results, target_classes.get('model_COPD-Lat', ['COPD'])
        )
        
        return results
    
    # ========== If PA, execute full model1~model5 flow ==========
    if is_pa:
        print(f"PA image detected (PA confidence: {pa_confidence:.3f}), executing full prediction flow...")
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # If image is color (3 channels), convert to grayscale before applying CLAHE
        if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
            # Convert to grayscale
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
            # Create CLAHE object
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # Apply CLAHE
            clahe_image = clahe.apply(gray_image)
            # Convert back to RGB format (3 channels)
            processed_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2RGB)
        else:
            # Already grayscale, apply CLAHE directly
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed_image = clahe.apply(resized_image)
            # Ensure it is 3 channels (if model requires)
            if len(processed_image.shape) == 2:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
        
        results['processed_image'] = processed_image
        
        # Model 1 Prediction (Pattern Detection) - Predict on both resized_image and processed_image
        model1 = model_manager.get_model('model1')
        results1_resized = model1(resized_image, conf=conf_set, iou=iou_set)[0]
        results1_processed = model1(processed_image, conf=conf_set, iou=iou_set)[0]
        
        # Model 1-Nod Prediction - Perform Bbox screening before merging
        nodule_class_name = config.get('nodule_class_name', 'nodule')
        size_threshold_ratio = config.get('size_threshold_ratio', 1/64)
        # Fixed rule: model1-Nod's nodule is 0, change to model1's 5 before merging
        model1_nod_nodule_id = int(config.get('model1_nod_nodule_id', 0))
        model1_nodule_id = int(config.get('model1_nodule_id', 5))
        
        model1_nod = model_manager.get_model('model1-Nod')
        results1_nod_resized = model1_nod(resized_image, conf=0.1, iou=0.05)[0]
        results1_nod_processed = model1_nod(processed_image, conf=0.1, iou=0.05)[0]
        
        # Apply Bbox screening separately to the two sets of model1-Nod results
        results1_nod_resized_filtered = apply_bbox_screening(
            results1_nod_resized,
            target_width, new_height,
            nodule_class_id=model1_nod_nodule_id,
            nodule_class_name=nodule_class_name,
            size_threshold_ratio=size_threshold_ratio
        )
        results1_nod_processed_filtered = apply_bbox_screening(
            results1_nod_processed,
            target_width, new_height,
            nodule_class_id=model1_nod_nodule_id,
            nodule_class_name=nodule_class_name,
            size_threshold_ratio=size_threshold_ratio
        )

        # Fix model1-Nod's nodule class id from 0 to 5 (without name search)
        results1_nod_resized_fixed = rewrite_class_id_fixed(
            results1_nod_resized_filtered, results1_resized,
            src_class_id=model1_nod_nodule_id,
            dst_class_id=model1_nodule_id
        )
        results1_nod_processed_fixed = rewrite_class_id_fixed(
            results1_nod_processed_filtered, results1_resized,
            src_class_id=model1_nod_nodule_id,
            dst_class_id=model1_nodule_id
        )

        # Merge model1 and model1-Nod's 4 results together using NMS
        results_merged = merge_and_apply_nms_multi(
            [
                results1_resized,
                results1_processed,
                results1_nod_resized_fixed,
                results1_nod_processed_fixed,
            ],
            iou_threshold=iou_set,
            orig_shape=(original_height, original_width)
        )
        
        results['model1_results'] = results_merged
        results['model1_Nod_results'] = results_merged
        results['model1_confidences'] = extract_detection_confidence(
            results_merged, target_classes.get('model1', [])
        )
        results['model1_Nod_confidences'] = extract_detection_confidence(
            results_merged, target_classes.get('model1', [])
        )
        
        # Check trigger conditions
        trigger_activated, triggered_classes = check_trigger_condition(
            results['model1_confidences'], trigger_classes, trigger_threshold
        )
        results['trigger_activated'] = trigger_activated
        results['triggered_classes'] = triggered_classes
        
        # Conditionally trigger Model 2 and 3
        if trigger_activated:
            # Model 2 (TB Classification) uses image without CLAHE processing
            model2 = model_manager.get_model('model2')
            results2 = model2(resized_image)[0]
            results['model2_confidences'] = extract_classification_confidence(
                results2, target_classes.get('model2', [])
            )
            
            # Model 3 (Pulmonary Edema Classification) uses CLAHE processed image
            model3 = model_manager.get_model('model3')
            results3 = model3(processed_image)[0]
            results['model3_confidences'] = extract_classification_confidence(
                results3, target_classes.get('model3', [])
            )
        
        # Model 4 and 5 always execute - use CLAHE processed image
        # Model 4 (COPD Classification)
        model4 = model_manager.get_model('model4')
        results4 = model4(processed_image)[0]
        results['model4_confidences'] = extract_classification_confidence(
            results4, target_classes.get('model4', [])
        )
        
        # Model 5 (Pulmonary HTN Classification)
        model5 = model_manager.get_model('model5')
        results5 = model5(processed_image)[0]
        results['model5_confidences'] = extract_classification_confidence(
            results5, target_classes.get('model5', [])
        )
    
    return results


def save_yolo_annotations(results1, output_path, img_width, img_height, scale_factor=1.0):
    """Save YOLO format annotation txt file
    
    YOLO format: One object per line: class_id x_center y_center width height confidence (coordinates normalized 0~1, confidence is prediction confidence)
    
    Args:
        results1: YOLO prediction results (including boxes)
        output_path: txt output path (including filename)
        img_width: Original image width
        img_height: Original image height
        scale_factor: Coordinate scaling factor, used to map prediction boxes from scaled image back to original size
    
    Returns:
        bool: Whether any annotation was written
    """
    if results1.boxes is None or len(results1.boxes) == 0:
        # Write empty file when no detection results
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            pass
        return False
    
    lines = []
    for i, box in enumerate(results1.boxes):
        xyxy_array = box.xyxy[0].cpu().numpy()
        if xyxy_array.ndim == 2 and xyxy_array.shape[0] >= 1:
            xyxy_array = xyxy_array[0]
        if xyxy_array.size != 4:
            continue
        
        x1, y1, x2, y2 = xyxy_array.flatten()[:4]
        # Map coordinates from scaled image back to original size
        x1, y1, x2, y2 = x1 / scale_factor, y1 / scale_factor, x2 / scale_factor, y2 / scale_factor
        
        # Convert to YOLO normalized format (center_x, center_y, width, height)
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Ensure within 0~1 range
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.6f}")
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return len(lines) > 0


def save_detection_result(results1, output_path, original_image=None, scale_factor=None, save_to_disk=True):
    """Save object detection result image
    
    Args:
        results1: YOLO prediction results
        output_path: Output path
        original_image: Original size image (numpy array), if provided, draw on this image
        scale_factor: Scaling factor, used to map bounding boxes from scaled image back to original size
        save_to_disk: Whether to save to disk, default True; if False, only return annotated image, do not write to file
    
    Returns:
        numpy.ndarray: Annotated image
    """
    if original_image is not None and scale_factor is not None:
        # Draw results on the original size image
        annotated_img = original_image.copy()
        
        if results1.boxes is not None and len(results1.boxes) > 0:
            # Get class names and colors
            class_names = results1.names
            colors = Colors()  # YOLOv11 default colors
            
            # Calculate font size (dynamically adjust based on image height)
            img_height = annotated_img.shape[0]
            font_scale = max(1.0, img_height / 2000.0)  # Base font size, adjusted by image height
            font_thickness = max(2, int(font_scale * 2))  # Font thickness
            box_thickness = max(2, int(font_scale * 2))  # Box thickness
            
            # Collect all bounding box info and sort by confidence (from low to high, so highest is drawn last and appears on top)
            boxes_data = []
            for i, box in enumerate(results1.boxes):
                # Get scaled coordinates and convert back to original size
                # Ensure box.xyxy[0] returns a tensor of shape [4]
                xyxy_array = box.xyxy[0].cpu().numpy()
                # If xyxy_array shape is incorrect, try to reshape
                if xyxy_array.ndim == 0:
                    # Scalar (should not happen)
                    continue
                elif xyxy_array.ndim == 1 and xyxy_array.shape[0] == 4:
                    # Correct shape [4]
                    x1, y1, x2, y2 = xyxy_array / scale_factor
                elif xyxy_array.ndim == 2 and xyxy_array.shape[0] == 1 and xyxy_array.shape[1] == 4:
                    # Shape is [1, 4], take first
                    x1, y1, x2, y2 = xyxy_array[0] / scale_factor
                elif xyxy_array.size == 4:
                    # Try to reshape to [4]
                    xyxy_flat = xyxy_array.flatten()
                    if len(xyxy_flat) == 4:
                        x1, y1, x2, y2 = xyxy_flat / scale_factor
                    else:
                        continue
                else:
                    # Incorrect shape, skip this box
                    continue
                    
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = class_names[class_id]
                
                boxes_data.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'class_id': class_id,
                    'confidence': confidence,
                    'class_name': class_name
                })
            
            # Sort by confidence from low to high (highest confidence drawn last, appears on top)
            boxes_data.sort(key=lambda x: x['confidence'])
            
            # Draw each bounding box (in order of confidence from low to high)
            for box_data in boxes_data:
                x1 = box_data['x1']
                y1 = box_data['y1']
                x2 = box_data['x2']
                y2 = box_data['y2']
                class_id = box_data['class_id']
                confidence = box_data['confidence']
                class_name = box_data['class_name']
                
                # Use YOLOv11 default colors (Colors returns RGB, need to convert to BGR)
                color_rgb = colors(class_id)  # Returns RGB format color
                # Convert to BGR format (OpenCV uses BGR)
                color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
                
                # Draw bounding box
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color_bgr, box_thickness)
                
                # Draw label
                label = f'{class_name} {confidence:.2f}'
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                label_y = max(y1, label_height + 10)
                
                # Draw label background
                cv2.rectangle(annotated_img, 
                             (x1, label_y - label_height - baseline - 5), 
                             (x1 + label_width, label_y + baseline + 5), 
                             color_bgr, -1)
                
                # Determine text color based on class (light background uses black text, dark background uses white text)
                # Pneumothorax (Yellow), Fibrosis (Light Green), Calcification (White) use black text
                light_background_classes = ['Pneumothorax', 'Fibrosis', 'Calcification']
                # Case-insensitive matching
                if class_name.lower() in [c.lower() for c in light_background_classes]:
                    text_color = (0, 0, 0)  # Black
                else:
                    text_color = (255, 255, 255)  # White
                
                # Draw label text
                cv2.putText(annotated_img, label, (x1, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 
                           font_thickness, cv2.LINE_AA)
    else:
        # If original image is not provided, use the original method
        annotated_img = results1.plot()
    
    if save_to_disk:
        cv2.imwrite(output_path, annotated_img)
    return annotated_img


def prepare_model1_for_llm(model1_confidences):
    """Prepare Model 1 results to send to LLM
    
    Args:
        model1_confidences: Model 1 confidence dictionary
    
    Returns:
        dict: Processed result dictionary
    """
    model1_for_llm = {}
    for class_name, conf_data in model1_confidences.items():
        if isinstance(conf_data, list):
            model1_for_llm[class_name] = max(conf_data)
        else:
            model1_for_llm[class_name] = float(conf_data)
    return model1_for_llm
