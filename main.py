# main.py
# Chest X-ray AI Analysis System - Main Program (Pure 9-pattern YOLO Object Detection Version)
# Preprocess with resize512 (maintain aspect ratio) and CLAHE (Contrast Limited Adaptive Histogram Equalization)
# model1 (Pattern) + model1-Nod: Perform detection on both 512-origin and 512-CLAHE. model1-Nod performs Bbox screening first.
# Total 4 results (model1/origin, model1/CLAHE, model1-Nod/origin, model1-Nod/CLAHE) integrated using NMS
# Added functionality to output YOLO format annotation + confidence txt files

import cv2
import os
import glob
from datetime import datetime
import traceback
import csv
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==================== Override cv2.imwrite to uniformly add Logo and Disclaimer ====================
_original_imwrite = cv2.imwrite

def custom_imwrite(filename, img, params=None):
    try:
        if img is None or not isinstance(img, np.ndarray):
            if params is None: return _original_imwrite(filename, img)
            return _original_imwrite(filename, img, params)
            
        # Check if it is a prediction image (only add to .jpg, .jpeg, .png, .dcm, or .dicom, filter out non-result images)
        if not (str(filename).lower().endswith('.jpg') or str(filename).lower().endswith('.png') or str(filename).lower().endswith('.jpeg') or str(filename).lower().endswith('.dcm') or str(filename).lower().endswith('.dicom')):
            if params is None: return _original_imwrite(filename, img)
            return _original_imwrite(filename, img, params)
            
        # Convert BGR to RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            if params is None: return _original_imwrite(filename, img)
            return _original_imwrite(filename, img, params)
            
        pil_img = Image.fromarray(img_rgb).convert("RGBA")
        
        # 1. Add Logo (Top Left)
        logo_path = r"C:\Users\tribb\Desktop\YOLO11-2025\6-model-integrate\logo_white_transparent.png"
        if os.path.exists(logo_path):
            try:
                logo = Image.open(logo_path).convert("RGBA")
                # Set Logo width to 20% of image width (limit max and min to avoid excessive distortion)
                logo_width = max(80, min(int(pil_img.width * 0.2), 500))
                logo_ratio = logo_width / float(logo.size[0])
                logo_height = int(float(logo.size[1]) * float(logo_ratio))
                logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
                
                padding = int(pil_img.width * 0.02)
                pil_img.paste(logo, (padding, padding), mask=logo)
            except Exception as logo_e:
                print(f"Failed to load Logo: {logo_e}")
            
        # 2. Add Disclaimer (Top Right)
        draw = ImageDraw.Draw(pil_img)
        text = "For research use only"
        font_size = max(8, int(pil_img.height * 0.02))
        try:
            # Try to load system font
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text size
        if hasattr(font, 'getbbox'):
            bbox = font.getbbox(text)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        elif hasattr(draw, 'textbbox'):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        else:
            text_w, text_h = draw.textsize(text, font=font)
            
        padding_x = int(pil_img.width * 0.02)
        padding_y = int(pil_img.height * 0.02)
        text_x = pil_img.width - text_w - padding_x
        text_y = padding_y
        
        # Draw black outline (improve readability)
        outline_color = (0, 0, 0, 255)
        outline_width = max(1, int(font_size * 0.08))
        for adj_x in range(-outline_width, outline_width + 1):
            for adj_y in range(-outline_width, outline_width + 1):
                if adj_x != 0 or adj_y != 0:
                    draw.text((text_x + adj_x, text_y + adj_y), text, font=font, fill=outline_color)
                    
        # Draw white text
        draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255, 255))
        
        # Convert back to BGR for cv2.imwrite to save
        final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGBA2BGR)
        
        # Handle DICOM saving
        if str(filename).lower().endswith('.dcm') or str(filename).lower().endswith('.dicom'):
            import pydicom
            from pydicom.dataset import FileDataset, FileMetaDataset
            from pydicom.uid import ExplicitVRLittleEndian
            import datetime
            
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7' # Secondary Capture Image Storage
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            
            ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
            ds.is_little_endian = True
            ds.is_implicit_VR = False
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.PatientName = "Anonymous"
            ds.PatientID = "123456"
            ds.Modality = "OT"
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()
            ds.StudyInstanceUID = pydicom.uid.generate_uid()
            ds.FrameOfReferenceUID = pydicom.uid.generate_uid()
            
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            
            rgb_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            ds.SamplesPerPixel = 3
            ds.PhotometricInterpretation = "RGB"
            ds.PlanarConfiguration = 0
            
            ds.Rows = rgb_img.shape[0]
            ds.Columns = rgb_img.shape[1]
            ds.PixelData = rgb_img.tobytes()
            
            dt = datetime.datetime.now()
            ds.StudyDate = dt.strftime('%Y%m%d')
            ds.StudyTime = dt.strftime('%H%M%S')
            
            ds.save_as(filename)
            return True

        if params is None:
            return _original_imwrite(filename, final_img)
        return _original_imwrite(filename, final_img, params)
    except Exception as e:
        print(f"Failed to add watermark ({filename}): {e}")
        import traceback
        traceback.print_exc()
        if params is None:
            return _original_imwrite(filename, img)
        return _original_imwrite(filename, img, params)

# Temporarily set, but will be reset again after all modules are imported to override ultralytics patch
cv2.imwrite = custom_imwrite
# =====================================================================================

# Import custom modules
from predict import (
    apply_bbox_screening, 
    rewrite_class_id_fixed, 
    merge_and_apply_nms_multi, 
    extract_detection_confidence,
    save_detection_result, 
    save_yolo_annotations
)
from image_utils import read_image

# Force override again, because ultralytics (imported in predict) will override cv2.imwrite
_original_imwrite = cv2.imwrite
cv2.imwrite = custom_imwrite

# Import custom settings
from config import *

class SimpleModelManager:
    """Simplified Model Manager: Only responsible for loading and managing YOLO model1 and model1-Nod"""
    def __init__(self, model_paths):
        self.model_paths = model_paths
        self.models = {}
        
    def load_all_models(self):
        print("Loading AI models...")
        try:
            from ultralytics import YOLO
            self.models['model1'] = YOLO(self.model_paths['model1'])
            self.models['model1-Nod'] = YOLO(self.model_paths['model1-Nod'])
            print("✓ Models loaded successfully (model1, model1-Nod)")
            return True
        except Exception as e:
            print(f"✗ Failed to load models: {str(e)}")
            return False
            
    def get_model(self, model_name):
        return self.models.get(model_name)

def get_all_image_files(folder_path):
    """Get all supported image files in the folder"""
    all_files = []
    for pattern in SUPPORTED_FORMATS:
        files = glob.glob(os.path.join(folder_path, pattern))
        all_files.extend(files)
    
    # Filter out images that are already result files
    filtered_files = []
    for file in all_files:
        basename = os.path.basename(file)
        if not any(suffix in basename for suffix in ['_result.']):
            filtered_files.append(file)
    
    # Remove duplicates and sort
    filtered_files = sorted(list(set(filtered_files)))
    return filtered_files

def extract_prediction_results_for_csv(model1_confidences, base_filename):
    """Extract prediction results and format as a dictionary for CSV"""
    from predict import get_confidence_value
    
    result = {
        'Original_Filename': base_filename,
        'Fracture': get_confidence_value(model1_confidences, 'Fracture', 0.0),
        'Atelectasis': get_confidence_value(model1_confidences, 'Atelectasis', 0.0),
        'Calcification': get_confidence_value(model1_confidences, 'Calcification', 0.0),
        'Cardiomegaly': get_confidence_value(model1_confidences, 'Cardiomegaly', 0.0),
        'Opacity': get_confidence_value(model1_confidences, 'Opacity', 0.0),
        'Nodule': get_confidence_value(model1_confidences, 'Nodule', 0.0),
        'Effusion': get_confidence_value(model1_confidences, 'Effusion', 0.0),
        'Pneumothorax': get_confidence_value(model1_confidences, 'Pneumothorax', 0.0),
        'Fibrosis': get_confidence_value(model1_confidences, 'Fibrosis', 0.0)
    }
    return result

def run_pattern_prediction(image, model_manager, config):
    """Execute 9-pattern YOLO prediction process"""
    conf_set = config.get('conf_set', 0.01)
    iou_set = config.get('iou_set', 0.05)
    target_classes = config.get('target_classes', {}).get('model1', [
        'Fracture', 'Atelectasis', 'Calcification', 'Cardiomegaly', 
        'Opacity', 'Nodule', 'Effusion', 'Pneumothorax', 'Fibrosis'
    ])
    
    # Save original image dimensions
    original_height, original_width = image.shape[:2]
    original_image = image.copy()
    
    # Resize image to 512 pixels width (maintain aspect ratio)
    target_width = 512
    scale_factor = target_width / original_width
    new_height = int(original_height * scale_factor)
    
    pil_image = Image.fromarray(image).convert("RGB")
    resized_pil = pil_image.resize((target_width, new_height), Image.Resampling.LANCZOS)
    resized_image = np.array(resized_pil)
    
    # Apply CLAHE processing
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
            
    # Model 1 prediction
    model1 = model_manager.get_model('model1')
    results1_resized = model1(resized_image, conf=conf_set, iou=iou_set)[0]
    results1_processed = model1(processed_image, conf=conf_set, iou=iou_set)[0]
    
    # Model 1-Nod prediction
    nodule_class_name = config.get('nodule_class_name', 'nodule')
    size_threshold_ratio = config.get('size_threshold_ratio', 1/64)
    model1_nod_nodule_id = int(config.get('model1_nod_nodule_id', 0))
    model1_nodule_id = int(config.get('model1_nodule_id', 5))
    
    model1_nod = model_manager.get_model('model1-Nod')
    results1_nod_resized = model1_nod(resized_image, conf=0.1, iou=0.05)[0]
    results1_nod_processed = model1_nod(processed_image, conf=0.1, iou=0.05)[0]
    
    # Bbox screening
    results1_nod_resized_filtered = apply_bbox_screening(
        results1_nod_resized, target_width, new_height,
        nodule_class_id=model1_nod_nodule_id, nodule_class_name=nodule_class_name,
        size_threshold_ratio=size_threshold_ratio
    )
    results1_nod_processed_filtered = apply_bbox_screening(
        results1_nod_processed, target_width, new_height,
        nodule_class_id=model1_nod_nodule_id, nodule_class_name=nodule_class_name,
        size_threshold_ratio=size_threshold_ratio
    )

    # Rewrite class id
    results1_nod_resized_fixed = rewrite_class_id_fixed(
        results1_nod_resized_filtered, results1_resized,
        src_class_id=model1_nod_nodule_id, dst_class_id=model1_nodule_id
    )
    results1_nod_processed_fixed = rewrite_class_id_fixed(
        results1_nod_processed_filtered, results1_resized,
        src_class_id=model1_nod_nodule_id, dst_class_id=model1_nodule_id
    )

    # Merge and NMS
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
    
    model1_confidences = extract_detection_confidence(results_merged, target_classes)
    
    return {
        'model1_results': results_merged,
        'model1_confidences': model1_confidences,
        'original_image': original_image,
        'original_size': (original_width, original_height),
        'scale_factor': scale_factor
    }

def process_single_image(img_path, output_dir, model_manager):
    """Complete process for a single image"""
    try:
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        image = read_image(img_path)
        if image is None:
            return False, f"Failed to read image: {img_path}", None
        
        out_ext = BATCH_CONFIG.get('OUTPUT_IMAGE_FORMAT', 'jpg').lower()
        if out_ext not in ['jpg', 'jpeg', 'png', 'dicom', 'dcm']:
            out_ext = 'jpg'
        if out_ext == 'dicom':
            out_ext = 'dcm'
        
        # Execute 9-pattern prediction
        prediction_results = run_pattern_prediction(image, model_manager, PREDICTION_CONFIG)
        
        # Save YOLO format annotation txt
        if BATCH_CONFIG.get('SAVE_ANNOTATION_TXT', True):
            labels_dir = os.path.join(output_dir, "labels")
            labels_path = os.path.join(labels_dir, f"{base_filename}.txt")
            orig_width, orig_height = prediction_results['original_size']
            save_yolo_annotations(
                prediction_results['model1_results'],
                labels_path,
                img_width=orig_width,
                img_height=orig_height,
                scale_factor=prediction_results['scale_factor']
            )
            
        # Draw and save results
        save_result_image = BATCH_CONFIG.get('SAVE_PREDICT_RESULT_IMAGE', True)
        if save_result_image:
            result_path = os.path.join(output_dir, f"{base_filename}_result.{out_ext}")
            save_detection_result(
                prediction_results['model1_results'], 
                result_path,
                original_image=prediction_results['original_image'],
                scale_factor=prediction_results['scale_factor'],
                save_to_disk=True
            )
            
        csv_result = extract_prediction_results_for_csv(prediction_results['model1_confidences'], base_filename)
        return True, "Success", csv_result
            
    except Exception as e:
        return False, str(e), None

def batch_process_images():
    """Main function for batch processing"""
    start_time = datetime.now()
    
    print("="*70)
    print("CXR 9-Pattern YOLOv11 Object Detection - Batch Processing Mode")
    print("="*70)
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"✗ Error: Input folder does not exist: {INPUT_FOLDER}")
        return
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Only load model1 and model1-Nod
    models_to_load = {
        'model1': MODEL_PATHS.get('model1'),
        'model1-Nod': MODEL_PATHS.get('model1-Nod')
    }
    model_manager = SimpleModelManager(models_to_load)
    if not model_manager.load_all_models():
        print("✗ Failed to load models, program terminated")
        return

    image_files = get_all_image_files(INPUT_FOLDER)
    
    if not image_files:
        print(f"✗ No image files found in folder {INPUT_FOLDER}")
        return
    
    if BATCH_CONFIG.get('PROCESS_LIMIT'):
        image_files = image_files[:BATCH_CONFIG['PROCESS_LIMIT']]
    
    total_count = len(image_files)
    print(f"\nFound {total_count} image files to process")
    print(f"Input Folder: {INPUT_FOLDER}")
    print(f"Output Folder: {OUTPUT_FOLDER}")
    print("-"*70)
    
    log_file_path = os.path.join(OUTPUT_FOLDER, f"batch_log_{start_time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    success_count = 0
    error_count = 0
    error_files = []
    csv_results = []
    
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"Batch processing started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total files: {total_count}\n")
        log_file.write("="*70 + "\n\n")
        
        for idx, img_path in enumerate(image_files, 1):
            file_start_time = datetime.now()
            print(f"\n[{idx}/{total_count}] Processing: {os.path.basename(img_path)}")
            
            success, message, csv_result = process_single_image(img_path, OUTPUT_FOLDER, model_manager)
            
            file_end_time = datetime.now()
            processing_time = (file_end_time - file_start_time).total_seconds()
            
            if success:
                success_count += 1
                status = "✓"
                log_file.write(f"[{idx:3d}] ✓ {os.path.basename(img_path)} - Success ({processing_time:.2f}s)\n")
                if csv_result is not None:
                    csv_results.append(csv_result)
            else:
                error_count += 1
                error_files.append(os.path.basename(img_path))
                status = "✗"
                log_file.write(f"[{idx:3d}] ✗ {os.path.basename(img_path)} - Failed: {message} ({processing_time:.2f}s)\n")
            
            progress = (idx / total_count) * 100
            print(f"    {status} Status: {message} (Time: {processing_time:.2f}s)")
            print(f"    Progress: {progress:.1f}% ({success_count} Success, {error_count} Failed)")
            log_file.flush()
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        log_file.write("\n" + "="*70 + "\n")
        log_file.write(f"Batch processing finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total processing time: {total_time:.2f} s ({total_time/60:.2f} min)\n")
        log_file.write(f"Successfully processed: {success_count} files\n")
        log_file.write(f"Failed to process: {error_count} files\n")
        
        if error_files:
            log_file.write("\nFailed files list:\n")
            for error_file in error_files:
                log_file.write(f"  - {error_file}\n")
    
    if csv_results:
        csv_file_path = os.path.join(OUTPUT_FOLDER, f"prediction_results_{start_time.strftime('%Y%m%d_%H%M%S')}.csv")
        fieldnames = [
            'Original_Filename', 'Fracture', 'Atelectasis', 'Calcification',
            'Cardiomegaly', 'Opacity', 'Nodule', 'Effusion', 'Pneumothorax',
            'Fibrosis'
        ]
        try:
            with open(csv_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_results)
            print(f"\n✓ Prediction results exported to: {csv_file_path}")
        except Exception as e:
            print(f"\n✗ CSV export failed: {str(e)}")
    
    print("\n" + "="*70)
    print("Batch processing complete!")
    print("="*70)
    print(f"Total processed: {total_count}")
    print(f"Success: {success_count} ({(success_count/total_count*100):.1f}%)")
    print(f"Failed: {error_count} ({(error_count/total_count*100):.1f}%)")
    print(f"Total time: {total_time:.2f} s ({total_time/60:.2f} min)")
    
    if total_count > 0:
        avg_time = total_time / total_count
        print(f"Average time: {avg_time:.2f} s/image")
    
    print(f"\nResults saved in: {OUTPUT_FOLDER}")
    print(f"Processing log: {log_file_path}")
    if error_files:
        print(f"\n⚠ {len(error_files)} files failed to process, see log file for details")

if __name__ == "__main__":
    try:
        batch_process_images()
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
    except Exception as e:
        print(f"\n✗ Unexpected error occurred: {str(e)}")
        print(traceback.format_exc())