# config.py

# ==================== File and Folder Settings ====================
# Set input folder path
INPUT_FOLDER = r".\Input"  # Please modify to your image folder path
OUTPUT_FOLDER = r".\Output"  # Result output folder (can be the same as input)

# Supported image formats
SUPPORTED_FORMATS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.dcm', '*.dicom']


# ==================== Model Path Settings ====================
MODEL_PATHS = {
    'model1': r"Best_models\CXR-Pattern-model_yolov11x.pt",
    'model1-Nod': r"Best_models\CXR-Pattern-model-nodule_yolov11x.pt",  # a dedicated Nod model
}

# ==================== Prediction Parameters Settings ====================
PREDICTION_CONFIG = {
    'conf_set': 0.01,           # Confidence threshold
    'iou_set': 0.2,            # IoU threshold
    
    # model1-Nod Bbox screening parameters
    'nodule_class_name': 'nodule',  # Nodule class name
    'size_threshold_ratio': 1/64,  # Bbox area to image area threshold ratio, remove nodule if exceeded
    
    # Define target class names for each model
    'target_classes': {
        'model1': ['Fracture', 'Atelectasis', 'Calcification', 'Cardiomegaly', 
                   'Opacity', 'Nodule', 'Effusion', 'Pneumothorax', 'Fibrosis'],
    }
}

# ==================== Batch Processing Settings ====================
BATCH_CONFIG = {
    'PROCESS_LIMIT': None,             # Set processing quantity limit, None means process all
    'SAVE_ANNOTATION_TXT': False,       # Whether to output YOLO annotation txt to Output/labels folder
    'SAVE_PREDICT_RESULT_IMAGE': True, # Whether to save YOLO predict result image (_result.jpg)
    'OUTPUT_IMAGE_FORMAT': 'jpg',      # Output result image format: 'jpg', 'png', 'dicom'
}