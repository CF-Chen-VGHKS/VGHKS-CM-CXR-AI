VGHKS-CM CXR-AI
# CXR 9-Pattern YOLO Object Detection

## Overview
This project is an AI-powered Chest X-ray (CXR) analysis system based on the YOLOv11 object detection framework. It is designed to automatically detect and localize 9 specific radiographic patterns in chest X-rays.

### Features
* **9-Pattern Detection**: Identifies Fracture, Atelectasis, Calcification, Cardiomegaly, Opacity, Nodule, Effusion, Pneumothorax, and Fibrosis.
* **Dual-Model Ensemble**: Utilizes a primary pattern model (`model1`) and a specialized nodule model (`model1-Nod`).
* **Dual-Preprocessing Inference**: Performs inference on both the original resized image and a CLAHE (Contrast Limited Adaptive Histogram Equalization) enhanced image to maximize detection sensitivity.
* **Advanced Nodule Screening**: Automatically filters out false-positive nodules that exceed a specific size threshold (e.g., > 1/64 of the image area).
* **NMS Integration**: Merges 4 sets of predictions (model1/origin, model1/CLAHE, model1-Nod/origin, model1-Nod/CLAHE) using Non-Maximum Suppression (NMS).
* **Comprehensive Outputs**: Generates annotated images (with watermarks), YOLO-format `.txt` annotations, and a `.csv` summary report containing the highest confidence scores for each pattern.
* **Broad Format Support**: Supports standard image formats (JPG, PNG, BMP, TIFF) as well as DICOM (`.dcm`, `.dicom`).

---

## Installation

1. **Clone or download the repository** to your local machine.
2. **Install the required dependencies** using `pip`:

```bash
pip install -r requirements.txt
```

> **Note on PyTorch (GPU Acceleration):** 
> To utilize GPU acceleration, it is highly recommended to install the CUDA-enabled version of PyTorch that matches your system's hardware. Please visit the [PyTorch Official Website](https://pytorch.org/get-started/locally/) for the specific installation command.

---

## Configuration (`config.py`)

Before running the script, adjust the parameters in `config.py` to match your environment and requirements:

### File and Folder Settings
* `INPUT_FOLDER`: Directory path containing the input CXR images.
* `OUTPUT_FOLDER`: Directory path where the results will be saved.

### Model Paths
* `MODEL_PATHS`: Dictionary defining the paths to the YOLO `.pt` weight files (`model1` and `model1-Nod`).

### Prediction Parameters (`PREDICTION_CONFIG`)
* `conf_set`: Confidence threshold for object detection (default: `0.01`).
* `iou_set`: Intersection over Union (IoU) threshold for NMS merging (default: `0.2`).
* `size_threshold_ratio`: The maximum area ratio allowed for a nodule bounding box relative to the entire image. Bboxes exceeding this (default: `1/64`) are removed.

### Batch Processing Settings (`BATCH_CONFIG`)
* `PROCESS_LIMIT`: Maximum number of images to process (set to `None` to process all).
* `SAVE_ANNOTATION_TXT`: Set to `True` to export bounding boxes in YOLO `.txt` format.
* `SAVE_PREDICT_RESULT_IMAGE`: Set to `True` to save the annotated output images.
* `OUTPUT_IMAGE_FORMAT`: Format for the output images (`'jpg'`, `'png'`, or `'dicom'`).

---

## Usage

Once your `config.py` is properly set up and your models are placed in the correct directories, you can start the batch processing by running:

```bash
python main.py
```

### Output Structure
After execution, the `OUTPUT_FOLDER` will contain:
1. **Annotated Images**: e.g., `image_name_result.jpg` (includes bounding boxes, confidence scores, and a research-only watermark).
2. **CSV Report**: `prediction_results_YYYYMMDD_HHMMSS.csv` containing the maximum confidence score for each of the 9 patterns per image.
3. **Log File**: `batch_log_YYYYMMDD_HHMMSS.txt` detailing the processing time and status of each image.
4. **Labels Folder** *(Optional)*: If `SAVE_ANNOTATION_TXT` is enabled, YOLO format text files will be saved in an `Output/labels` subfolder.


⚠️ **DISCLAIMER**
> **This model is developed purely for academic research and educational purposes. It should NOT be used for clinical diagnosis or treatment decisions.**

## License
This software, including its code and associated model weights, is licensed for **Research Only**. It is **NOT for commercial use**.

## Citation
If you use this software or model in your research, please cite our upcoming publication (currently in preparation/under review). 
*(Citation details will be updated once the manuscript is finalized and published.)*
