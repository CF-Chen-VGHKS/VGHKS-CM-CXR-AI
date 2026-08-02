import cv2
import numpy as np
import os

def read_image(image_path):
    """
    Read image file, supporting common formats (jpg, png, bmp) and DICOM (dcm, dicom).
    Returns a numpy array in BGR format (same as cv2.imread default behavior).
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.dcm', '.dicom']:
        try:
            import pydicom
            dicom_data = pydicom.dcmread(image_path)
            pixel_array = dicom_data.pixel_array
            
            # Normalize to 0-255 and convert to uint8
            if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
                slope = float(dicom_data.RescaleSlope)
                intercept = float(dicom_data.RescaleIntercept)
                pixel_array = pixel_array * slope + intercept
                
            # Window Center and Window Width
            if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
                wc = dicom_data.WindowCenter
                ww = dicom_data.WindowWidth
                if isinstance(wc, pydicom.multival.MultiValue): wc = wc[0]
                if isinstance(ww, pydicom.multival.MultiValue): ww = ww[0]
                wc = float(wc)
                ww = float(ww)
                vmin = wc - ww / 2.0
                vmax = wc + ww / 2.0
                pixel_array = np.clip(pixel_array, vmin, vmax)
                if vmax - vmin > 0:
                    pixel_array = (pixel_array - vmin) / (vmax - vmin) * 255.0
                else:
                    pixel_array = np.zeros_like(pixel_array)
            else:
                p_min = np.min(pixel_array)
                p_max = np.max(pixel_array)
                pixel_array = pixel_array - p_min
                if p_max - p_min > 0:
                    pixel_array = pixel_array / (p_max - p_min) * 255.0
            
            # Photometric Interpretation
            if hasattr(dicom_data, 'PhotometricInterpretation'):
                if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                    pixel_array = 255.0 - pixel_array
                
            image = pixel_array.astype(np.uint8)
            
            # Convert to BGR
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # If DICOM is RGB, convert to BGR for OpenCV consistency
                if getattr(dicom_data, 'PhotometricInterpretation', '') == 'RGB':
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            print(f"Failed to read DICOM {image_path}: {e}")
            return None
    else:
        # Use cv2.imdecode to support paths with non-ASCII characters
        try:
            with open(image_path, 'rb') as f:
                img_array = np.frombuffer(f.read(), np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            print(f"Failed to read image {image_path}: {e}")
            return None