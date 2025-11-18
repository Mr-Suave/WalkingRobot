import cv2
import numpy as np

def load_image(file_path: str) -> np.ndarray:
    """
        This function takes in the file path as input which is absolute or relative path and then the bgr version of image is read from the image file
        and then the file is converted to rgb and stored as an np array of float32 type as deep learning libraries expect data in float32.
        Input: file_path
        Output: np.ndarray, each element type is np.float32
    """
    image_bgr = cv2.imread(file_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at {file_path}")
    image_np = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)

    # the above type of image_np is of np.uint8 0 to 255 which needs 8 bits to represent but many deep learning models wants them to be in float32
    image_np = image_np.astype(np.float32) / 255.0
    return image_np

def preprocess_image(image: np.ndarray, target_size = (256,256)) -> np.ndarray:
    """
    Resize and prepare the image for pose estimation models.

    Inputs:
        image (np.ndarray): Input RGB image array
        target_size (tuple): Desired (width, height)

    Outputs:
        np.ndarray: Preprocessed, resized image
    """
    resized = cv2.resize(image,target_size)
    return resized