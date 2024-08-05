import json

from qwak.model.base import QwakModel
from qwak.model.tools import run_local
from huggingface_hub import snapshot_download
from ultralytics import YOLO
import random
import cv2


class FrogFactorAuthenticator(QwakModel):
    
    def __init__(self):
        weights = "yolov8m-oiv7.pt"
        snapshot_download(
            repo_id="frog-factor1", revision="main", allow_patterns=weights, local_dir="."
        )
        self._model = YOLO(weights)

    def overlay_image_alpha(img, img_overlay, pos):
        """Overlay img_overlay on top of img at the position specified by pos."""
        x, y = pos

        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        y1, y2 = int(y1), int(y2)
        x1, x2 = int(x1), int(x2)
        y1o, y2o = int(y1o), int(y2o)
        x1o, x2o = int(x1o), int(x2o)
        
        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return img

        # Blend overlay within the determined ranges
        img_crop = img[y1:y2, x1:x2]
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]

        # Split the alpha channel and the color channels
        if img_overlay_crop.shape[2] == 4:  # Ensure the overlay has an alpha channel
            img_overlay_color = img_overlay_crop[:, :, :3]
            alpha_mask = img_overlay_crop[:, :, 3] / 255.0

            alpha_inv = 1.0 - alpha_mask

            for c in range(0, 3):
                img_crop[:, :, c] = (alpha_mask * img_overlay_color[:, :, c] +
                                    alpha_inv * img_crop[:, :, c])
        else:
            img_crop[:, :, :] = img_overlay_crop

        return img


    def build(self):
        pass

    def predict(self, frame) -> np.ndarray:

        # Load the frog image with alpha channel
        config_file = "./config.json"
      
        with open(config_file, 'r') as f:
            config = json.load(f)

        classes        = config['classes']
        target_classes = config['target_classes']

        frame_height, frame_width = frame.shape[:2]
        frog_x = frame_width  # Start position for the frog (outside the screen on the right)
        # Set the frog image at the top of the screen
        frog_y = 0    

        results = self._model.predict(source=frame, 
                                      show=False, 
                                      classes=classes, 
                                      conf=0.05,
                                      max_det=2)

        return results


if __name__ == "__main__":
    cv = ImageClassifier()
    img = Image.open('cat.jpeg')
    img_ndarray = np.array(img)
    img_list = img_ndarray.tolist()
    img_json1 = json.dumps(img_list)  # This is the JSON string
    test_model = run_local(cv, img_json1)  # Pass the JSON string directly
    print(test_model)
