import json
import numpy as np
import pandas as pd
import qwak
from huggingface_hub import snapshot_download
from pandas import DataFrame
from qwak.model.adapters import NumpyInputAdapter
from qwak.model.base import QwakModel
from qwak.model.tools import run_local
from ultralytics import YOLO


class FrogFactorAuthenticator(QwakModel):

    def __init__(self, config_path: str = None):
        self._model = None
        self.weights_file_name = "yolov8m-oiv7.pt"
        self.repo_id = "frog-factor1"

        # Use the default path of pass external config values
        config_path = config_path or "main/config.json"
        self.config = self.read_config(config_path)

    @staticmethod
    def read_config(file_path):
        with open(file_path, 'r') as f:
            config = json.load(f)
        return config

    def build(self):
        snapshot_download(
            repo_id=self.repo_id,
            revision="main",
            allow_patterns=self.weights_file_name,
            local_dir=".",
        )
        self._model = YOLO(self.weights_file_name)

    @staticmethod
    def preprocess_image(img_ndarray):
        
        if img_ndarray.dtype != np.float32:
            img_ndarray = img_ndarray.astype('float32')

        return img_ndarray

    @qwak.api(input_adapter=NumpyInputAdapter())
    def predict(self, img_ndarray: np.ndarray) -> pd.DataFrame:
        # The frame is wrapped in a list of size 1
        frame = img_ndarray[0]
        # Basic cleaning
        new_frame = self.preprocess_image(frame)
        results = self._model.predict(
            source=new_frame,
            show=False,
            classes=self.config['classes'],
            conf=self.config['conf'],
            max_det=self.config['max_det']
        )
        print(results)
        return DataFrame([
            {"results": json.loads(results[0].tojson())}
        ])


if __name__ == "__main__":
    from PIL import Image

    cv = FrogFactorAuthenticator(config_path="./config.json")
    img = Image.open('./img/tom.jpg')
    image_rgb = img.convert('RGB')
    img_ndarray = np.array(image_rgb)
    img_list = img_ndarray.tolist()
    img_json = json.dumps(img_list)  # This is the JSON string

    test_model = run_local(cv, img_json)  # Pass the JSON string directly
    print(test_model)