import os
import cv2
import fire
from PIL import Image

from ultralytics.engine.results import Results

from utils import load_model_with_checkpoint
from __init__ import class_names, class_color


DEFAULT_MODEL = "former_best.pt"


class YoloPredict:
    def __init__(
        self, model_name: str, path: str, output_path: str, is_save: bool
    ) -> None:
        # self.model = YOLO(model_name)
        self.model = load_model_with_checkpoint(model_name)
        self.target_path = path
        self.output_path = output_path
        self.is_save = is_save

    def __save(self, result: Results, file_name: str):
        if self.is_save:
            result.save(filename=f"{file_name}_result.jpg")

    def __draw(self, result: Results):
        img = result.orig_img.copy()
        if result.boxes is None:
            return

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            obj_name = class_names[cls_id]
            conf = box.conf[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), class_color[cls_id], 2)
            cv2.putText(
                img,
                f"{obj_name}, {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                class_color[cls_id],
                2,
            )

        im_rgb = Image.fromarray(img[..., ::-1])
        im_rgb.show()

    def predict(self):
        files = os.listdir(self.target_path)
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                file_path = os.path.join(self.target_path, file)
                results = self.model(file_path)
                for i, r in enumerate(results):
                    self.__draw(r)
                    self.__save(r, file)


def predict(
    model_name: str = DEFAULT_MODEL,
    path: str = "",
    output_path: str = "",
    is_save: bool = False,
):
    yolo_predict = YoloPredict(model_name, path, output_path, is_save)
    yolo_predict.predict()


if __name__ == "__main__":
    fire.Fire({"predict": predict})
