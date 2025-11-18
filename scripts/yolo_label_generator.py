import os

import fire
from ultralytics.data.annotator import auto_annotate


class YoloLabelGenerator:
    def __init__(
        self, model_name: str, seg_model_name: str, path: str, output: str
    ) -> None:
        self.model = model_name
        self.sam_model = seg_model_name
        self.target_path = path
        self.output_path = output

    def generate_output(self) -> str:
        current_path = os.path.curdir
        output_dir = os.path.join(current_path, self.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def generate_label_txt(self) -> None:
        files = os.listdir(self.target_path)
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                file_path = os.path.join(self.target_path, file)
                auto_annotate(
                    data=file_path,
                    det_model=self.model,
                    sam_model=self.sam_model,
                    output_dir=self.generate_output(),
                )


def run(
    model_name: str = "yolo11n.pt",
    seg_model_name: str = "sam2.1_b.pt",
    path: str = "",
    output: str = "",
):
    yolo_label_generator = YoloLabelGenerator(model_name, seg_model_name, path, output)
    yolo_label_generator.generate_label_txt()


if __name__ == "__main__":
    fire.Fire(
        {
            "run": run,
        }
    )
