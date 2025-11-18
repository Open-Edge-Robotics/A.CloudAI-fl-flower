import os
import fire
from ultralytics.models import SAM
from ultralytics.engine.results import Results
from PIL import Image


class SamTest:
    def __init__(
        self, model_name: str, path: str, is_save: bool, save_dir: str
    ) -> None:
        self.sam_model = SAM(model_name)
        self.target_path = path
        self.is_save = is_save
        self.save_dir = save_dir

        self.sam_model.info()

    def __generate_output(self) -> str:
        current_path = os.path.curdir
        save_dir = os.path.join(current_path, self.save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir

    def __draw(self, result: Results) -> None:
        im_bgr = result.plot()
        im_rgb = Image.fromarray(im_bgr[..., ::-1])
        im_rgb.show()

        if self.is_save:
            save_dir = self.__generate_output()
            result.save(save_dir)

    def image_segment(self) -> None:
        files = os.listdir(self.target_path)
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(self.target_path, file)
                results = self.sam_model(img_path)
                for result in results:
                    self.__draw(result)


def run(
    model_name: str = "sam2.1_b.pt",
    path: str = "",
    is_save: bool = False,
    save_dir: str = "",
) -> None:
    sam_test = SamTest(model_name, path, is_save, save_dir)
    sam_test.image_segment()


if __name__ == "__main__":
    fire.Fire({"run": run})
