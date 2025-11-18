from ultralytics.data.annotator import auto_annotate


# img_path = "./scripts/images/korea-4643876_1280.jpg"
# img_path = "./scripts/images/seoul-2713200_1280.jpg"
img_path = "/home/seoyc/Downloads/former0045/former0045_image_20250421_055838.jpg"

auto_annotate(data=img_path, det_model="yolo11x.pt", sam_model="sam2.1_b.pt")
