import fire
import numpy as np
import matplotlib.pyplot as plt

import os
import glob

from utils import load_custom_dataset, load_data, load_model_with_checkpoint


def __draw_image(count: int, **kwargs):
    fig = plt.figure(figsize=(14, 14))

    col = 4
    row = count // col

    if (
        "x_test" not in kwargs
        or "predictions" not in kwargs
        or "labels" not in kwargs
        or count <= 0
    ):
        print("x_test, predictions, and labels are required.")
        return

    for i in range(count):
        fig.add_subplot(col, row, i + 1)
        plt.imshow(kwargs["x_test"][i])
        plt.axis("off")
        pred = np.argmax(kwargs["predictions"][i])
        plt.title(f'Predicted: {kwargs["labels"][pred]}')

    plt.show()


def __get_model_weights_from_checkpoint(model):
    # get checkpoint in npz files
    checkpoint_path: str = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if os.getenv("KUBERNETES_SERVICE_HOST") is not None:
        checkpoint_path = "/mnt/checkpoints/"

    list_of_files = [fname for fname in glob.glob(checkpoint_path + "/*.npz")]

    print(f"List of files: {list_of_files}")

    if len(list_of_files) > 0:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Latest file: {latest_file}")
        checkpoint_file_path = os.path.join(checkpoint_path, latest_file)

        if os.path.exists(checkpoint_file_path):
            print(f"Loading checkpoint from {checkpoint_file_path}")
            weights = np.load(checkpoint_file_path)
            model.set_weights(list(weights.values()))

    return model


def predict():
    model = load_model_with_checkpoint()
    __get_model_weights_from_checkpoint(model)

    # Load test data
    _, _, x_test, y_test = load_data(0, 1)

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose="auto")
    labels = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    # Predict the first 10 test images
    predictions = model.predict(x_test[:10])
    print("Predictions:")
    print(predictions)

    __draw_image(
        10,
        x_test=x_test[:10],
        predictions=predictions,
        labels=labels,
    )


def predict_custom_dataset():
    model = load_model_with_checkpoint()
    __get_model_weights_from_checkpoint(model)

    # Load test data
    data = load_custom_dataset("1")
    if data is None:
        return

    x_test = data[2]
    y_test = data[3]

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test, verbose="auto")
    labels = ["circle", "square", "star", "triangle"]
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    # Predict the first 10 test images
    predictions = model.predict(x_test[:20])
    print("Predictions:")
    print(predictions)

    __draw_image(
        20,
        x_test=x_test[:20],
        predictions=predictions,
        labels=labels,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "predict": predict,
            "predict_custom_dataset": predict_custom_dataset,
        }
    )
