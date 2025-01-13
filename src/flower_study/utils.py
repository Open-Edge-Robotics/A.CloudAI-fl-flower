import glob
import os

import keras
import numpy as np
from logging import INFO
import tensorflow_datasets as tfds
from flwr.common.logger import log
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = keras.models.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            # keras.Input(shape=(200, 200, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="Adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def load_model_with_checkpoint():
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = keras.models.Sequential(
        [
            # keras.Input(shape=(32, 32, 3)),
            keras.Input(shape=(200, 200, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="Adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # get checkpoint in npz files
    checkpoint_path: str = os.path.join(os.getcwd(), "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    if os.getenv("KUBERNETES_SERVICE_HOST") is not None:
        checkpoint_path = "/mnt/checkpoints/"

    list_of_files = [fname for fname in glob.glob(checkpoint_path + "/*.npz")]

    log(INFO, f"List of files: {list_of_files}")

    if len(list_of_files) > 0:
        latest_file = max(list_of_files, key=os.path.getctime)
        log(INFO, f"Latest file: {latest_file}")
        checkpoint_file_path = os.path.join(checkpoint_path, latest_file)

        if os.path.exists(checkpoint_file_path):
            print(f"Loading checkpoint from {checkpoint_file_path}")
            weights = np.load(checkpoint_file_path)
            model.set_weights(list(weights.values()))

    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition_split = partition.train_test_split(test_size=0.2)
    x_train, y_train = (
        partition_split["train"]["img"] / 255.0,  # type: ignore
        partition_split["train"]["label"],
    )
    x_test, y_test = (
        partition_split["test"]["img"] / 255.0,  # type: ignore
        partition_split["test"]["label"],
    )

    return x_train, y_train, x_test, y_test


def load_custom_dataset(id: str):
    dataset_dir = f"/home/seoyc/tensorflow_datasets/robot_dataset_{id}/0.0.1"
    if os.getenv("KUBERNETES_SERVICE_HOST") is not None:
        dataset_dir = f"/dataset/robot_dataset_{id}/0.0.1"

    builder = tfds.builder_from_directory(dataset_dir)
    if builder is None:
        print("Failed to load dataset from directory: ", dataset_dir)
        return
    # builder.download_and_prepare()
    dataset = builder.as_dataset()

    train, test = dataset["train"], dataset["test"]  # type: ignore
    train = train.map(lambda x: (x["image"], x["label"]))
    test = test.map(lambda x: (x["image"], x["label"]))
    train = train.shuffle(1000).batch(32).prefetch(1)
    test = test.batch(32).prefetch(1)

    x_train, y_train = train.as_numpy_iterator().next()
    x_test, y_test = test.as_numpy_iterator().next()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test
