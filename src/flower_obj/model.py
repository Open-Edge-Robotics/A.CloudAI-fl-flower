from collections import OrderedDict
import os

import torch
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision

from torch.utils.data import DataLoader
from torchvision.models.detection import MaskRCNN

from dataset import PennFudanDataset
from libs.coco_eval import CocoEvaluator
from libs.utils import collate_fn
from libs.engine import train_one_epoch, evaluate


class RCNNModel:
    def get_pretrained_model(self):
        # load a model pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        # replace the classifier with a new one, that has
        # num_classes which is user-defined
        num_classes = 2  # 1 class (person) + background
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

    def get_model_instance_segmentation(self, num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        return model

    def forward(self, x):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        dataset = PennFudanDataset("data/PennFudanPed", get_transform(train=True))
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, collate_fn=collate_fn
        )

        # For Training
        images, targets = next(iter(data_loader))
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        output = model(images, targets)  # Returns losses and detections
        print(output)

        # For inference
        model.eval()
        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        predictions = model(x)  # Returns predictions
        print(predictions[0])


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def load_data(batch_size: int):
    data_path = os.path.join(os.getcwd(), "data/PennFudanPed")
    dataset = PennFudanDataset(data_path, get_transform(train=True))
    dataset_test = PennFudanDataset(data_path, get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return data_loader, data_loader_test


def train(model, trainloader, valloader, epochs, learning_rate, device) -> None:
    """Train the model."""
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    for epoch in range(epochs):
        metric = train_one_epoch(
            model, optimizer, trainloader, device, epoch, print_freq=10
        )
        lr_scheduler.step()

    return metric.meters["loss"].value


def eval(model: MaskRCNN, testloader: DataLoader, device: torch.device):
    """Validate the model on the test set."""
    model.to(device)
    result: CocoEvaluator = evaluate(model, testloader, device=device)

    return result.coco_eval["bbox"].stats[0], result.coco_eval["bbox"].stats[1]
