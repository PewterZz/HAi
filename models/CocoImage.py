import os
import torch
import torchvision
from pycocotools.coco import COCO
from torchvision import transforms
from PIL import Image

class CustomTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img, target):
        img = self.transform(img)
        return img, target

# Define the transformation, including resizing and tensor conversion
transform = CustomTransform(transforms.Compose([
    transforms.Resize((1280, 720)),  # Resize all images to 1280x720 pixels
    transforms.ToTensor()
]))


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transform=None):
        self.root = root
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        if self.transform:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

# Define your transformations
transform = CustomTransform(transforms.ToTensor())

# Then create your dataset and dataloader
dataset = COCODataset(root='data/images/train2017', annotation='data/annotations/stuff_train2017.json', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


# Define the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None) 

# Define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for images, targets in dataloader:
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()