transform = transforms.Compose([transforms.ToTensor()])
dataset = COCODataset(root='COCO/images/train2017', annotation='COCO/annotations/instances_train2017.json', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
