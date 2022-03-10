from urllib.parse import urlparse

import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import pandas as pd
from PIL import Image
from efficientnet_pytorch import EfficientNet
# import mlflow.projects.kubernetes
import boto3
# import kubernetes
import mlflow.projects
import mlflow.pytorch




class MultiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = (self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        # 5 for 2nd level label
        # 7 for 1st level label
        y1 = self.data_frame.iloc[idx, 7]
        y2 = self.data_frame.iloc[idx, 5]
        age = self.data_frame.iloc[idx, 1]
        # age = torch.FloatTensor(age)
        gender = self.data_frame.iloc[idx, 2]
        # gender = torch.FloatTensor(gender)

        if self.transform:
            image = self.transform(image)

        return image, y1, y2, age, gender, img_name

normalize = transforms.Normalize(mean=[0.5115, 0.5115, 0.5115],
                                  std=[0.1316, 0.1316, 0.1316])

train_data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),normalize])

val_test_data_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),normalize])

data_folder = 'C&G_labels/'
# traced_script_module = None

image_datasets = {
    'train':
    MultiDataset(csv_file = data_folder+'train.csv',transform = train_data_transforms),
    'validation':
    MultiDataset(csv_file = data_folder+'val.csv',transform = val_test_data_transforms),
    'test':
    MultiDataset(csv_file = data_folder+'test.csv',transform = val_test_data_transforms)
}

dataloaders = {
    'train':
        torch.utils.data.DataLoader(image_datasets['train'],
                                    batch_size=1,
                                    shuffle=True,
                                    num_workers=8, pin_memory=True),
    'validation':
        torch.utils.data.DataLoader(image_datasets['validation'],
                                    batch_size=16,
                                    shuffle=False,
                                    num_workers=8, pin_memory=True),

    'test':
        torch.utils.data.DataLoader(image_datasets['test'],
                                    batch_size=16,
                                    shuffle=False,
                                    num_workers=8, pin_memory=True)
}

def train():
    model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=7)
    device = torch.device("cpu")
    # device = torch.device("cuda:0")
    model.to(device)

    #Set up parameters
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    num_epochs = 100

    def adjust_learning_rate(optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = 0.1 * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    # Create the model, change the output layer to 3
    best_acc = 0.0

    # Set up Adam optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Training starts
    with mlflow.start_run():
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
            adjust_learning_rate(optimizer, epoch)
            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects1 = 0
                running_corrects2 = 0
                for inputs, labels1, labels2, _, _, _ in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels1 = labels1.to(device)
                    labels2 = labels2.to(device)

                    # ============================================
                    outputs = model(inputs)
                    # global traced_script_module
                    # traced_script_module = torch.jit.trace(model, inputs)
                    # outputs = traced_script_module(inputs)
                    # ============================================

                    # criterion1 is softmax loss
                    loss1 = criterion1(outputs[:, :2], labels1)
                    loss2 = criterion2(outputs[:, 2:], labels2)
                    loss = loss1 + loss2
                    print("loss====={}".format(float(loss)))
                    mlflow.log_metric("loss", float(loss))
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds1 = torch.max(outputs[:, :2], 1)
                    running_corrects1 += torch.sum(preds1 == labels1.data)

                    _, preds2 = torch.max(outputs[:, 2:], 1)
                    running_corrects2 += torch.sum(preds2 == labels2.data)

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = (running_corrects1.double() + running_corrects2.double()) / len(image_datasets[phase]) / 2
                mlflow.log_metric("epoch_loss", float(epoch_loss))
                print("epoch_loss====={}".format(float(epoch_loss)))
                mlflow.log_metric("epoch_acc", float(epoch_acc))
                print("epoch_acc====={}".format(float(epoch_acc)))

                if phase == 'validation':
                    if epoch_acc > best_acc:
                        torch.save(model, 'best_weight_E_B2')
                        # traced_script_module.save("best_weight_E_B2.pt")
                        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
                        # Model registry does not work with file store
                        if tracking_url_type_store != "file":
                            # Register the model
                            # There are other ways to use the Model Registry, which depends on the use case,
                            # please refer to the doc for more information:
                            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                            mlflow.sklearn.log_model(model, "model", registered_model_name="XyModel")
                        else:
                            mlflow.sklearn.log_model(model, "model")


                        best_acc = epoch_acc
                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))


if __name__ == '__main__':
    os.environ["MLFLOW_TRACKING_URI"]="https://mlflow.mesh-nonprod.aws.megarobo.tech"
    os.environ["MLFLOW_S3_ENDPOINT_URL"]="https://minio-datalake-service.mesh-nonprod.aws.megarobo.tech"
    os.environ["AWS_ACCESS_KEY_ID"]="BTVCA6KV77CXTJCFWAE7"
    os.environ["AWS_SECRET_ACCESS_KEY"]="b9g7O0tCLo9jcDKMGKdsjSERSqQQP5lAzZjb6kEs"

    train()
    # mlflow.projects.run("E:/workspace/xy_fastapi/mlflow_xy_ai_model/train", backend="local",entry_point="train.py")