import torch
import torch.nn as nn
import torch.utils.data as torch_data
from tqdm import tqdm
import numpy as np


class FaceRecognitionCNN(nn.Module):
    classes = []

    def __init__(self, classes, device, load_model=None):
        super().__init__()
        if load_model:
            self.load_model(load_model, device)
        else:
            self.classes = classes
            # base model modelled after AlexNet (https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
            self.convolution = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2)
            )
            self.pool = nn.AdaptiveAvgPool2d((3, 3))

            # Calculate the correct input size for the linear layer
            with torch.no_grad():
                x = torch.randn(1, 3, 255, 255)
                x = self.convolution(x)
                x = self.pool(x)
                x = torch.flatten(x, 1)
                self.fc_input_size = x.size(1)

            self.linearClassifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(self.fc_input_size, 4096),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.LeakyReLU(inplace=True),
                nn.Linear(4096, len(classes)),
            )

    def forward(self, data):
        data = self.convolution(data)
        data = self.pool(data)
        data = torch.flatten(data, 1)
        data = self.linearClassifier(data)
        return data

    def train_epoch(self, data_loader: torch_data.DataLoader, optimizer: torch.optim.Optimizer, device, loss_function):
        # initialize fancy output
        console_out = tqdm(data_loader)
        console_out.set_description('Training epoch')

        # set the model to training mode
        self.train(True)

        mean_loss = []
        correct = 0
        total = 0

        # iterate over the batch
        for i, data in enumerate(console_out):
            # get the image and labels
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            # zero the gradients
            optimizer.zero_grad()

            # perform a forward pass
            outputs = self(image)

            # calculate the loss
            loss = loss_function(outputs, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            mean_loss.append(loss.item())

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            console_out.set_postfix({
                "Loss": np.mean(mean_loss),
                "Acc": 100. * correct / total
            })

    def evaluate_model(self, dataloader, device, loss_function):
        # initialize fancy output
        console_out = tqdm(dataloader)
        console_out.set_description('Evaluating model')

        # set the model to evaluation mode
        self.train(False)

        mean_loss = []
        # iterate over the batch
        for i, data in enumerate(console_out):
            # get the image and labels
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = self(image)
                loss = loss_function(outputs, labels)

            mean_loss.append(loss.to("cpu").item())

            console_out.set_postfix({"Loss": np.array(mean_loss).mean()})

        return np.array(mean_loss).mean()

    def predict(self, image, device):
        self.train(False)
        with torch.no_grad():
            image = image.to(device)

            output = self(image)

            print(output)

            index = torch.argmax(output, dim=1).item()
            # get the class name
            name = list(self.classes.keys())[index]
            return name

    def save_model(self, path, loss):
        save = {
            "model": self.state_dict(),
            "classes": self.classes,
            "loss": loss
        }
        torch.save(save, path)

    def load_model(self, path, device):
        save = torch.load(path)
        self.classes = save["classes"]

        # Recreate the model architecture
        self.__init__(self.classes, device)

        # Now load the state dict
        self.load_state_dict(save["model"])
        return save["loss"], self
