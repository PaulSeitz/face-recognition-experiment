import torch
import torch.nn as nn
import torch.utils.data as torch_data
from tqdm import tqdm
import numpy as np


def calculate_corrects(outputs, labels):
    # Get the predicted class (index of the maximum value) for each sample in outputs
    predicted = torch.argmax(outputs, dim=1)

    # Get the true class (index of the 1 in one-hot encoded labels) for each sample
    true_classes = torch.argmax(labels, dim=1)

    # Compare predictions with true classes and sum up the correct predictions
    correct_predictions = (predicted == true_classes).sum().item()

    return correct_predictions


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
                nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                nn.Conv2d(48, 128, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )

            self.linearClassifier = nn.Sequential(
                nn.Dropout(p=0.8),
                nn.Linear(128 * 3 * 3, 4096),
                nn.ReLU(inplace=True),

                nn.Dropout(p=0.8),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),

                nn.Linear(4096, len(self.classes)),
            )

    def forward(self, x):
        x = self.convolution(x)
        x = torch.flatten(x, 1)
        x = self.linearClassifier(x)
        return x

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
            labels = labels.float().to(device)

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

            #_, predicted = outputs.max(1)
            total += labels.size(0)
            correct += calculate_corrects(outputs, labels)

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

        correct = 0
        total = 0
        mean_loss = []
        # iterate over the batch
        for i, data in enumerate(console_out):
            # get the image and labels
            image, labels = data
            image = image.to(device)
            labels = labels.float().to(device)

            with torch.no_grad():
                outputs = self(image)
                loss = loss_function(outputs, labels)

            mean_loss.append(loss.to("cpu").item())

            #_, predicted = outputs.max(1)
            total += labels.size(0)
            correct += calculate_corrects(outputs, labels)

            console_out.set_postfix({
                "Loss": np.mean(mean_loss),
                "Acc": 100. * correct / total
            })

            console_out.set_postfix({
                "Loss": np.mean(mean_loss),
                "Acc": 100. * correct / total
            })

        return np.array(mean_loss).mean()

    def predict(self, image, device):
        self.train(False)
        with torch.no_grad():
            image = image.to(device)

            output = self(image)
            output = torch.nn.functional.softmax(output, dim=1)

            print(output)

            index = torch.argmax(output, dim=1).item()
            # get the class name
            name = self.classes[index]
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
