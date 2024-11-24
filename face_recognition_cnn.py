import torch
import torch.nn as nn
import torch.utils.data as torch_data
from tqdm import tqdm
import numpy as np
import torch.nn.functional as functional

import utils

# base model modelled after AlexNet (https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
class FaceRecognitionCNN(nn.Module):
    """
    A CNN model for face recognition with the following (rather simple) structure:
    1. Feature extraction layers (convolutional)
    2. Global average pooling
    3. Classifier layers (fully connected)
    """
    def __init__(self, classes, device, dropout_rate=0.4, confidence_threshold=0.7):
        super().__init__()
        self.classes = classes
        self.confidence_threshold = confidence_threshold

        # Define a consistent conv block structure
        def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        # Feature extraction layers with BatchNorm and residual connections
        self.features = nn.Sequential(
            # First block
            conv_block(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Second block
            conv_block(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Third block
            nn.Sequential(
                conv_block(192, 384, kernel_size=3, padding=1),
                conv_block(384, 512, kernel_size=3, padding=1),
                conv_block(512, 512, kernel_size=3, padding=1),
            ),

            # Fourth block
            conv_block(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier layers with reduced dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, len(self.classes))
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming normal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_epoch(self, data_loader, optimizer, loss_func, device):
        """
        Train the model for one epoch (part of the training loop)
        :param data_loader: the dataloader to be used for training
        :param optimizer: the optimizer to be used
        :param loss_func: the loss function to be used
        :param device: the device to process the data on
        :return: the average loss and the accuracy
        """
        self.train()
        total_loss = []
        correct = 0
        total = 0

        loop = tqdm(data_loader)
        loop.set_description("Training")

        for data in loop:
            images, labels = data
            images = images.to(device)
            labels = labels.float().to(device)  # Convert to float for loss calculation

            optimizer.zero_grad()
            outputs = self(images)

            loss = loss_func(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)

            optimizer.step()
            total_loss.append(loss.item())

            # Calculate accuracy with one-hot labels
            _, predicted = torch.max(outputs.data, 1)
            _, true_classes = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == true_classes).sum().item()

            loop.set_postfix(loss=loss.item())

        return np.array(total_loss).mean(), correct / total

    def evaluate_model(self, dataloader, loss_func, device, print_predictions=True):
        """
        Evaluate the model on a dataset (part of the training loop)
        :param dataloader: the dataloader to be used for the evaluation
        :param loss_func: the loss function to be used
        :param device: the device to process the data on
        :param print_predictions: whether to print the predictions of the first batch
        :return: the average loss and the accuracy
        """
        self.eval()
        correct = 0
        total = 0
        mean_loss = []

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                image, labels = data
                image = image.to(device)
                labels = labels.float().to(device)

                outputs = self(image)
                loss = loss_func(outputs, labels)
                mean_loss.append(loss.item())

                # Get predictions and confidences
                probabilities = functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                _, true_classes = torch.max(labels, 1)

                # Count correct predictions (only for known classes)
                total += labels.size(0)
                correct += (predicted == true_classes).sum().item()

                if print_predictions and i == 0:
                    print("\nSample predictions from first batch:")
                    for j in range(min(5, len(predicted))):
                        pred_confidence = confidence[j].item()
                        if pred_confidence < self.confidence_threshold:
                            pred_class = "unknown"
                        else:
                            pred_class = self.classes[predicted[j].item()]
                        true_class = self.classes[true_classes[j].item()]
                        print(f"Predicted: {pred_class} ({pred_confidence:.2%}), True: {true_class}")

        avg_loss = np.mean(mean_loss)
        accuracy = correct / total

        if print_predictions:
            print(f"\nEvaluation metrics:")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Accuracy: {accuracy:.2%}")

        return avg_loss, accuracy

    def predict(self, image, device):
        """
        Predict the class of an image
        :param image: a face image tensor
        :param device: the device to process the image on
        :return: the predicted class and the confidence
        """
        self.eval()
        with torch.no_grad():
            image = image.to(device)
            outputs = self(image)
            probabilities = functional.softmax(outputs, dim=1)

            # Get maximum probability and corresponding class
            confidence, predicted = torch.max(probabilities, 1)
            confidence = confidence.item()

            # Check if the confidence is below the threshold
            if confidence < self.confidence_threshold:
                return "unknown", confidence

            return self.classes[predicted.item()], confidence


    def set_confidence_threshold(self, threshold):
        """Update the confidence threshold for unknown prediction"""
        self.confidence_threshold = threshold

    def save_model(self, path, loss):
        save = {
            "model": self.state_dict(),
            "classes": self.classes,
            "loss": loss,
            "confidence_threshold": self.confidence_threshold
        }
        torch.save(save, path)

    def load_model(self, path, device):
        save = torch.load(path)
        self.classes = save["classes"]
        self.confidence_threshold = save.get("confidence_threshold", 0.7)  # Default if loading older models
        self.__init__(self.classes, device, confidence_threshold=self.confidence_threshold)
        self.load_state_dict(save["model"])
        return save["loss"], self