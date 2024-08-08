from face_recognition_cnn import FaceRecognitionCNN
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataloader import get_dataloader
from sklearn.model_selection import KFold
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(classes, parameters, load_model):
    dataset = get_dataloader(classes, save_faces=True)

    # create the model
    model = FaceRecognitionCNN(classes, DEVICE).to(DEVICE)

    print(float(parameters["learning_rate"]))
    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(parameters["learning_rate"]))

    # get the current class distributions by counting the labels
    class_counts = [len([label for _, label in dataset if label == i]) for i in range(len(classes))]

    class_weights = torch.tensor([max(class_counts) / count for count in class_counts]).to(DEVICE)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    if int(parameters["k-fold"]) == 1:
        best_loss = 1000

        if load_model:
            # get the model from the path and set its loss
            best_loss, model = model.load_model(load_model, DEVICE)

        model.to(DEVICE)

        train_sampler = SubsetRandomSampler(range(int(len(dataset) * 0.8)))
        test_sampler = SubsetRandomSampler(range(int(len(dataset) * 0.8), len(dataset)))

        train_dataloader = DataLoader(dataset, batch_size=int(parameters["batch_size"]), sampler=train_sampler)
        test_dataloader = DataLoader(dataset, batch_size=int(parameters["batch_size"]), sampler=test_sampler)

        for epoch in range(int(parameters["epochs"])):
            model.train_epoch(train_dataloader, optimizer, DEVICE, loss_function)

            eval_loss = model.evaluate_model(test_dataloader, DEVICE, loss_function)

            last_loss = eval_loss
            if eval_loss < best_loss:
                best_loss = eval_loss
                print("saving model")
                model.save_model("./trained_models/cnn_model.pth", eval_loss)

        model.save_model("./trained_models/final_cnn_model.pth", last_loss)

        print("Training completed")
    else:
        # create a k-fold cross validator
        k_fold = KFold(n_splits=int(parameters["k-fold"]), shuffle=True, random_state=42)

        best_loss = 1000

        if load_model:
            # get the model from the path and set its loss
            best_loss, model = model.load_model(load_model, DEVICE)

        model.to(DEVICE)

        last_loss = 0

        for k, (train_idx, test_idx) in enumerate(k_fold.split(dataset)):

            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(test_idx)

            train_dataloader = DataLoader(dataset, batch_size=int(parameters["batch_size"]), sampler=train_sampler)
            test_dataloader = DataLoader(dataset, batch_size=int(parameters["batch_size"]), sampler=test_sampler)

            for epoch in range(int(parameters["epochs"])):
                model.train_epoch(train_dataloader, optimizer, DEVICE, loss_function)

            eval_loss = model.evaluate_model(test_dataloader, DEVICE, loss_function)
            print(f"Fold {k + 1} completed, Evaluation Loss: {eval_loss}")
            last_loss = eval_loss
            if eval_loss < best_loss:
                best_loss = eval_loss
                print("saving model")
                model.save_model("./trained_models/cnn_model.pth", eval_loss)

        model.save_model("./trained_models/final_cnn_model.pth", last_loss)

        print("Training completed")
