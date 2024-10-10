from face_recognition_cnn import FaceRecognitionCNN
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataloader import get_dataloader, show_images, show_tensor_image
from sklearn.model_selection import KFold
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKER = 4

def train(classes, parameters, load_model):
    dataset = get_dataloader(classes, save_faces=True)

    classes = dataset.get_classes()

    # create the model
    model = FaceRecognitionCNN(classes, DEVICE).to(DEVICE)

    # create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(parameters["learning_rate"]))

    loss_function = nn.CrossEntropyLoss()

    best_loss = 1000

    print(DEVICE)

    if load_model:
        # get the model from the path and set its loss
        best_loss, model = model.load_model(load_model, DEVICE)

    model.to(DEVICE)

    train_sampler = SubsetRandomSampler(range(int(len(dataset) * 0.8)))
    test_sampler = SubsetRandomSampler(range(int(len(dataset) * 0.8), len(dataset)))

    train_dataloader = DataLoader(dataset, batch_size=int(parameters["batch_size"]), sampler=train_sampler, num_workers=WORKER)
    test_dataloader = DataLoader(dataset, batch_size=int(parameters["batch_size"]), sampler=test_sampler, num_workers=WORKER)

    for epoch in range(int(parameters["epochs"])):
        model.train_epoch(train_dataloader, optimizer, DEVICE, loss_function)

        eval_loss = model.evaluate_model(test_dataloader, DEVICE, loss_function)
        print(f"Epoch {epoch + 1} completed, Evaluation Loss: {eval_loss}")

        last_loss = eval_loss
        if eval_loss < best_loss:
            best_loss = eval_loss
            print("saving model")
            model.save_model("./trained_models/cnn_model.pth", eval_loss)

    model.save_model("./trained_models/final_cnn_model.pth", last_loss)

    print("Training completed")