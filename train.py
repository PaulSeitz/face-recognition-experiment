import utils
from face_recognition_cnn import FaceRecognitionCNN
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from dataloader import get_datasets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKER = 4


def train(classes, parameters, load_model, save_faces):
    """
    Train the face recognition model on the given classes with the given parameters
    :param classes: the different labels the model should learn to recognize
    :param parameters: the (common) training parameters - epochs, batch size, learning rate, weight decay
    :param load_model: the path to a model that should be loaded before training, if one is provided
    :param save_faces: whether the detected faces should be saved to a file or loaded from one
    """
    # get the dataset paths and the (updated) class string with the final classes
    train_dataset, eval_dataset, class_string = get_datasets(classes, save_faces)

    # Create model
    model = FaceRecognitionCNN(class_string, DEVICE).to(DEVICE)

    print(f"Using {DEVICE} for training")

    if load_model:
        best_loss, model = model.load_model(load_model, DEVICE)

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(parameters["batch_size"]),
        shuffle=True,
        num_workers=WORKER
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=int(parameters["batch_size"]),
        shuffle=False,
        num_workers=WORKER
    )

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters.get("learning_rate", 1e-4)),
        weight_decay=float(parameters.get("weight_decay", 0.01))
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    # get the weighted loss function (done since the classes are in part very imbalanced)
    loss_func = utils.get_weighted_loss_func(train_dataset, model.classes, DEVICE)

    best_loss = float('inf')
    for epoch in range(int(parameters["epochs"])):
        # Training phase
        model.train()
        loss, acc = model.train_epoch(train_dataloader, optimizer, loss_func, DEVICE)

        # Evaluation phase
        eval_loss, eval_acc = model.evaluate_model(eval_dataloader, loss_func, DEVICE)

        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}, "
              f"Eval Loss={eval_loss:.4f}, Eval Acc={eval_acc:.4f}")

        scheduler.step(eval_loss)

        if eval_loss < best_loss:
            best_loss = eval_loss
            print("Saving best model...")
            model.save_model("./trained_models/best_model.pth", eval_loss)

        # Early stopping check
        if epoch > 10 and eval_loss > best_loss * 1.1:
            print("\nStopping early due to loss increase")
            break

    # save the final model in a separate file
    model.save_model("./trained_models/final_model.pth", eval_loss)
    print("Training completed")