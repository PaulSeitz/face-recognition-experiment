import torch
from face_recognition_cnn import FaceRecognitionCNN
from dataloader import get_datasets
from torch.utils.data import DataLoader
import utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKER = 4


def modify_model_for_fine_tuning(model: FaceRecognitionCNN, new_classes):
    """
    Modify the model architecture for fine-tuning:
    1. Freeze early convolutional layers
    2. Replace the final classification layer with a new one
    3. Keep some later layers trainable for better adaptation
    :param model: the model to modify
    :param new_classes: the new classes that should be learned
    :return: the modified model
    """
    # Freeze early layers (first 2/3 of feature extractor)
    layers_to_freeze = len(list(model.features.children())) * 2 // 3
    for i, layer in enumerate(model.features.children()):
        if i < layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False

    # Get the input features for the final layer
    num_features = model.classifier[-1].in_features

    # Replace the final layer with a new one for the new classes
    model.classifier[-1] = torch.nn.Linear(num_features, len(new_classes))

    # Initialize the new layer with Xavier/Glorot initialization
    torch.nn.init.xavier_uniform_(model.classifier[-1].weight)
    torch.nn.init.zeros_(model.classifier[-1].bias)

    # Update the classes attribute
    model.classes = new_classes

    return model


def fine_tune(base_model_path, classes, parameters, save_faces=True):
    """
    Fine-tune the model on a new set of classes
    :param base_model_path: path to the base model that should be fine-tuned
    :param classes: the new classes that should be learned
    :param parameters: the training parameters
    :param save_faces: whether the detected faces should be saved to a file or loaded from one
    """
    print("\nStarting fine-tuning process...")

    # Load datasets
    train_dataset, eval_dataset, class_string = get_datasets(classes, save_faces, data_path="./data/finetune/")

    # Load the base model
    base_model = FaceRecognitionCNN({}, DEVICE)
    _, base_model = base_model.load_model(base_model_path, DEVICE)

    print(f"\nOriginal model classes: {len(base_model.classes)}")
    print(f"New model classes: {len(class_string)}")
    print("New classes:", class_string)

    # Modify model for fine-tuning
    model = modify_model_for_fine_tuning(base_model, class_string)
    model.to(DEVICE)

    # Separate parameters into two groups
    final_layer_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:  # Only consider trainable parameters
            if 'classifier.6' in name:  # Final layer
                final_layer_params.append(param)
            else:
                other_params.append(param)

    # Create optimizer with separate parameter groups to allow for different learning rates (the newly initialized layers use a higher learning rate)
    optimizer = torch.optim.AdamW([
        {'params': other_params,
         'lr': float(parameters.get("learning_rate", 1e-4)) * 0.1},
        {'params': final_layer_params,
         'lr': float(parameters.get("learning_rate", 1e-4))}
    ], weight_decay=float(parameters.get("weight_decay", 0.01)))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    # print out a list of the model parameters and their training status for logging purposes
    print("\nFine-tuning model parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'classifier.6' in name:
                print(f"{name}: trainable (high lr)")
            else:
                print(f"{name}: trainable (low lr)")
        else:
            print(f"{name}: frozen")

    # Verify dataset classes match model classes to prevent errors later on
    assert len(model.classes) == len(train_dataset.classes), \
        f"Model classes ({len(model.classes)}) doesn't match dataset classes ({len(train_dataset.classes)})"

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

    # get the weighted loss function (done since the classes can be imbalanced)
    loss_func = utils.get_weighted_loss_func(train_dataset, model.classes, DEVICE)

    best_loss = float('inf')
    best_acc = 0
    for epoch in range(int(parameters["epochs"])):
        # Training phase
        model.train()
        loss, acc = model.train_epoch(train_dataloader, optimizer, loss_func, DEVICE)

        # Evaluation phase
        eval_loss, eval_acc = model.evaluate_model(eval_dataloader, loss_func, DEVICE)

        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}, "
              f"Eval Loss={eval_loss:.4f}, Eval Acc={eval_acc:.4f}")

        scheduler.step(eval_loss)

        # Save best model by accuracy instead of loss (since we are trying to maximize the recognition accuracy in the fine-tuning)
        if eval_acc > best_acc:
            best_acc = eval_acc
            best_loss = eval_loss
            print(f"Saving best model... (Accuracy: {best_acc:.4f})")
            model.save_model("./trained_models/best_finetune_model.pth", eval_loss)

        # Early stopping check
        if epoch > 10 and eval_loss > best_loss * 1.1:
            print("\nStopping early due to loss increase")
            break

    # Save the final model
    model.save_model("./trained_models/finetuned_final_model.pth", eval_loss)
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_acc:.4f}")