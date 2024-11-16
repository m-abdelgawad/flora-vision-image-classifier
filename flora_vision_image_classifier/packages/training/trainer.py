import time

import torch
from torch import optim


def initialize_train_params(epochs, print_every, learning_rate):
    """
    Initializes training hyperparameters and tracking variables.
    """
    train_params = {
        "epochs": epochs,
        "print_every": print_every,
        "learning_rate": learning_rate,
        "training_losses": [],
        "validation_losses": [],
        "validation_accuracies": [],
    }
    print("Training parameters initialized.")
    return train_params


def setup_optimizer_and_scheduler(model, learning_rate):
    """
    Configures the optimizer and learning rate scheduler for training.
    """
    optimizer = optim.Adam(
        model.classifier.parameters(), lr=learning_rate, weight_decay=0.0001
    )
    print("Optimizer initialized with learning rate: {}".format(learning_rate))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    print("Scheduler initialized with mode='max' for validation accuracy.")
    return optimizer, scheduler


def train_and_validate_model(
        model,
        criterion,
        training_loader,
        validation_loader,
        device,
        train_params,
        optimizer,
        scheduler
):
    """
    Trains and validates the model for the specified number of epochs.
    """
    training_start = time.time()
    train_results = {
        "training_losses": [],
        "validation_losses": [],
        "validation_accuracies": [],
    }
    for epoch in range(train_params["epochs"]):
        print("\nStarting epoch #{} out of {}".format(epoch + 1, train_params["epochs"]))
        print("-" * 50)
        epoch_training_loss = 0
        model.train()
        for idx, (inputs, labels) in enumerate(training_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(inputs)
            loss = criterion(logps, labels)
            epoch_training_loss += loss.item()
            loss.backward()
            optimizer.step()
            if idx % train_params["print_every"] == 0:
                print(
                    "Batch #{}/{} - Loss so far: {:.2f}".format(
                        idx, len(training_loader), epoch_training_loss
                    )
                )
        avg_training_loss = epoch_training_loss / len(training_loader)
        train_results["training_losses"].append(avg_training_loss)
        print("Starting validation...")
        model.eval()
        epoch_validation_loss, correct_preds, total_preds = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model(inputs)
                loss = criterion(logps, labels)
                epoch_validation_loss += loss.item()
                _, predictions = torch.max(logps, dim=1)
                correct_preds += (predictions == labels).sum().item()
                total_preds += labels.size(0)
        avg_validation_loss = epoch_validation_loss / len(validation_loader)
        validation_accuracy = (correct_preds / total_preds) * 100
        train_results["validation_losses"].append(avg_validation_loss)
        train_results["validation_accuracies"].append(validation_accuracy)
        scheduler.step(validation_accuracy)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print("Learning rate after scheduler step: {:.6f}".format(current_lr))
        print(
            "Epoch {} Summary: Training Loss={:.2f}, Validation Loss={:.2f}, "
            "Validation Accuracy={:.2f}%".format(
                epoch + 1,
                avg_training_loss,
                avg_validation_loss,
                validation_accuracy,
            )
        )
        print("-" * 50)
    training_duration_secs = time.time() - training_start
    train_results["training_duration_secs"] = training_duration_secs
    if training_duration_secs > 60:
        print("Training completed in {:.2f} minutes.".format(training_duration_secs / 60))
    else:
        print("Training completed in {:.2f} seconds.".format(training_duration_secs))
    return train_results


def validate_on_test_set(model, testingloader, criterion, device):
    """
    Validates the model on the test dataset and computes the average loss
    and accuracy.
    """
    print("Start validation on the test set...")
    total_testing_loss = 0
    testing_correct_predictions = 0
    testing_total_predictions = 0
    model.eval()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testingloader):
            print(f"Starting batch {idx + 1}/{len(testingloader)}...")
            inputs = inputs.to(device)
            labels = labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            total_testing_loss += batch_loss.item()
            _, predictions_labels = torch.max(logps, dim=1)
            equals = predictions_labels == labels.view(*predictions_labels.shape)
            testing_correct_predictions += equals.sum().item()
            testing_total_predictions += labels.shape[0]
            print(
                f"Batch {idx + 1} summary: batch_loss={batch_loss:.2f}, "
                f"correct_predictions={testing_correct_predictions}, "
                f"total_predictions={testing_total_predictions}"
            )
    print("Finished validation on the test set...")
    testing_avg_loss = total_testing_loss / len(testingloader)
    print(f"\nAverage loss per batch on the testing dataset is: {testing_avg_loss:.2f}")
    testing_accuracy = (testing_correct_predictions / testing_total_predictions) * 100
    print(f"Accuracy of the model on the testing dataset is: {testing_accuracy:.2f}%")
    return {
        'testing_avg_loss': testing_avg_loss,
        'testing_accuracy': testing_accuracy
    }
