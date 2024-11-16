import torch


def print_class_indices(training_dataset, validation_dataset, testing_dataset):
    """
    Prints the class indices for training, validation, and testing datasets.
    """
    print("Training dataset class indices:\n{}".format(training_dataset.class_to_idx))
    print("\nValidation dataset class indices:\n{}".format(validation_dataset.class_to_idx))
    print("\nTesting dataset class indices:\n{}".format(testing_dataset.class_to_idx))


def get_device(use_gpu=False):
    """
    Determines the device to use for training and evaluation.
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU.")
    else:
        device = torch.device("cpu")
        if use_gpu:
            print("GPU not available. Falling back to CPU.")
        else:
            print("Using CPU.")
    return device


def save_model_checkpoint(model_file_path, model, arch, metadata):
    """
    Saves the trained model checkpoint along with metadata to a file.
    """
    model.class_to_idx = metadata['class_to_idx']
    checkpoint = {
        'state_dict': model.state_dict(),
        'epochs': metadata['epochs'],
        'testing_accuracy': metadata['testing_accuracy'],
        'learning_rate': metadata['learning_rate'],
        'in_features_count': metadata['in_features_count'],
        'hidden_units_count': metadata['hidden_units_count'],
        'output_units_count': metadata['output_units_count'],
        'class_to_idx': metadata['class_to_idx'],
        'arch': arch,
    }
    torch.save(checkpoint, model_file_path)
    print("Model checkpoint saved to: {}".format(model_file_path))
