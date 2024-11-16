from collections import OrderedDict

import torch
from torch import nn
from torchvision import models

from ..data.dataset import process_image


def load_pretrained_model(arch='vgg16'):
    """
    Loads a pre-trained model from torchvision.models based on the specified architecture.
    """
    if arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    elif arch == 'vgg13':
        model = models.vgg13(weights=models.VGG13_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported architecture '{arch}'. Available options: 'vgg16', 'vgg13'.")
    print(f"Loaded {arch} model successfully...")
    return model


def inspect_original_classifier(model):
    """
    Prints the classifier of the original pre-trained model and its input features.
    """
    print("Original model's classifier is: \n{}".format(model.classifier))
    in_features = model.classifier[0].in_features
    print("The in_features of the classifier's first layer: {}".format(in_features))


def configure_custom_classifier_params(model, hidden_units, output_units, dropout_rate):
    """
    Configures the parameters for the custom classifier.
    """
    # Get the input size of the classifier
    if hasattr(model.classifier, 'in_features'):
        model_in_features = model.classifier.in_features
    elif isinstance(model.classifier, nn.Sequential):
        # Assume the first layer is Linear
        model_in_features = None
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                model_in_features = layer.in_features
                break
        if model_in_features is None:
            raise ValueError("Cannot find in_features in model.classifier")
    else:
        raise ValueError("Unsupported model.classifier type")

    print("in_features of the classifier: {}".format(model_in_features))
    print("Set hidden units to: {}".format(hidden_units))
    print("Set output units to: {}".format(output_units))
    print("Set dropout rate to: {}".format(dropout_rate))
    return model_in_features, hidden_units, output_units, dropout_rate


def build_custom_model(arch, model_in_features, hidden_units, output_units, dropout_rate):
    """
    Builds a custom model based on the specified pre-trained architecture.
    """
    model = load_pretrained_model(arch)
    print(f"Loaded pre-trained {arch} model successfully...")

    # Freeze feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False
    print("Feature extractor layers frozen...")

    # Build custom classifier
    classifier = nn.Sequential(
        OrderedDict([
            ('fc1', nn.Linear(model_in_features, hidden_units)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=dropout_rate)),
            ('fc2', nn.Linear(hidden_units, int(hidden_units / 2))),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=dropout_rate)),
            ('fc3', nn.Linear(int(hidden_units / 2), output_units)),
            ('output', nn.LogSoftmax(dim=1))
        ])
    )
    model.classifier = classifier
    print("Custom classifier built and replaced the original classifier...")
    return model


def move_model_to_device(model, device):
    """
    Moves the model to the specified device.
    """
    model.to(device)
    print("Moved model to device: {}".format(device))
    return model


def load_model_checkpoint(model_file_path, device):
    """
    Loads a model checkpoint and rebuilds the model.
    """
    # Load the checkpoint safely
    try:
        checkpoint = torch.load(model_file_path, map_location=device, weights_only=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Extract required metadata for model rebuilding
    arch = checkpoint.get('arch', 'vgg16')
    in_features_count = checkpoint.get('in_features_count')
    hidden_units_count = checkpoint.get('hidden_units_count')
    output_units_count = checkpoint.get('output_units_count')
    dropout_rate = checkpoint.get('dropout_rate', 0.3)

    # Rebuild the model
    model = build_custom_model(
        arch=arch,
        model_in_features=in_features_count,
        hidden_units=hidden_units_count,
        output_units=output_units_count,
        dropout_rate=dropout_rate
    )

    # Load the model's state dictionary
    model.load_state_dict(checkpoint['state_dict'])

    # Restore class-to-index mapping
    model.class_to_idx = checkpoint['class_to_idx']

    # Set the model to evaluation mode
    model.eval()

    # Move model to device
    model.to(device)

    # Extract and return metadata
    metadata = {k: checkpoint[k] for k in checkpoint if k != 'state_dict'}
    print(f"Model loaded with metadata: {metadata}")
    return model, metadata


def invert_class_to_idx(class_to_idx):
    """
    Inverts a class_to_idx dictionary to create an idx_to_class mapping.
    """
    if not isinstance(class_to_idx, dict):
        raise ValueError("The provided class_to_idx must be a dictionary.")
    return {v: k for k, v in class_to_idx.items()}


def predict(image_path, model, cat_to_name, idx_to_class, topk=5):
    """
    Predicts the top K classes and their probabilities for a given image using a trained model.
    """
    # Set model to evaluation mode
    model.eval()

    # Determine the device of the model
    device = next(model.parameters()).device

    # Process the image and move it to the correct device
    processed_image = process_image(image_path).to(device).unsqueeze(0)

    # Perform forward pass through the model without tracking gradients
    with torch.no_grad():
        y_hat_logps = model(processed_image)

    # Convert the output log-probabilities to probabilities
    y_hat_ps = torch.exp(y_hat_logps)

    # Get the top-k probabilities and indices
    top_ps, top_indices = y_hat_ps.topk(topk, dim=1)

    # Convert tensors to lists
    top_ps = top_ps.flatten().tolist()
    top_indices = top_indices.flatten().tolist()

    # Map indices to their actual class labels
    true_top_indices = [idx_to_class[index] for index in top_indices]

    # Map class labels to their category names
    top_labels = [cat_to_name.get(x, "Unknown") for x in true_top_indices]

    # Convert probabilities to percentages
    top_ps_perc = [round(ps * 100, 2) for ps in top_ps]

    return top_labels, top_ps_perc
