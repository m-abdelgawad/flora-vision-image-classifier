import json
from collections import OrderedDict

import torch
from PIL import Image
from torchvision import datasets, transforms


def get_testing_transforms():
    """
    Returns the standard image transforms for validation and testing datasets.
    """
    return transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def load_datasets(train_dir, valid_dir, test_dir):
    """
    Defines image transforms and loads datasets for training, validation, and testing.
    """
    # Define training transforms with data augmentation
    training_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Define validation and testing transforms
    validation_transforms = get_testing_transforms()
    testing_transforms = get_testing_transforms()
    # Load datasets
    training_dataset = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transforms)
    # Ensure consistent class_to_idx across all datasets
    class_to_idx = training_dataset.class_to_idx
    validation_dataset.class_to_idx = class_to_idx
    testing_dataset.class_to_idx = class_to_idx
    return training_dataset, validation_dataset, testing_dataset


def create_dataloaders(training_dataset, validation_dataset, testing_dataset, batch_size=64):
    """
    Creates PyTorch dataloaders for training, validation, and testing datasets.
    """
    trainingloader = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True
    )
    print(
        "Training loader has {} batches, and each batch has {} samples".format(
            len(trainingloader), batch_size
        )
    )
    validationloader = torch.utils.data.DataLoader(
        validation_dataset, batch_size=batch_size
    )
    print(
        "Validation loader has {} batches, and each batch has {} samples".format(
            len(validationloader), batch_size
        )
    )
    testingloader = torch.utils.data.DataLoader(
        testing_dataset, batch_size=batch_size
    )
    print(
        "Testing loader has {} batches, and each batch has {} samples".format(
            len(testingloader), batch_size
        )
    )
    return trainingloader, validationloader, testingloader


def load_category_mapping(mapping_file_path):
    """
    Loads the category-to-name mapping from a JSON file and prints it in order.
    """
    with open(mapping_file_path, 'r') as f:
        cat_to_name = json.load(f)
    ordered_cat_to_name = OrderedDict(sorted(cat_to_name.items(), key=lambda x: int(x[0])))
    return ordered_cat_to_name


def process_image(image_path, transform=None):
    """
    Processes a PIL image for use in a PyTorch model by scaling, cropping,
    and normalizing the image.
    """
    try:
        input_image = Image.open(image_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load image from {image_path}: {e}")

    if transform is None:
        transform = get_testing_transforms()

    output_image = transform(input_image)

    return output_image
