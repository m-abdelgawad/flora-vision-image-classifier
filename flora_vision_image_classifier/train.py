import argparse
import os

import torch
import yaml

# Import modules from packages
from packages.data.dataset import (
    load_datasets,
    create_dataloaders,
    load_category_mapping,
)
from packages.models.model_utils import (
    load_pretrained_model,
    inspect_original_classifier,
    configure_custom_classifier_params,
    build_custom_model,
    move_model_to_device,
)
from packages.training.trainer import (
    initialize_train_params,
    setup_optimizer_and_scheduler,
    train_and_validate_model,
    validate_on_test_set,
)
from packages.utils.utils import (
    get_device,
    save_model_checkpoint,
    print_class_indices,
)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a neural network on a dataset.')
    parser.add_argument('data_directory', nargs='?', default=None,
                        help='Path to the dataset directory (overrides config.yaml if provided)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default=None,
                        help='Choose architecture ("vgg13" or "vgg16")')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate for training')
    parser.add_argument('--hidden_units', type=int, default=None,
                        help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training if available')
    args = parser.parse_args()

    # Load configurations
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get the base directory from config
    base_dir = config.get('base_dir', '.')
    base_dir = os.path.abspath(base_dir)

    # Print the current working directory for verification
    print(f"Current working directory: {base_dir}")

    # Determine data directory
    data_config = config['data']
    if args.data_directory:
        data_directory = os.path.abspath(args.data_directory)
        print(f"Using data directory from command-line argument: {data_directory}")
        train_dir = os.path.join(data_directory, 'train')
        valid_dir = os.path.join(data_directory, 'valid')
        test_dir = os.path.join(data_directory, 'test')
    else:
        print("Using data directories from config.yaml")
        train_dir = os.path.join(base_dir, data_config['train_dir'])
        valid_dir = os.path.join(base_dir, data_config['valid_dir'])
        test_dir = os.path.join(base_dir, data_config['test_dir'])

    # Load datasets
    training_dataset, validation_dataset, testing_dataset = load_datasets(
        train_dir, valid_dir, test_dir
    )

    # Print class indices
    print_class_indices(training_dataset, validation_dataset, testing_dataset)

    # Create dataloaders for training, validation, and testing datasets
    batch_size = data_config.get('batch_size', 64)
    trainingloader, validationloader, testingloader = create_dataloaders(
        training_dataset, validation_dataset, testing_dataset, batch_size=batch_size
    )

    # Load the category-to-name mapping
    categories_map_path = os.path.join(base_dir, data_config['category_mapping'])
    cat_to_name = load_category_mapping(categories_map_path)

    # Determine the model architecture
    if args.arch:
        arch = args.arch
        print(f"Using architecture from command-line argument: {arch}")
    else:
        model_config = config.get('model', {})
        arch = model_config.get('arch', 'vgg16')
        print(f"Using default architecture from config.yaml: {arch}")

    # Load the pre-trained model
    model = load_pretrained_model(arch)

    # Inspect the classifier of the original model
    inspect_original_classifier(model)

    # Configure parameters for the custom classifier
    model_config = config.get('model', {})
    if args.hidden_units:
        hidden_units = args.hidden_units
        print(f"Using hidden_units from command-line argument: {hidden_units}")
    else:
        hidden_units = model_config.get('hidden_units', 4096)
        print(f"Using default hidden_units from config.yaml: {hidden_units}")

    output_units = model_config.get('output_units', 102)
    dropout_rate = model_config.get('dropout_rate', 0.3)

    # Now configure the classifier parameters
    model_in_features, hidden_units, output_units, dropout_rate = configure_custom_classifier_params(
        model,
        hidden_units=hidden_units,
        output_units=output_units,
        dropout_rate=dropout_rate,
    )

    # Build the custom pre-trained model
    model = build_custom_model(
        arch=arch,
        model_in_features=model_in_features,
        hidden_units=hidden_units,
        output_units=output_units,
        dropout_rate=dropout_rate
    )
    print("Built the custom pre-trained model: \n{}".format(model))

    # Determine the device
    device = get_device(use_gpu=args.gpu)
    model = move_model_to_device(model, device)

    # Initialize training parameters
    train_config = config.get('training', {})
    if args.learning_rate:
        learning_rate = args.learning_rate
        print(f"Using learning_rate from command-line argument: {learning_rate}")
    else:
        learning_rate = train_config.get('learning_rate', 0.001)
        print(f"Using default learning_rate from config.yaml: {learning_rate}")

    if args.epochs:
        epochs = args.epochs
        print(f"Using epochs from command-line argument: {epochs}")
    else:
        epochs = train_config.get('epochs', 2)
        print(f"Using default epochs from config.yaml: {epochs}")

    print_every = train_config.get('print_every', 50)

    train_params = initialize_train_params(
        epochs=epochs,
        print_every=print_every,
        learning_rate=learning_rate
    )

    # Define the loss function and optimizer
    criterion = torch.nn.NLLLoss()
    optimizer, scheduler = setup_optimizer_and_scheduler(model, train_params['learning_rate'])

    # Train and validate the model
    train_results = train_and_validate_model(
        model,
        criterion,
        trainingloader,
        validationloader,
        device,
        train_params,
        optimizer,
        scheduler
    )

    # Validate the model on the test set
    test_results = validate_on_test_set(model, testingloader, criterion, device)

    # Save the model checkpoint
    checkpoint_config = config.get('checkpoint', {})

    if args.save_dir:
        output_dir = os.path.abspath(args.save_dir)
        print(f"Using save directory from command-line argument: {output_dir}")
    else:
        output_dir = os.path.join(base_dir, checkpoint_config.get('output_dir', ''))
        print(f"Using default save directory from config.yaml: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    model_file_name_pattern = checkpoint_config.get('file_name_pattern', 'checkpoint.pth')
    model_file_name = model_file_name_pattern.format(
        testing_accuracy=int(test_results.get('testing_accuracy', 0))
    )
    model_file_path = os.path.join(output_dir, model_file_name)

    # Create metadata for the checkpoint
    checkpoint_metadata = {
        'arch': arch,
        'epochs': train_params['epochs'],
        'testing_accuracy': test_results.get('testing_accuracy', 0),
        'learning_rate': train_params['learning_rate'],
        'in_features_count': model_in_features,
        'hidden_units_count': hidden_units,
        'output_units_count': output_units,
        'dropout_rate': dropout_rate,
        'class_to_idx': training_dataset.class_to_idx
    }

    # Save the current model checkpoint
    save_model_checkpoint(model_file_path, model, arch, checkpoint_metadata)


if __name__ == '__main__':
    main()
