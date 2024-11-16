import argparse
import os

import yaml

# Import modules from packages
from packages.data.dataset import (
    load_category_mapping,
)
from packages.models.model_utils import (
    load_model_checkpoint,
    invert_class_to_idx,
    predict,
)
from packages.utils.utils import (
    get_device,
)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Predict image class using a trained model.')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('checkpoint_path', help='Path to the model checkpoint file')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return top K most likely classes (default: 5)')
    parser.add_argument('--category_names', type=str, default=None,
                        help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for inference if available')
    args = parser.parse_args()

    # Check if required arguments are provided
    if not args.image_path or not args.checkpoint_path:
        parser.error("the following arguments are required: image_path, checkpoint_path")

    # Load configurations (if needed for other settings)
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get the base directory from config
    base_dir = config.get('base_dir', '.')
    base_dir = os.path.abspath(base_dir)

    # Determine image_path
    image_path = os.path.abspath(args.image_path)
    if not os.path.exists(image_path):
        parser.error(f"Image file not found at {image_path}")
    print(f"Using image path: {image_path}")

    # Determine checkpoint_path
    checkpoint_path = os.path.abspath(args.checkpoint_path)
    if not os.path.exists(checkpoint_path):
        parser.error(f"Checkpoint file not found at {checkpoint_path}")
    print(f"Using checkpoint path: {checkpoint_path}")

    # Determine top_k
    top_k = args.top_k
    if top_k <= 0:
        parser.error("Argument --top_k must be a positive integer")
    print(f"Using top_k: {top_k}")

    # Determine category_names mapping
    if args.category_names:
        category_names_path = os.path.abspath(args.category_names)
        if not os.path.exists(category_names_path):
            parser.error(f"Category names file not found at {category_names_path}")
        print(f"Using category names mapping from: {category_names_path}")
    else:
        # Use default category mapping from config.yaml
        data_config = config['data']
        category_names_path = os.path.join(base_dir, data_config['category_mapping'])
        print(f"Using default category names mapping from config.yaml: {category_names_path}")

    # Load the category-to-name mapping
    cat_to_name = load_category_mapping(category_names_path)

    # Get device
    device = get_device(use_gpu=args.gpu)

    # Load the model with the trained weights
    model, model_metadata = load_model_checkpoint(checkpoint_path, device)

    # Invert class_to_idx to create idx_to_class mapping
    idx_to_class = invert_class_to_idx(model.class_to_idx)

    # Predict the top labels and probabilities for the image
    top_labels, top_ps_perc = predict(
        image_path=image_path,
        model=model,
        cat_to_name=cat_to_name,
        idx_to_class=idx_to_class,
        topk=top_k
    )

    # Output the predicted class names and confidence levels
    print("\nPrediction Results:")
    for label, probability in zip(top_labels, top_ps_perc):
        print(f"Class Name: {label}, Confidence: {probability}%")


if __name__ == '__main__':
    main()
