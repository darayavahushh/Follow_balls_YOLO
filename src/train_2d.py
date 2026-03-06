"""
=============================================================================
Football 2D Detection - Training Pipeline
=============================================================================
Fine-tunes YOLO26 on a soccer dataset for detecting persons and ball.

Usage:
    python src/train_2d.py --config configs/config.yaml
    python src/train_2d.py --config configs/config.yaml --epochs 50

Author: [Your Name]
Challenge: Computer Vision - Football Detection & Tracking
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Third-party imports
import yaml
from tqdm import tqdm

# Local imports from tools
from tools.config_loader import load_config, resolve_paths


# =============================================================================
# DATASET PREPROCESSING
# =============================================================================

def remap_label_file(
    input_path: Path, 
    output_path: Path, 
    class_mapping: Dict[int, int]
) -> int:
    """
    Remap class IDs in a single YOLO format label file.
    
    YOLO format per line: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]
    
    Args:
        input_path: Source label file (.txt)
        output_path: Destination label file (.txt)
        class_mapping: Dict mapping old_class_id -> new_class_id
    
    Returns:
        Number of annotations processed
    """
    annotations_count = 0
    new_lines = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 5:
                continue
            
            old_class_id = int(parts[0])
            
            if old_class_id not in class_mapping:
                continue
            
            new_class_id = class_mapping[old_class_id]
            parts[0] = str(new_class_id)
            new_lines.append(" ".join(parts))
            annotations_count += 1
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_lines))
    
    return annotations_count


def prepare_dataset(config: Dict[str, Any]) -> Path:
    """
    Prepare dataset with remapped classes.
    
    Creates structure:
        dataset_processed/
        ├── images/
        │   ├── train/
        │   └── val/
        ├── labels/
        │   ├── train/
        │   └── val/
        └── data.yaml
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Path to generated data.yaml
    """
    print("\n" + "=" * 60)
    print("📁 PREPARING DATASET")
    print("=" * 60)
    
    source_dir = Path(config['paths']['source_dataset'])
    output_dir = Path(config['paths']['processed_dataset'])
    
    # Convert string keys to int for class mapping
    class_mapping = {int(k): v for k, v in config['classes']['mapping'].items()}
    class_names = {int(k): v for k, v in config['classes']['names'].items()}
    
    # Validate source directory
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")
    
    # Clean previous processed data
    if output_dir.exists():
        print(f"Removing existing processed dataset at {output_dir}")
        shutil.rmtree(output_dir)
    
    # Map source splits to destination splits
    split_mapping = {"train": "train", "test": "val"}
    stats = {"images": 0, "labels": 0, "annotations": 0}
    
    for src_split, dst_split in split_mapping.items():
        print(f"\n→ Processing '{src_split}' → '{dst_split}'...")
        
        src_images_dir = source_dir / "images" / src_split
        src_labels_dir = source_dir / "labels" / src_split
        dst_images_dir = output_dir / "images" / dst_split
        dst_labels_dir = output_dir / "labels" / dst_split
        
        # Validate source directories
        if not src_images_dir.exists():
            print(f"  ⚠️ Warning: {src_images_dir} not found, skipping...")
            continue
            
        dst_images_dir.mkdir(parents=True, exist_ok=True)
        dst_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [
            f for f in src_images_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        print(f"  Found {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=f"  Copying {dst_split}"):
            # Copy image
            dst_img_path = dst_images_dir / img_path.name
            shutil.copy2(img_path, dst_img_path)
            stats["images"] += 1
            
            # Process corresponding label
            label_name = img_path.stem + ".txt"
            src_label_path = src_labels_dir / label_name
            dst_label_path = dst_labels_dir / label_name
            
            if src_label_path.exists():
                ann_count = remap_label_file(
                    src_label_path, 
                    dst_label_path, 
                    class_mapping
                )
                stats["labels"] += 1
                stats["annotations"] += ann_count
            else:
                # Create empty label file (background image)
                dst_label_path.touch()
    
    # Create data.yaml for YOLO training
    data_yaml_content = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "names": class_names,
        "nc": config['classes']['num_classes']
    }
    
    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Dataset prepared successfully!")
    print(f"   📊 Images: {stats['images']}")
    print(f"   📝 Labels: {stats['labels']}")
    print(f"   🏷️  Annotations: {stats['annotations']}")
    print(f"   📄 Config: {yaml_path}")
    
    return yaml_path


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model(data_yaml_path: Path, config: Dict[str, Any]) -> Path:
    """
    Fine-tune YOLO26 on the prepared dataset.
    
    Args:
        data_yaml_path: Path to data.yaml configuration
        config: Configuration dictionary
    
    Returns:
        Path to best model weights
    """
    # Import here to avoid loading heavy dependencies during preprocessing
    from ultralytics import YOLO
    
    model_config = config['model']
    train_config = config['training']
    
    print("\n" + "=" * 60)
    print("🚀 TRAINING MODEL")
    print("=" * 60)
    print(f"   Model: {model_config['name']}")
    print(f"   Epochs: {train_config['epochs']}")
    print(f"   Image Size: {model_config['image_size']}")
    print(f"   Batch Size: {train_config['batch_size']}")
    print("=" * 60)
    
    # Load pretrained model
    model = YOLO(model_config['name'])
    
    # Create output directory
    models_dir = Path(config['paths']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Build training arguments from config
    train_args = {
        # Dataset
        "data": str(data_yaml_path),
        
        # Training duration
        "epochs": train_config['epochs'],
        "patience": train_config['patience'],
        
        # Input
        "imgsz": model_config['image_size'],
        "batch": train_config['batch_size'],
        
        # Output
        "project": str(models_dir),
        "name": "train_run",
        "exist_ok": True,
        
        # Augmentation
        "augment": train_config['augmentation']['enabled'],
        "mosaic": train_config['augmentation']['mosaic'],
        "mixup": train_config['augmentation']['mixup'],
        "copy_paste": train_config['augmentation']['copy_paste'],
        "scale": train_config['augmentation']['scale'],
        
        # Optimization
        "optimizer": train_config['optimizer']['name'],
        "lr0": train_config['optimizer']['lr_initial'],
        "lrf": train_config['optimizer']['lr_final_factor'],
        "weight_decay": train_config['optimizer']['weight_decay'],
        "warmup_epochs": train_config['optimizer']['warmup_epochs'],
        
        # Loss weights
        "box": train_config['loss_weights']['box'],
        "cls": train_config['loss_weights']['cls'],
        
        # Hardware
        "device": train_config['hardware']['device'],
        "workers": train_config['hardware']['workers'],
        
        # Logging
        "verbose": True,
        "plots": True,
        "save": True,
        "save_period": train_config['checkpoints']['save_period'],
    }
    
    # Start training
    model.train(**train_args)
    
    best_model_path = models_dir / "train_run" / "weights" / "best.pt"
    
    print(f"\n✅ Training complete!")
    print(f"   📦 Best model saved to: {best_model_path}")
    
    return best_model_path


# =============================================================================
# VALIDATION
# =============================================================================

def validate_model(
    model_path: Path, 
    data_yaml_path: Path, 
    config: Dict[str, Any]
) -> Any:
    """
    Run validation metrics on trained model.
    
    Args:
        model_path: Path to trained weights (best.pt)
        data_yaml_path: Path to data.yaml
        config: Configuration dictionary
    
    Returns:
        Validation metrics object
    """
    from ultralytics import YOLO
    
    print("\n" + "=" * 60)
    print("📊 VALIDATING MODEL")
    print("=" * 60)
    
    model = YOLO(str(model_path))
    metrics = model.val(data=str(data_yaml_path), verbose=True)
    
    class_names = {int(k): v for k, v in config['classes']['names'].items()}
    
    print(f"\n   Validation Results:")
    print(f"   mAP50:    {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    
    print(f"\n   Per-class AP50:")
    for i, ap in enumerate(metrics.box.ap50):
        class_name = class_names.get(i, f"class_{i}")
        print(f"     {class_name}: {ap:.4f}")
    
    return metrics


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """Main training pipeline execution."""
    
    parser = argparse.ArgumentParser(
        description="Train YOLO26 for football detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs from config"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size from config"
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip dataset preprocessing"
    )
    
    args = parser.parse_args()
    
    # Load and resolve config
    config = load_config(args.config)
    config = resolve_paths(config, PROJECT_ROOT)
    
    # Apply CLI overrides
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    print("\n" + "=" * 60)
    print("⚽ BALL 2D DETECTION - TRAINING PIPELINE")
    print("=" * 60)
    print(f"   Config: {args.config}")
    print(f"   Project Root: {PROJECT_ROOT}")
    
    # Step 1: Prepare dataset
    if not args.skip_preprocessing:
        data_yaml_path = prepare_dataset(config)
    else:
        data_yaml_path = Path(config['paths']['processed_dataset']) / "data.yaml"
        if not data_yaml_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {data_yaml_path}\n"
                "Run without --skip-preprocessing first."
            )
        print(f"⏭️  Skipping preprocessing, using: {data_yaml_path}")
    
    # Step 2: Train model
    best_model_path = train_model(data_yaml_path, config)
    
    # Step 3: Validate
    validate_model(best_model_path, data_yaml_path, config)
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n   Trained model: {best_model_path}")
    print(f"\n   Next step - Run inference:")
    print(f"   python src/inference_2d.py --config {args.config}")


if __name__ == "__main__":
    main()
