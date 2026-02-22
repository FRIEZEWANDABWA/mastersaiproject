"""
FRIENDS Framework Compliance Checker

This script automatically assesses project status against the FRIENDS framework
criteria (Adhikari, 2020), focusing on the 'Feasible' and 'Narrow' aspects
through dataset analysis.

Author: WANDABWA Frieze (ST62/55175/2025)
Project: MSc AI - Maize Disease Classification
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


# Project configuration
PROJECT_ROOT = Path(r'C:\websites\mastersaiproject')
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
REQUIRED_CLASSES = ['maize_healthy', 'maize_streak', 'maize_mln']

# Minimum dataset requirements for feasibility
MIN_IMAGES_PER_CLASS = 50  # Minimum for basic training
RECOMMENDED_IMAGES_PER_CLASS = 200  # Recommended for good performance
OPTIMAL_IMAGES_PER_CLASS = 500  # Optimal for robust model


def count_images_in_directory(directory: Path) -> int:
    """
    Count image files in a directory.
    
    Args:
        directory: Path to directory
    
    Returns:
        int: Number of image files found
    """
    if not directory.exists():
        return 0
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    count = 0
    
    for file in directory.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            count += 1
    
    return count


def analyze_dataset() -> Dict[str, int]:
    """
    Analyze the dataset structure and count images per class.
    
    Returns:
        dict: Class names mapped to image counts
    """
    dataset_info = {}
    
    for class_name in REQUIRED_CLASSES:
        class_dir = DATA_DIR / class_name
        image_count = count_images_in_directory(class_dir)
        dataset_info[class_name] = image_count
    
    return dataset_info


def assess_feasibility(dataset_info: Dict[str, int]) -> Tuple[str, str, List[str]]:
    """
    Assess project feasibility based on dataset size.
    
    Args:
        dataset_info: Dictionary of class names to image counts
    
    Returns:
        tuple: (status, message, recommendations)
    """
    total_images = sum(dataset_info.values())
    min_class_size = min(dataset_info.values()) if dataset_info else 0
    
    recommendations = []
    
    # Check if dataset exists
    if total_images == 0:
        status = "❌ NOT FEASIBLE"
        message = "No images found in dataset. Data collection required."
        recommendations.append("Start collecting images from your farm")
        recommendations.append(f"Target: At least {MIN_IMAGES_PER_CLASS} images per class")
        recommendations.append("Ensure balanced distribution across all 3 classes")
    
    # Check minimum requirements
    elif min_class_size < MIN_IMAGES_PER_CLASS:
        status = "⚠️ PARTIALLY FEASIBLE"
        message = f"Dataset too small. Minimum class has only {min_class_size} images."
        recommendations.append(f"Collect at least {MIN_IMAGES_PER_CLASS - min_class_size} more images for smallest class")
        recommendations.append("Aim for balanced classes (similar number of images)")
    
    # Check recommended size
    elif min_class_size < RECOMMENDED_IMAGES_PER_CLASS:
        status = "✅ FEASIBLE (Basic)"
        message = f"Dataset meets minimum requirements ({total_images} total images)."
        recommendations.append(f"Consider collecting more images (target: {RECOMMENDED_IMAGES_PER_CLASS} per class)")
        recommendations.append("More data will improve model performance and generalization")
    
    # Check optimal size
    elif min_class_size < OPTIMAL_IMAGES_PER_CLASS:
        status = "✅ FEASIBLE (Good)"
        message = f"Dataset size is good ({total_images} total images)."
        recommendations.append(f"Optional: Collect more images for optimal performance (target: {OPTIMAL_IMAGES_PER_CLASS} per class)")
        recommendations.append("Current dataset should produce reliable results")
    
    else:
        status = "✅ FEASIBLE (Optimal)"
        message = f"Dataset size is optimal ({total_images} total images)."
        recommendations.append("Dataset meets all requirements!")
        recommendations.append("Ready to proceed with model training")
    
    return status, message, recommendations


def assess_narrowness(dataset_info: Dict[str, int]) -> Tuple[str, str]:
    """
    Assess if project scope remains narrow and focused.
    
    Args:
        dataset_info: Dictionary of class names to image counts
    
    Returns:
        tuple: (status, message)
    """
    # Check if we have exactly 3 classes (narrow scope)
    actual_classes = [cls for cls, count in dataset_info.items() if count > 0]
    
    if len(actual_classes) == 3:
        status = "✅ NARROW"
        message = "Project maintains focused scope: 3 specific maize disease states."
    elif len(actual_classes) < 3:
        status = "⚠️ INCOMPLETE"
        message = f"Only {len(actual_classes)}/3 classes have data. Complete all classes."
    else:
        status = "⚠️ SCOPE CREEP"
        message = "Additional classes detected. Maintain focus on 3 core states."
    
    return status, message


def check_class_balance(dataset_info: Dict[str, int]) -> Tuple[str, str, List[str]]:
    """
    Check if dataset is balanced across classes.
    
    Args:
        dataset_info: Dictionary of class names to image counts
    
    Returns:
        tuple: (status, message, recommendations)
    """
    if not dataset_info or sum(dataset_info.values()) == 0:
        return "⚠️ NO DATA", "No images to analyze", []
    
    counts = list(dataset_info.values())
    max_count = max(counts)
    min_count = min(counts)
    
    recommendations = []
    
    # Calculate imbalance ratio
    if min_count == 0:
        status = "❌ IMBALANCED"
        message = "Some classes have no images."
        recommendations.append("Collect images for all 3 disease states")
    elif max_count / min_count > 2.0:
        status = "⚠️ IMBALANCED"
        message = f"Class imbalance detected (ratio: {max_count/min_count:.2f}:1)"
        recommendations.append("Try to balance classes (similar number of images)")
        recommendations.append("Imbalanced data can bias the model")
    elif max_count / min_count > 1.5:
        status = "⚠️ SLIGHTLY IMBALANCED"
        message = f"Minor class imbalance (ratio: {max_count/min_count:.2f}:1)"
        recommendations.append("Consider balancing classes for optimal performance")
    else:
        status = "✅ BALANCED"
        message = "Dataset is well-balanced across classes."
        recommendations.append("Good class distribution!")
    
    return status, message, recommendations


def generate_report():
    """
    Generate comprehensive project status report.
    """
    print("=" * 70)
    print("FRIENDS FRAMEWORK COMPLIANCE REPORT")
    print("=" * 70)
    print(f"Project: Maize Disease Classification using Deep Learning")
    print(f"Candidate: WANDABWA Frieze (ST62/55175/2025)")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Analyze dataset
    dataset_info = analyze_dataset()
    
    print("\n📊 DATASET OVERVIEW")
    print("-" * 70)
    for class_name, count in dataset_info.items():
        print(f"   {class_name:20s}: {count:4d} images")
    print(f"   {'TOTAL':20s}: {sum(dataset_info.values()):4d} images")
    
    # Feasibility assessment
    print("\n🔍 FEASIBILITY ASSESSMENT (F in FRIENDS)")
    print("-" * 70)
    feasibility_status, feasibility_msg, feasibility_recs = assess_feasibility(dataset_info)
    print(f"   Status: {feasibility_status}")
    print(f"   {feasibility_msg}")
    if feasibility_recs:
        print("\n   Recommendations:")
        for rec in feasibility_recs:
            print(f"   • {rec}")
    
    # Narrowness assessment
    print("\n🎯 NARROWNESS ASSESSMENT (N in FRIENDS)")
    print("-" * 70)
    narrowness_status, narrowness_msg = assess_narrowness(dataset_info)
    print(f"   Status: {narrowness_status}")
    print(f"   {narrowness_msg}")
    
    # Balance assessment
    print("\n⚖️ CLASS BALANCE ASSESSMENT")
    print("-" * 70)
    balance_status, balance_msg, balance_recs = check_class_balance(dataset_info)
    print(f"   Status: {balance_status}")
    print(f"   {balance_msg}")
    if balance_recs:
        print("\n   Recommendations:")
        for rec in balance_recs:
            print(f"   • {rec}")
    
    # Overall status
    print("\n" + "=" * 70)
    print("OVERALL PROJECT STATUS")
    print("=" * 70)
    
    total_images = sum(dataset_info.values())
    if total_images >= MIN_IMAGES_PER_CLASS * 3:
        print("✅ Project is READY for model development")
        print("\n   Next Steps:")
        print("   1. Run: python src/data_loader.py (test data loading)")
        print("   2. Build CNN model architecture")
        print("   3. Begin training with current dataset")
    else:
        print("⚠️ Project needs MORE DATA before model training")
        print("\n   Next Steps:")
        print("   1. Continue data collection from farm")
        print(f"   2. Target: {MIN_IMAGES_PER_CLASS * 3} total images minimum")
        print("   3. Re-run this script to check progress")
    
    print("=" * 70)


if __name__ == "__main__":
    try:
        generate_report()
    except Exception as e:
        print(f"\n❌ Error generating report: {e}")
        print("\nPlease ensure:")
        print("  • Project directory exists at C:\\websites\\mastersaiproject")
        print("  • Data folders exist: data/raw/maize_healthy, maize_streak, maize_mln")
