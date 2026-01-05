"""
Large-scale Evaluation Script
Supports various dataset formats and parallel processing
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm

# Import the refactored evaluation framework
# from diffusion_evaluator import DiffusionEvaluator, EvalConfig


# ==================== Dataset Loaders ====================

class DatasetLoader:
    """Base class for dataset loading"""
    
    @staticmethod
    def load_from_directory(
        directory: str,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
        recursive: bool = False
    ) -> List[str]:
        """
        Load all images from a directory
        
        Args:
            directory: Path to directory
            extensions: Tuple of valid image extensions
            recursive: If True, recursively search subdirectories
            
        Returns:
            List of image paths
        """
        directory = Path(directory)
        image_paths = []
        
        if recursive:
            # Recursively find all images in subdirectories
            for ext in extensions:
                image_paths.extend(directory.rglob(f"*{ext}"))
                image_paths.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            # Only search top-level directory
            for ext in extensions:
                image_paths.extend(directory.glob(f"*{ext}"))
                image_paths.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted([str(p) for p in image_paths])
    
    @staticmethod
    def load_from_csv(
        csv_path: str,
        image_col: str = "image_path",
        prompt_col: str = None
    ) -> Tuple[List[str], List[str]]:
        """Load images and optional prompts from CSV"""
        df = pd.read_csv(csv_path)
        
        image_paths = df[image_col].tolist()
        prompts = df[prompt_col].tolist() if prompt_col and prompt_col in df.columns else None
        
        return image_paths, prompts
    
    @staticmethod
    def load_from_txt(txt_path: str) -> List[str]:
        """Load image paths from text file (one per line)"""
        with open(txt_path, 'r') as f:
            paths = [line.strip() for line in f if line.strip()]
        return paths
    
    @staticmethod
    def load_multiple_directories(
        directories: List[str],
        labels: List[str] = None,
        recursive: bool = False
    ) -> Tuple[List[str], List[str]]:
        """
        Load images from multiple directories with labels
        
        Args:
            directories: List of directory paths
            labels: Optional labels for each directory
            recursive: If True, recursively search subdirectories
            
        Returns:
            Tuple of (image_paths, labels)
        """
        all_paths = []
        all_labels = []
        
        for i, directory in enumerate(directories):
            paths = DatasetLoader.load_from_directory(directory, recursive=recursive)
            all_paths.extend(paths)
            
            label = labels[i] if labels else Path(directory).name
            all_labels.extend([label] * len(paths))
        
        return all_paths, all_labels
    
    @staticmethod
    def load_with_subdirectory_labels(
        root_directory: str,
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')
    ) -> Tuple[List[str], List[str]]:
        """
        Load images from subdirectories, using subdirectory names as labels
        
        Example structure:
            root_directory/
                class_a/
                    img1.jpg
                    img2.jpg
                class_b/
                    img3.jpg
                    
        Returns:
            Tuple of (image_paths, labels) where labels are subdirectory names
        """
        root = Path(root_directory)
        image_paths = []
        labels = []
        
        # Get all subdirectories
        subdirs = [d for d in root.iterdir() if d.is_dir()]
        
        if not subdirs:
            # No subdirectories, treat as single directory
            paths = DatasetLoader.load_from_directory(str(root), extensions, recursive=False)
            return paths, [root.name] * len(paths)
        
        # Load from each subdirectory
        for subdir in sorted(subdirs):
            paths = DatasetLoader.load_from_directory(str(subdir), extensions, recursive=True)
            image_paths.extend(paths)
            labels.extend([subdir.name] * len(paths))
        
        return image_paths, labels
    
    @staticmethod
    def get_directory_structure(
        directory: str,
        max_depth: int = 3
    ) -> Dict:
        """
        Get directory structure with image counts
        
        Args:
            directory: Root directory
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary with directory structure and counts
        """
        def count_images(path: Path, depth: int = 0):
            if depth > max_depth:
                return {"path": str(path), "count": 0, "subdirs": {}}
            
            result = {
                "path": str(path),
                "count": 0,
                "subdirs": {}
            }
            
            # Count images in current directory
            for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
                result["count"] += len(list(path.glob(f"*{ext}")))
                result["count"] += len(list(path.glob(f"*{ext.upper()}")))
            
            # Recurse into subdirectories
            for subdir in sorted([d for d in path.iterdir() if d.is_dir()]):
                result["subdirs"][subdir.name] = count_images(subdir, depth + 1)
            
            return result
        
        return count_images(Path(directory))


# ==================== Evaluation Pipeline ====================

class EvaluationPipeline:
    """Pipeline for large-scale evaluation"""
    
    def __init__(self, config: 'EvalConfig'):
        self.config = config
        self.evaluator = None  # Lazy initialization
    
    def initialize_evaluator(self):
        """Initialize evaluator (lazy loading)"""
        if self.evaluator is None:
            # from diffusion_evaluator import DiffusionEvaluator
            from manifold_evaluator import DiffusionEvaluator
            self.evaluator = DiffusionEvaluator(self.config)
    
    def evaluate_dataset(
        self,
        image_paths: List[str],
        prompts: List[str] = None,
        metadata: Dict = None
    ) -> pd.DataFrame:
        """Evaluate a dataset and return results as DataFrame"""
        self.initialize_evaluator()
        
        # Run evaluation
        results = self.evaluator.evaluate_images(image_paths, prompts)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Add metadata if provided
        if metadata:
            for key, values in metadata.items():
                df[key] = values
        
        return df
    
    def evaluate_multiple_datasets(
        self,
        datasets: List[Tuple[str, List[str], List[str]]],  # (name, paths, prompts)
    ) -> pd.DataFrame:
        """Evaluate multiple datasets"""
        all_results = []
        
        for name, paths, prompts in datasets:
            print(f"\n{'='*60}")
            print(f"Evaluating dataset: {name}")
            print(f"{'='*60}")
            
            df = self.evaluate_dataset(
                paths,
                prompts,
                metadata={"dataset": [name] * len(paths)}
            )
            
            all_results.append(df)
        
        # Combine all results
        return pd.concat(all_results, ignore_index=True)
    
    def compare_real_vs_generated(
        self,
        real_dir: str,
        gen_dirs: List[str],
        gen_labels: List[str]
    ) -> pd.DataFrame:
        """Compare real images against multiple generated datasets"""
        # Load real images
        real_paths = DatasetLoader.load_from_directory(real_dir)
        
        # Load generated images
        datasets = [("real", real_paths, None)]
        
        for gen_dir, label in zip(gen_dirs, gen_labels):
            gen_paths = DatasetLoader.load_from_directory(gen_dir)
            datasets.append((label, gen_paths, None))
        
        # Evaluate all
        return self.evaluate_multiple_datasets(datasets)


# ==================== Analysis Tools ====================

class ResultAnalyzer:
    """Tools for analyzing evaluation results"""
    
    @staticmethod
    def compute_statistics(df: pd.DataFrame, group_by: str = None) -> pd.DataFrame:
        """Compute statistics on results"""
        if group_by:
            stats = df.groupby(group_by)["criterion"].agg([
                'count', 'mean', 'std', 'min', 'max',
                ('q25', lambda x: x.quantile(0.25)),
                ('median', lambda x: x.quantile(0.5)),
                ('q75', lambda x: x.quantile(0.75))
            ]).reset_index()
        else:
            stats = df["criterion"].describe().to_frame().T
        
        return stats
    
    @staticmethod
    def save_statistics(df: pd.DataFrame, output_path: str):
        """Save statistics to CSV"""
        stats = ResultAnalyzer.compute_statistics(df, group_by="dataset" if "dataset" in df.columns else None)
        stats.to_csv(output_path, index=False)
        print(f"Statistics saved to {output_path}")
    
    @staticmethod
    def find_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Find outliers based on z-score"""
        mean = df["criterion"].mean()
        std = df["criterion"].std()
        df["z_score"] = (df["criterion"] - mean) / std
        outliers = df[df["z_score"].abs() > threshold]
        return outliers
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, output_path: str):
        """Export full results to CSV"""
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")


# ==================== Command Line Interface ====================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Large-scale diffusion model evaluation"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dir", type=str,
        help="Directory containing images"
    )
    input_group.add_argument(
        "--csv", type=str,
        help="CSV file with image paths"
    )
    input_group.add_argument(
        "--txt", type=str,
        help="Text file with image paths (one per line)"
    )
    input_group.add_argument(
        "--multi-dir", type=str, nargs='+',
        help="Multiple directories to compare"
    )
    input_group.add_argument(
        "--class-dir", type=str,
        help="Directory with subdirectories as class labels"
    )
    
    # Directory traversal options
    parser.add_argument(
        "--recursive", "-r", action="store_true",
        help="Recursively search subdirectories for images"
    )
    parser.add_argument(
        "--show-structure", action="store_true",
        help="Show directory structure and exit (use with --dir or --class-dir)"
    )
    
    # CSV-specific options
    parser.add_argument(
        "--image-col", type=str, default="image_path",
        help="Column name for image paths in CSV"
    )
    parser.add_argument(
        "--prompt-col", type=str, default=None,
        help="Column name for prompts in CSV"
    )
    
    # Multi-directory options
    parser.add_argument(
        "--labels", type=str, nargs='+',
        help="Labels for multiple directories"
    )
    
    # Model configuration
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--num-noise", type=int, default=8,
        help="Number of noise samples per image"
    )
    parser.add_argument(
        "--time-frac", type=float, default=0.01,
        help="Time fraction for diffusion"
    )
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="Image size"
    )
    
    # Output options
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--return-terms", action="store_true",
        help="Return detailed terms"
    )
    
    # Analysis options
    parser.add_argument(
        "--compute-stats", action="store_true",
        help="Compute and save statistics"
    )
    parser.add_argument(
        "--find-outliers", action="store_true",
        help="Find and save outliers"
    )
    
    return parser.parse_args()


def main():
    """Main execution function"""
    args = parse_args()
    
    # Show directory structure if requested
    if args.show_structure:
        if args.dir:
            structure = DatasetLoader.get_directory_structure(args.dir)
            print_directory_structure(structure)
        elif args.class_dir:
            structure = DatasetLoader.get_directory_structure(args.class_dir)
            print_directory_structure(structure)
        else:
            print("Error: --show-structure requires --dir or --class-dir")
        return
    
    # Import here to avoid circular dependency
    # from diffusion_evaluator import EvalConfig
    from manifold_evaluator import EvalConfig
    
    # Create configuration
    config = EvalConfig(
        device=args.device,
        batch_size=args.batch_size,
        num_noise=args.num_noise,
        time_frac=args.time_frac,
        image_size=args.image_size,
        output_dir=args.output_dir,
        return_terms=args.return_terms,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    if args.dir:
        image_paths = DatasetLoader.load_from_directory(args.dir, recursive=args.recursive)
        prompts = None
        metadata = None
        print(f"Loaded {len(image_paths)} images from {args.dir}")
        if args.recursive:
            print("  (searched recursively in subdirectories)")
        
    elif args.csv:
        image_paths, prompts = DatasetLoader.load_from_csv(
            args.csv,
            image_col=args.image_col,
            prompt_col=args.prompt_col
        )
        metadata = None
        print(f"Loaded {len(image_paths)} images from {args.csv}")
        
    elif args.txt:
        image_paths = DatasetLoader.load_from_txt(args.txt)
        prompts = None
        metadata = None
        print(f"Loaded {len(image_paths)} images from {args.txt}")
        
    elif args.class_dir:
        image_paths, labels = DatasetLoader.load_with_subdirectory_labels(args.class_dir)
        prompts = None
        metadata = {"class": labels}
        print(f"Loaded {len(image_paths)} images from {args.class_dir}")
        print(f"  Found {len(set(labels))} classes: {sorted(set(labels))}")
        
    elif args.multi_dir:
        image_paths, labels = DatasetLoader.load_multiple_directories(
            args.multi_dir,
            args.labels,
            recursive=args.recursive
        )
        prompts = None
        metadata = {"dataset": labels}
        print(f"Loaded {len(image_paths)} images from {len(args.multi_dir)} directories")
        if args.recursive:
            print("  (searched recursively in subdirectories)")
        for directory in args.multi_dir:
            dir_count = sum(1 for l in labels if l == Path(directory).name or l in args.labels)
            print(f"  {directory}: {dir_count} images")
    
    if len(image_paths) == 0:
        print("Error: No images found!")
        return
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(config)
    
    # Run evaluation
    print("\nStarting evaluation...")
    df = pipeline.evaluate_dataset(image_paths, prompts, metadata)
    
    # Save results
    print("\nSaving results...")
    ResultAnalyzer.export_to_csv(
        df,
        os.path.join(args.output_dir, "full_results.csv")
    )
    
    # Compute statistics
    if args.compute_stats:
        print("\nComputing statistics...")
        ResultAnalyzer.save_statistics(
            df,
            os.path.join(args.output_dir, "statistics.csv")
        )
        
        # Print statistics by group if available
        if "dataset" in df.columns or "class" in df.columns:
            group_col = "dataset" if "dataset" in df.columns else "class"
            stats = ResultAnalyzer.compute_statistics(df, group_by=group_col)
            print(f"\nStatistics by {group_col}:")
            print(stats.to_string(index=False))
    
    # Find outliers
    if args.find_outliers:
        print("\nFinding outliers...")
        outliers = ResultAnalyzer.find_outliers(df)
        outliers.to_csv(
            os.path.join(args.output_dir, "outliers.csv"),
            index=False
        )
        print(f"Found {len(outliers)} outliers")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Total images evaluated: {len(df)}")
    print(f"Mean criterion: {df['criterion'].mean():.6f}")
    print(f"Std criterion: {df['criterion'].std():.6f}")
    print(f"Results saved to: {args.output_dir}")
    print("="*60)


def print_directory_structure(structure: Dict, indent: int = 0):
    """Pretty print directory structure"""
    path = Path(structure["path"])
    name = path.name if indent > 0 else str(path)
    count = structure["count"]
    
    prefix = "  " * indent
    if count > 0:
        print(f"{prefix}📁 {name} ({count} images)")
    else:
        print(f"{prefix}📁 {name}")
    
    # Print subdirectories
    for subdir_name, subdir_data in structure["subdirs"].items():
        print_directory_structure(subdir_data, indent + 1)
    
    # Print total if root
    if indent == 0:
        total = count_total_images(structure)
        print(f"\nTotal images: {total}")


def count_total_images(structure: Dict) -> int:
    """Count total images in structure"""
    total = structure["count"]
    for subdir_data in structure["subdirs"].values():
        total += count_total_images(subdir_data)
    return total


if __name__ == "__main__":
    main()