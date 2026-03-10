"""
=============================================================================
Run Manager
=============================================================================
Manages experiment runs with sequential IDs, directory structure, and
configuration snapshots.

Usage:
    from tools.run_manager import RunManager
    
    # Create new run
    run_manager = RunManager.create_new_run(config)
    
    # Load existing run
    run_manager = RunManager.load_run("run_001")
=============================================================================
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import yaml


@dataclass
class RunInfo:
    """Container for run information."""
    run_id: str
    run_dir: Path
    logs_dir: Path
    models_dir: Path
    results_dir: Path
    config_snapshot_path: Path
    created_at: str
    operations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "created_at": self.created_at,
            "operations": self.operations
        }


class RunManager:
    """
    Manages experiment runs with sequential IDs.
    
    Directory structure per run:
        run_XXX/
        ├── config_snapshot.yaml    # Config used for this run
        ├── run_info.yaml           # Run metadata
        ├── logs/                   # Log files per operation
        │   ├── train_2d.log
        │   └── inference_2d.log
        ├── models/                 # Trained model weights
        │   └── train_run/
        │       └── weights/
        │           ├── best.pt
        │           └── last.pt
        └── results/                # Inference outputs
            ├── rgb_2d_detected.avi
            └── rgb_detections.json
    """
    
    DEFAULT_RUNS_DIR = "outputs/runs"
    RUN_PREFIX = "run_"
    
    def __init__(self, run_info: RunInfo):
        """
        Initialize run manager with existing run info.
        
        Args:
            run_info: RunInfo dataclass with run details
        """
        self.run_info = run_info
    
    @classmethod
    def get_runs_base_dir(cls, project_root: Optional[Path] = None) -> Path:
        """Get the base directory for all runs."""
        if project_root is None:
            project_root = Path(__file__).parent.parent
        return project_root / cls.DEFAULT_RUNS_DIR
    
    @classmethod
    def _get_next_run_id(cls, runs_dir: Path) -> str:
        """
        Get the next sequential run ID.
        
        Scans existing run directories and returns the next available ID.
        Format: run_001, run_002, etc.
        
        Args:
            runs_dir: Base directory containing all runs
            
        Returns:
            Next available run ID (e.g., "run_003")
        """
        runs_dir.mkdir(parents=True, exist_ok=True)
        
        # Find existing run directories
        existing_runs = []
        pattern = re.compile(rf"^{cls.RUN_PREFIX}(\d+)$")
        
        for item in runs_dir.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    existing_runs.append(int(match.group(1)))
        
        # Get next ID
        next_id = max(existing_runs, default=0) + 1
        
        return f"{cls.RUN_PREFIX}{next_id:03d}"
    
    @classmethod
    def list_runs(cls, project_root: Optional[Path] = None) -> List[str]:
        """
        List all existing run IDs.
        
        Args:
            project_root: Project root directory
            
        Returns:
            List of run IDs sorted by number
        """
        runs_dir = cls.get_runs_base_dir(project_root)
        
        if not runs_dir.exists():
            return []
        
        runs = []
        pattern = re.compile(rf"^{cls.RUN_PREFIX}(\d+)$")
        
        for item in runs_dir.iterdir():
            if item.is_dir() and pattern.match(item.name):
                runs.append(item.name)
        
        # Sort by number
        runs.sort(key=lambda x: int(x.replace(cls.RUN_PREFIX, "")))
        
        return runs
    
    @classmethod
    def run_exists(cls, run_id: str, project_root: Optional[Path] = None) -> bool:
        """
        Check if a run ID exists.
        
        Args:
            run_id: Run ID to check (e.g., "run_001")
            project_root: Project root directory
            
        Returns:
            True if run exists
        """
        runs_dir = cls.get_runs_base_dir(project_root)
        run_dir = runs_dir / run_id
        return run_dir.exists() and run_dir.is_dir()
    
    @classmethod
    def create_new_run(
        cls,
        config: Dict[str, Any],
        project_root: Optional[Path] = None
    ) -> "RunManager":
        """
        Create a new run with sequential ID.
        
        Creates directory structure and saves config snapshot.
        
        Args:
            config: Configuration dictionary to snapshot
            project_root: Project root directory
            
        Returns:
            RunManager instance for the new run
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent
        
        runs_dir = cls.get_runs_base_dir(project_root)
        run_id = cls._get_next_run_id(runs_dir)
        run_dir = runs_dir / run_id
        
        # Create directory structure
        logs_dir = run_dir / "logs"
        models_dir = run_dir / "models"
        results_dir = run_dir / "results"
        
        logs_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config snapshot
        config_snapshot_path = run_dir / "config_snapshot.yaml"
        with open(config_snapshot_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Create run info
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        run_info = RunInfo(
            run_id=run_id,
            run_dir=run_dir,
            logs_dir=logs_dir,
            models_dir=models_dir,
            results_dir=results_dir,
            config_snapshot_path=config_snapshot_path,
            created_at=created_at,
            operations=[]
        )
        
        # Save run info
        run_info_path = run_dir / "run_info.yaml"
        with open(run_info_path, 'w', encoding='utf-8') as f:
            yaml.dump(run_info.to_dict(), f, default_flow_style=False)
        
        return cls(run_info)
    
    @classmethod
    def load_run(
        cls,
        run_id: str,
        project_root: Optional[Path] = None
    ) -> "RunManager":
        """
        Load an existing run.
        
        Args:
            run_id: Run ID to load (e.g., "run_001")
            project_root: Project root directory
            
        Returns:
            RunManager instance for the existing run
            
        Raises:
            ValueError: If run doesn't exist
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent
        
        runs_dir = cls.get_runs_base_dir(project_root)
        run_dir = runs_dir / run_id
        
        if not run_dir.exists():
            available = cls.list_runs(project_root)
            available_str = ", ".join(available) if available else "none"
            raise ValueError(
                f"Run '{run_id}' not found.\n"
                f"Available runs: {available_str}\n"
                f"Create a new run with: python src/train_2d.py --config configs/config.yaml"
            )
        
        # Load run info
        run_info_path = run_dir / "run_info.yaml"
        
        if run_info_path.exists():
            with open(run_info_path, 'r', encoding='utf-8') as f:
                info_data = yaml.safe_load(f)
        else:
            info_data = {
                "created_at": "unknown",
                "operations": []
            }
        
        run_info = RunInfo(
            run_id=run_id,
            run_dir=run_dir,
            logs_dir=run_dir / "logs",
            models_dir=run_dir / "models",
            results_dir=run_dir / "results",
            config_snapshot_path=run_dir / "config_snapshot.yaml",
            created_at=info_data.get("created_at", "unknown"),
            operations=info_data.get("operations", [])
        )
        
        return cls(run_info)
    
    def add_operation(self, operation_name: str) -> None:
        """
        Record an operation in the run.
        
        Args:
            operation_name: Name of the operation (e.g., "train_2d")
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.run_info.operations.append(f"{operation_name} ({timestamp})")
        
        # Update run_info.yaml
        run_info_path = self.run_info.run_dir / "run_info.yaml"
        with open(run_info_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.run_info.to_dict(), f, default_flow_style=False)
    
    def get_log_path(self, operation_name: str) -> Path:
        """
        Get log file path for an operation.
        
        Args:
            operation_name: Name of the operation (e.g., "train_2d")
            
        Returns:
            Path to the log file
        """
        return self.run_info.logs_dir / f"{operation_name}.log"
    
    def get_model_path(self, filename: str = "best.pt") -> Path:
        """
        Get path to a model file.
        
        Args:
            filename: Model filename (default: "best.pt")
            
        Returns:
            Path to the model file
        """
        return self.run_info.models_dir / "train_run" / "weights" / filename
    
    def get_results_path(self, filename: str) -> Path:
        """
        Get path to a results file.
        
        Args:
            filename: Results filename
            
        Returns:
            Path to the results file
        """
        return self.run_info.results_dir / filename
    
    @property
    def run_id(self) -> str:
        """Get run ID."""
        return self.run_info.run_id
    
    @property
    def run_dir(self) -> Path:
        """Get run directory."""
        return self.run_info.run_dir
    
    @property
    def logs_dir(self) -> Path:
        """Get logs directory."""
        return self.run_info.logs_dir
    
    @property
    def models_dir(self) -> Path:
        """Get models directory."""
        return self.run_info.models_dir
    
    @property
    def results_dir(self) -> Path:
        """Get results directory."""
        return self.run_info.results_dir
