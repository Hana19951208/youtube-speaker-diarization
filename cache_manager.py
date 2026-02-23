"""
Simplified cache and checkpoint management for YouTube Speaker Pipeline.

Provides:
- Directory structure management (WORK_DIR, CACHE_DIR, RUN_DIR)
- Environment variable setup for model caching
- Run state tracking
- Step completion detection and skip logic
"""
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheManager:
    """Simplified cache manager for checkpoint/resume functionality."""
    
    def __init__(
        self,
        work_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        youtube_url: Optional[str] = None,
        resume: bool = True,
        force: bool = False,
    ):
        """Initialize cache manager."""
        self.resume = resume
        self.force = force
        self.timing = {}
        
        # Setup directories
        self._setup_directories(work_dir, cache_dir)
        self._setup_run_id(run_id, youtube_url)
        self._setup_environment()
        
        # Initialize run state
        self.run_state_path = self.run_dir / "run_state.json"
        self._load_or_init_run_state()
        
        logger.info(f"CacheManager initialized: run_id={self.run_id}")
        logger.info(f"  Work dir: {self.work_dir}")
        logger.info(f"  Run dir: {self.run_dir}")
        logger.info(f"  Cache dir: {self.cache_dir}")
    
    def _setup_directories(self, work_dir: Optional[str], cache_dir: Optional[str]):
        """Setup working and cache directories."""
        # Detect platform
        self.is_colab = self._detect_colab()
        self.is_kaggle = self._detect_kaggle()
        
        # Set work directory
        if work_dir:
            self.work_dir = Path(work_dir)
        elif self.is_colab:
            self.work_dir = Path("/content/work")
        elif self.is_kaggle:
            self.work_dir = Path("/kaggle/working/work")
        else:
            self.work_dir = Path("./work")
        
        # Set cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        elif self.is_colab:
            drive_path = Path("/content/drive/MyDrive")
            if drive_path.exists():
                self.cache_dir = drive_path / "asr_cache"
                logger.info("Google Drive mounted, using Drive for cache")
            else:
                self.cache_dir = Path("/content/cache")
                logger.warning("Google Drive not mounted, using local Colab cache (will not persist)")
        elif self.is_kaggle:
            self.cache_dir = Path("/kaggle/working/cache")
            logger.info("Using Kaggle working directory for cache")
        else:
            self.cache_dir = Path("./cache")
        
        # Create directories
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _detect_colab(self) -> bool:
        """Detect if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _detect_kaggle(self) -> bool:
        """Detect if running in Kaggle."""
        return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    
    def _setup_run_id(self, run_id: Optional[str], youtube_url: Optional[str]):
        """Setup unique run identifier."""
        if run_id:
            self.run_id = run_id
        elif youtube_url:
            video_id = self._extract_video_id(youtube_url)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{video_id}_{timestamp}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"run_{timestamp}"
        
        # Setup run directory
        self.runs_dir = self.work_dir / "runs"
        self.run_dir = self.runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def _extract_video_id(self, youtube_url: str) -> str:
        """Extract video ID from YouTube URL."""
        import re
        
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        
        import hashlib
        return hashlib.md5(youtube_url.encode()).hexdigest()[:11]
    
    def _setup_environment(self):
        """Setup environment variables for caching."""
        huggingface_cache = self.cache_dir / "huggingface"
        torch_cache = self.cache_dir / "torch"
        xdg_cache = self.cache_dir / "xdg"
        pip_cache = self.cache_dir / "pip"
        
        huggingface_cache.mkdir(parents=True, exist_ok=True)
        torch_cache.mkdir(parents=True, exist_ok=True)
        xdg_cache.mkdir(parents=True, exist_ok=True)
        pip_cache.mkdir(parents=True, exist_ok=True)
        
        env_vars = {
            "HF_HOME": str(huggingface_cache),
            "TRANSFORMERS_CACHE": str(huggingface_cache / "transformers"),
            "HF_HUB_CACHE": str(huggingface_cache / "hub"),
            "TORCH_HOME": str(torch_cache),
            "XDG_CACHE_HOME": str(xdg_cache),
            "PIP_CACHE_DIR": str(pip_cache),
            "WHISPER_CACHE_DIR": str(huggingface_cache / "whisper"),
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
        
        logger.info("Environment variables set for caching")
    
    def _load_or_init_run_state(self):
        """Load existing run state or initialize new one."""
        if self.run_state_path.exists() and self.resume and not self.force:
            with open(self.run_state_path, "r") as f:
                self.run_state = json.load(f)
            logger.info(f"Loaded existing run state: {self.run_state.get('status', 'unknown')}")
        else:
            if self.force:
                logger.info("Force mode enabled - starting fresh run")
            else:
                logger.info("Initializing new run state")
            
            self.run_state = {
                "run_id": self.run_id,
                "status": "initialized",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "config": {},
                "steps": {},
                "inputs_hash": None,
            }
            self._save_run_state()
    
    def _save_run_state(self):
        """Save run state to file atomically."""
        self.run_state["updated_at"] = datetime.now().isoformat()
        
        temp_path = self.run_state_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(self.run_state, f, indent=2)
        
        os.replace(temp_path, self.run_state_path)
    
    @contextmanager
    def step_context(self, step_name: str):
        """Context manager for running a step with state tracking."""
        self.update_step_status(step_name, "running")
        start_time = time.time()
        
        try:
            yield
            self.update_step_status(step_name, "done", duration=time.time() - start_time)
        except Exception as e:
            self.update_step_status(step_name, "failed", error=str(e), duration=time.time() - start_time)
            raise
    
    def update_step_status(self, step_name: str, status: str, duration: float = None, error: str = None):
        """Update step status in run state."""
        if step_name not in self.run_state["steps"]:
            self.run_state["steps"][step_name] = {}
        
        step_info = self.run_state["steps"][step_name]
        step_info["status"] = status
        
        if status == "running":
            step_info["started_at"] = datetime.now().isoformat()
        elif status in ["done", "failed"]:
            step_info["ended_at"] = datetime.now().isoformat()
            if duration:
                step_info["duration_sec"] = duration
        
        if error:
            step_info["error"] = error
        
        self._save_run_state()
        logger.info(f"Step '{step_name}' status: {status}")
    
    def should_run_step(self, step_name: str) -> bool:
        """Check if a step should be run."""
        if self.force:
            return True
        if not self.resume:
            return True
        
        step_info = self.run_state.get("steps", {}).get(step_name, {})
        return step_info.get("status") != "done"
    
    def get_step_file_path(self, filename: str) -> str:
        """Get full path for a file in the run directory."""
        return str(self.run_dir / filename)


if __name__ == "__main__":
    print("Cache manager module loaded successfully")
