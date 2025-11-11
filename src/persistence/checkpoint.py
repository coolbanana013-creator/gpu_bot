"""
Persistence utilities for saving and loading evolution state.

Allows resuming evolution from checkpoints.
"""
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..bot_generator.compact_generator import CompactBotConfig
from ..backtester.compact_simulator import BacktestResult


class EvolutionCheckpoint:
    """Manages saving and loading evolution state."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def save_checkpoint(
        self,
        generation: int,
        population: List[CompactBotConfig],
        results: List[BacktestResult],
        best_fitness: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save evolution state to checkpoint file.
        
        Args:
            generation: Current generation number
            population: Current population
            results: Backtest results for population
            best_fitness: Best fitness score in current generation
            metadata: Optional metadata (params, timestamps, etc.)
            
        Returns:
            Path to saved checkpoint file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = self.checkpoint_dir / f"checkpoint_gen{generation}_{timestamp}.pkl"
        
        checkpoint_data = {
            'generation': generation,
            'population': population,
            'results': results,
            'best_fitness': best_fitness,
            'metadata': metadata or {},
            'timestamp': timestamp
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Also save metadata as JSON for easier inspection
        json_file = checkpoint_file.with_suffix('.json')
        metadata_json = {
            'generation': generation,
            'best_fitness': float(best_fitness),
            'population_size': len(population),
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata_json, f, indent=2)
            
        return checkpoint_file
        
    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """
        Load evolution state from checkpoint file.
        
        Args:
            checkpoint_file: Path to checkpoint file (.pkl)
            
        Returns:
            Dictionary with keys: generation, population, results, best_fitness, metadata
        """
        checkpoint_path = Path(checkpoint_file)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
            
        return checkpoint_data
        
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint metadata dictionaries
        """
        checkpoints = []
        
        for json_file in sorted(self.checkpoint_dir.glob("checkpoint_*.json")):
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                    metadata['file'] = str(json_file.with_suffix('.pkl'))
                    checkpoints.append(metadata)
            except (json.JSONDecodeError, IOError):
                continue
                
        return checkpoints
        
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to most recent checkpoint file.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
            
        # Sort by generation (descending)
        latest = max(checkpoints, key=lambda x: x['generation'])
        return latest['file']
        
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Delete old checkpoint files, keeping only the last N.
        
        Args:
            keep_last_n: Number of most recent checkpoints to keep
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            return  # Nothing to delete
            
        # Sort by generation
        sorted_checkpoints = sorted(checkpoints, key=lambda x: x['generation'], reverse=True)
        
        # Delete old checkpoints (keep_last_n onwards)
        for checkpoint in sorted_checkpoints[keep_last_n:]:
            pkl_file = Path(checkpoint['file'])
            json_file = pkl_file.with_suffix('.json')
            
            try:
                if pkl_file.exists():
                    pkl_file.unlink()
                if json_file.exists():
                    json_file.unlink()
            except OSError:
                pass  # Ignore deletion errors


__all__ = ['EvolutionCheckpoint']
