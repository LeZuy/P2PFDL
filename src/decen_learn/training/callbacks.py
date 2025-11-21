# src/decen_learn/training/callbacks.py
class TrainingCallback:
    """Base class for training callbacks."""
    
    def on_train_start(self) -> None:
        pass
    
    def on_epoch_end(self, metrics: EpochMetrics) -> None:
        pass
    
    def on_train_end(self) -> None:
        pass


class MetricsLogger(TrainingCallback):
    """Logs metrics to files."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(self, metrics: EpochMetrics) -> None:
        np.savetxt(
            self.output_dir / f"loss_epoch_{metrics.epoch + 1}.txt",
            metrics.losses,
            fmt="%.4f"
        )


class ProjectionSnapshot(TrainingCallback):
    """Saves projected weights for visualization."""
    
    def __init__(self, output_dir: Path, interval: int = 20):
        self.output_dir = output_dir
        self.interval = interval
    
    def on_epoch_end(self, metrics: EpochMetrics) -> None:
        if metrics.epoch % self.interval == 0:
            # Save projection snapshots
            pass