import wandb
import numpy as np
import torch
import psutil
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ScalingExperimentConfig:
    """Configuration for a scaling law experiment"""
    # Model params
    n_layer: int
    n_heads: list
    layer_dims: list
    max_tokens: int
    
    # Training params
    batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_tokens: int
    final_tokens: int
    
    # Dataset params
    datasets: Dict[str, float]  # dataset name -> sampling weight
    eval_datasets: list
    
    # Compute params
    device: str
    dtype: str
    
    def asdict(self):
        return {k: v for k, v in self.__dict__.items()}

class WandBLogger:
    """Enhanced WandB logger supporting both general training and scaling law experiments"""
    
    def __init__(self, 
                 config: Any,
                 project: str = "VSLM",
                 entity: str = "wcml",
                 name: Optional[str] = None,
                 group: Optional[str] = None):
        
        self.config = config
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.start_time = datetime.now()
        
        # Initialize run with richer config
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            config=config if not isinstance(config, ScalingExperimentConfig) else config.asdict(),
            tags=self._get_tags()
        )
        
        # Initialize scaling metrics if using ScalingExperimentConfig
        if isinstance(config, ScalingExperimentConfig):
            self.train_tokens = 0
            self.best_train_loss = float('inf')
            self.best_val_loss = float('inf')
        
        # Log initial system info
        self.log_system_info()
        
        if isinstance(config, ScalingExperimentConfig):
            self.log_model_architecture()

    def _get_tags(self):
        """Get appropriate tags based on config type"""
        if isinstance(self.config, ScalingExperimentConfig):
            return [
                f"n_params_{self._get_param_count()}",
                f"n_layers_{self.config.n_layer}",
                f"datasets_{'-'.join(self.config.datasets.keys())}"
            ]
        else:
            return [
                self.config.get('model_architecture', 'original'),
                f"dims_{self.config.layer_dims[-1]}",
                f"layers_{self.config.n_layer}",
                self.config.get('shape_variant', 'none')
            ]

    def _get_param_count(self) -> int:
        """Calculate non-embedding parameter count"""
        if isinstance(self.config, ScalingExperimentConfig):
            return sum(12 * dim * dim for dim in self.config.layer_dims)
        return None

    def log_metrics(self, metrics: Dict[str, Any]):
        """Generic method to log any metrics through wandb"""
        wandb.log(metrics)

    def log_memory_per_gpu(self):
        """Log memory statistics for each GPU"""
        if torch.cuda.is_available():
            metrics = {}
            for i in range(torch.cuda.device_count()):
                metrics.update({
                    f'memory/gpu{i}_allocated': torch.cuda.memory_allocated(i) / 1e9,
                    f'memory/gpu{i}_reserved': torch.cuda.memory_reserved(i) / 1e9,
                })
            self.log_metrics(metrics)

    def log_basic_metrics(self, metrics: Dict[str, Any]):
        """Log basic training metrics including CPU and RAM usage"""
        self.log_metrics({
            'memory/cpu_percent': psutil.cpu_percent(),
            'memory/ram_percent': psutil.virtual_memory().percent,
            **metrics
        })

    def log_scaling_metrics(self, tokens_so_far: int, comp: float, time_elapsed: float):
        """Log metrics specific to scaling experiments"""
        self.log_metrics({
            'tokens': tokens_so_far,
            'compute': comp,
            'time_elapsed': time_elapsed,
            'tokens_per_second': tokens_so_far / time_elapsed if time_elapsed > 0 else 0,
            'compute_per_second': comp / time_elapsed if time_elapsed > 0 else 0,
        })

    def log_system_info(self):
        """Log system information at the start of training"""
        system_info = {
            "system/cpu_count": psutil.cpu_count(),
            "system/total_ram_gb": psutil.virtual_memory().total / 1e9
        }
        
        if torch.cuda.is_available():
            system_info.update({
                f"gpu/device_name": torch.cuda.get_device_name(self.device),
                f"gpu/total_memory_gb": torch.cuda.get_device_properties(self.device).total_memory / 1e9
            })
        
        self.log_metrics(system_info)

    def log_model_architecture(self):
        """Create and log model architecture visualization"""
        if isinstance(self.config, ScalingExperimentConfig):
            # Create layer dimension plot
            layer_plot_data = [[i, dim] for i, dim in enumerate(self.config.layer_dims)]
            table = wandb.Table(data=layer_plot_data, columns=["layer", "dimension"])
            self.log_metrics({
                "layer_dimensions": wandb.plot.line(
                    table, "layer", "dimension",
                    title="Layer Dimensions Architecture"
                )
            })

    def log_gpu_stats(self):
        """Log GPU statistics"""
        if torch.cuda.is_available():
            gpu_stats = {
                f"gpu{self.device}/memory_allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
                f"gpu{self.device}/memory_reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
                f"gpu{self.device}/utilization": torch.cuda.utilization(self.device)
            }
            return gpu_stats
        return {}

    def log_system_stats(self):
        """Log system statistics"""
        stats = {
            "system/cpu_percent": psutil.cpu_percent(),
            "system/ram_percent": psutil.virtual_memory().percent,
            "system/ram_used_gb": psutil.virtual_memory().used / 1e9,
            "system/run_time_hours": (datetime.now() - self.start_time).total_seconds() / 3600
        }
        return stats

    def log_training_step(self, iter_num: int, metrics: Dict[str, float]):
        """Log training metrics for each step"""
        # Basic metrics
        log_dict = {
            "iter": iter_num,
            "train/loss": metrics.get('loss'),
            "learning_rate": metrics.get('lr'),
            "mfu": metrics.get('mfu', 0) * 100,  # Model flops utilization
        }
        
        # Add scaling-specific metrics if using ScalingExperimentConfig
        if isinstance(self.config, ScalingExperimentConfig):
            self.train_tokens += metrics.get('batch_tokens', 0)
            log_dict.update({
                "train/tokens": self.train_tokens,
                "train/tokens_per_second": metrics.get('tokens_per_second', 0),
                "train/compute_per_second": metrics.get('compute_per_second', 0),
                "train/grad_norm": metrics.get('grad_norm')
            })
        
        # Add GPU stats if available
        log_dict.update(self.log_gpu_stats())
        
        # Add system stats every 10 iterations
        if iter_num % 10 == 0:
            log_dict.update(self.log_system_stats())
        
        self.log_metrics(log_dict)

    def log_evaluation(self, iter_num: int, train_loss: float, val_loss: float, compute: float):
        """Log evaluation metrics"""
        eval_dict = {
            "iter": iter_num,
            "eval/train_loss": train_loss,
            "eval/val_loss": val_loss,
            "eval/train_perplexity": np.exp(train_loss),
            "eval/val_perplexity": np.exp(val_loss),
            "eval/cumulative_compute": compute,
            "compute_test/compute": compute,
            "compute_test/loss": val_loss
        }
        
        # Update best losses if using scaling config
        if isinstance(self.config, ScalingExperimentConfig):
            self.best_train_loss = min(self.best_train_loss, train_loss)
            self.best_val_loss = min(self.best_val_loss, val_loss)
            eval_dict.update({
                "eval/best_train_loss": self.best_train_loss,
                "eval/best_val_loss": self.best_val_loss
            })
            
        # Add GPU and system stats
        eval_dict.update(self.log_gpu_stats())
        eval_dict.update(self.log_system_stats())
        
        self.log_metrics(eval_dict)

    def log_transfer_results(self, dataset: str, loss: float, perplexity: float):
        """Log transfer learning results on other datasets"""
        self.log_metrics({
            f"transfer/{dataset}/loss": loss,
            f"transfer/{dataset}/perplexity": perplexity
        })

    def log_memorization_stats(self, unique_token_ratio: float, repeat_token_ratio: float):
        """Log dataset memorization statistics"""
        self.log_metrics({
            "memorization/unique_tokens": unique_token_ratio,
            "memorization/repeat_tokens": repeat_token_ratio
        })

    def log_gradient_flow(self, named_parameters):
        """Log gradient flow information"""
        gradients = []
        layers = []
        
        for n, p in named_parameters:
            if p.requires_grad and p.grad is not None:
                gradients.append(p.grad.abs().mean().item())
                layers.append(n)
        
        if gradients:
            self.log_metrics({
                "gradients": wandb.plot.bar(
                    wandb.Table(data=[[l, g] for l, g in zip(layers, gradients)],
                              columns=["layer", "gradient"]),
                    "layer",
                    "gradient",
                    title="Gradient Flow"
                )
            })

    def log_memory_summary(self):
        """Log detailed memory usage summary"""
        if torch.cuda.is_available():
            self.log_metrics({
                "memory/summary": wandb.Html(
                    torch.cuda.memory_summary(device=self.device, abbreviated=False)
                )
            })

    def finish(self):
        """Close the wandb run and log final statistics"""
        # Log final system stats
        final_stats = self.log_system_stats()
        final_stats.update(self.log_gpu_stats())
        
        # Add scaling-specific final stats
        if isinstance(self.config, ScalingExperimentConfig):
            final_stats.update({
                "final/train_loss": self.best_train_loss,
                "final/val_loss": self.best_val_loss,
                "final/total_tokens": self.train_tokens,
                "final/total_params": self._get_param_count()
            })
        
        self.log_metrics(final_stats)
        
        # Log memory summary at the end
        self.log_memory_summary()
        
        # Close the run
        wandb.finish()
