import wandb
import numpy as np
import torch
import psutil
from typing import Dict, Any
from datetime import datetime

class WandBLogger:
    def __init__(self, config: Dict[str, Any]):
        """Initialize WandB logger with config"""
        self.run = wandb.init(
            project=config.get('wandb_project', 'VSLM'),
            entity=config.get('wandb_entity', 'wcml'),
            name=config.get('wandb_run_name'),
            group=config.get('wandb_group'),
            config=config,
            tags=[
                config.get('model_architecture', 'original'),
                f"dims_{config['layer_dims'][-1]}",
                f"layers_{config['n_layer']}",
                config.get('shape_variant', 'none')
            ]
        )
        
        # Store device information
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.start_time = datetime.now()
        
        # Log initial system info
        self.log_system_info()
        
        # Log model architecture diagram
        self.log_model_architecture(config)
        
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
        
        wandb.log(system_info)
        
    def log_model_architecture(self, config: Dict[str, Any]):
        """Create and log model architecture visualization"""
        layer_dims = config['layer_dims']
        n_layers = len(layer_dims)
        
        # Create layer dimension plot
        layer_plot_data = [[i, dim] for i, dim in enumerate(layer_dims)]
        table = wandb.Table(data=layer_plot_data, columns=["layer", "dimension"])
        self.run.log({
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
        # Combine all metrics
        log_dict = {
            "iter": iter_num,
            "train/loss": metrics.get('loss'),
            "learning_rate": metrics.get('lr'),
            "mfu": metrics.get('mfu', 0) * 100,  # Model flops utilization
            "tokens": metrics.get('tokens', 0),
            "compute": metrics.get('compute', 0),
            "time_elapsed": metrics.get('time_elapsed', 0),
            "tokens_per_second": metrics.get('tokens_per_second', 0),
            "compute_per_second": metrics.get('compute_per_second', 0),
        }
        
        # Add GPU stats if available
        log_dict.update(self.log_gpu_stats())
        
        # Add system stats every 10 iterations to avoid overhead
        if iter_num % 10 == 0:
            log_dict.update(self.log_system_stats())
        
        wandb.log(log_dict)

    def log_evaluation(self, iter_num: int, train_loss: float, val_loss: float, compute: float):
        """Log evaluation metrics"""
        eval_dict = {
            "iter": iter_num,
            "eval/train_loss": train_loss,
            "eval/val_loss": val_loss,
            "eval/train_perplexity": np.exp(train_loss),
            "eval/val_perplexity": np.exp(val_loss),
            "eval/cumulative_compute": compute,  # Add this line
            # This creates a separate series specifically for the compute vs test loss plot
            "compute_test/compute": compute,     # Add this line
            "compute_test/loss": val_loss        # Add this line
        }
        
        # Add GPU stats for evaluation as well
        eval_dict.update(self.log_gpu_stats())
        eval_dict.update(self.log_system_stats())
        
        wandb.log(eval_dict)

    def log_gradient_flow(self, named_parameters):
        """Log gradient flow information"""
        gradients = []
        layers = []
        
        for n, p in named_parameters:
            if p.requires_grad and p.grad is not None:
                gradients.append(p.grad.abs().mean().item())
                layers.append(n)
        
        if gradients:
            wandb.log({
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
            wandb.log({
                "memory/summary": wandb.Html(
                    torch.cuda.memory_summary(device=self.device, abbreviated=False)
                )
            })

    def finish(self):
        """Close the wandb run and log final statistics"""
        # Log final system stats
        final_stats = self.log_system_stats()
        final_stats.update(self.log_gpu_stats())
        wandb.log(final_stats)
        
        # Log memory summary at the end
        self.log_memory_summary()
        
        # Close the run
        wandb.finish()