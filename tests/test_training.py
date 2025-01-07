import pytest
import logging
from hydra import compose, initialize
from omegaconf import OmegaConf

from train import TrainingManager

def test_basic_train():
    with initialize(config_path="../config/hydra"):
        # compose config with overrides
        cfg = compose(config_name="train.yaml", overrides=[
            "device=cpu",
            "max_iters=10",
            "batch_size=2",
            "wandb_log=false"
        ])
    logging.info(f"Test config:\n{OmegaConf.to_yaml(cfg)}")

    trainer = TrainingManager(cfg)
    trainer.train()

    # If it runs without error, pass
    assert True
