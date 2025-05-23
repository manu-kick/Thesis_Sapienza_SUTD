import os
from pathlib import Path
import warnings

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

os.environ["CUDA_VISIBLE_DEVICES"]="1"


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.refiner import get_refiner
    from src.model.model_wrapper_kd_RefGaussian_precomputation import ModelWrapper_KD_RefinementPrecomputation


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main_kd",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,'resume': "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=True,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing: Keep only the best 2 models for each metric and overwrite old ones
    callbacks.append(
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            save_top_k=2,  # Keep only the top 2 models for this metric
            monitor="train_loss",
            mode="min",  # Lower train loss is better
            auto_insert_metric_name=False,  # Prevent auto metric name insertion
            filename="best_train_loss",  # Static name ensures overwriting
        )
    )

    callbacks.append(
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            save_top_k=2,  # Keep only the top 2 models for this metric
            monitor="val/psnr_mean",
            mode="max",  # Higher PSNR is better
            auto_insert_metric_name=False,  # Prevent auto metric name insertion
            filename="best_val_psnr",  # Static name ensures overwriting
        )
    )

    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy="auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        # num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        num_sanity_val_steps=0,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    losses = get_losses(cfg.loss)

    if cfg.enable_refinement:
        refiner_kwargs = {
            "name": "refiner",
            "decoder": decoder,
            "losses": losses
        }
        refiner = get_refiner(**refiner_kwargs)
        
    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": decoder,
        "losses": losses,
        "refiner": refiner if cfg.enable_refinement else None,
        "refiner_cfg": cfg.dataset.refinement_cfg if cfg.enable_refinement else None,
        "step_tracker": step_tracker,
        }
                        
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    # Finetune or continue training
    if cfg.mode == "train" :
        model_wrapper = ModelWrapper_KD_RefinementPrecomputation.load_from_checkpoint( checkpoint_path, **model_kwargs, strict=False)
        print(cyan(f"Loaded weigths from {checkpoint_path}."))

        model_wrapper = ModelWrapper_KD_RefinementPrecomputation(**model_kwargs)
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=None)
    
   

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()
