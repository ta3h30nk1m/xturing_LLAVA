import datetime
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam
from pytorch_lightning import callbacks
from pytorch_lightning.trainer.trainer import Trainer

from xturing.config import DEFAULT_DEVICE, IS_INTERACTIVE
from xturing.datasets.base import BaseDataset
from xturing.engines.base import BaseEngine
from xturing.preprocessors.base import BasePreprocessor


GLOBAL_BATCHES = 32    # (will be Accumulate_grad_batches in argument of pytorch lightning trainer)
# todo : this is hard coded, fix later
# Global batch(accumulate grad batches) LLaVA settings
#       first stage : bs = 128      (train mm_projector) (freeze CLIP, LLM, LLMLoRA)  
#       second stage : bs = 32      (train mm_projector, LLMLoRA)  (freeze CLIP, LLM) 


class TuringLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_engine: BaseEngine,
        train_dataset: BaseDataset,
        preprocessor: Optional[BasePreprocessor] = None,
        batch_size: int = 2,
        learning_rate: float = 5e-5,
        optimizer_name: str = "adamw",
        saved_path: str = "saved_model",
    ):
        super().__init__()
        self.model_engine = model_engine
        self.pytorch_model = self.model_engine.model#.mm_projector
        self.train_dataset = train_dataset
        self.preprocessor = preprocessor
        
        print("\nEntered trainers/TuringLightningModule")
        print(f"called preprocessor = {self.preprocessor}\n")
        
        # Hyperparameters
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.optimizer_name = optimizer_name
        self.saved_path = saved_path
        
        self.losses = []
        
        print("\ntrainers/TuringLightningModule __init__ finished")
        print(f"\tbatchsize = {self.batch_size}")
        print(f"\tsaved_path = {self.saved_path}\n")

    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.pytorch_model.parameters(), lr=self.learning_rate
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.adam(
                self.pytorch_model.parameters(), lr=self.learning_rate
            )
        elif self.optimizer_name == "cpu_adam":
            optimizer = DeepSpeedCPUAdam(
                self.pytorch_model.parameters(), lr=self.learning_rate
            )
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        self.train_dl = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.preprocessor,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            batch_size=self.batch_size,
        )

        return self.train_dl

    def training_step(self, batch, batch_idx):
        loss = self.model_engine.training_step(batch)
        self.losses.append(loss.item())
        self.log("loss", loss.item(), prog_bar=True)
        
        if batch_idx % 100 == 0 :
            print(f"Runtime log : batch_idx = {batch_idx}, loss = {loss.item()}")

        return loss

    def validation_step(self, batch, batch_idx):
        return self.model_engine.validation_step(batch)

    def on_save_checkpoint(self, checkpoint):
        self.model_engine.save(self.saved_path)


class LightningTrainer:
    config_name: str = "lightning_trainer"

    def __init__(
        self,
        model_engine: BaseEngine,
        train_dataset: BaseDataset,
        preprocessor: BasePreprocessor,
        max_epochs: int = 3,
        batch_size: int = 2,
        learning_rate: float = 1e-3,
        optimizer_name: str = "adamw",
        use_lora: bool = False,
        use_deepspeed: bool = False,
        max_training_time_in_secs: Optional[int] = None,
        lora_type: int = 16,
        output_dir: str = ""
    ):
        self.lightning_model = TuringLightningModule(
            model_engine=model_engine,
            train_dataset=train_dataset,
            preprocessor=preprocessor,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name,
            saved_path=output_dir
        )

        # checkpoints_dir_path = Path("saved_model")

        # if not checkpoints_dir_path.exists():
        #     checkpoints_dir_path.mkdir(exist_ok=True, parents=True)

        training_callbacks = []

        if len(train_dataset) > 100:
            training_callbacks.append(callbacks.LearningRateFinder())

        # if not IS_INTERACTIVE:
        #     training_callbacks.append(callbacks.BatchSizeFinder())

        if max_training_time_in_secs is not None:
            training_callbacks.append(
                callbacks.Timer(
                    duration=datetime.timedelta(seconds=max_training_time_in_secs)
                )
            )
        model_engine.model.train()

        try:
            model_engine.model.print_trainable_parameters()
        except AttributeError:
            pass

        if DEFAULT_DEVICE.type == "cpu":   ############ not enter
            print(f"\nPytorch Lightning Trainer Instance Generated  (You use decive cpu, r u serious?)")
            print(f"Runtime log :  accumulate_grad_batches = {GLOBAL_BATCHES},   enable_checkpointing = False,   max_epochs = {max_epochs}\n")  ## todo : fix enable_checkpointing log, it is hard coded 
            self.trainer = Trainer(
                num_nodes=1,
                accelerator="cpu",
                max_epochs=max_epochs,
                callbacks=training_callbacks,
                enable_checkpointing=False,
                log_every_n_steps=50,
                accumulate_grad_batches=GLOBAL_BATCHES
            )
        elif not use_lora and not use_deepspeed: ######## first stage enter (?)
            print(f"\nPytorch Lightning Trainer Instance Generated  (Should be First stage training)")
            print(f"Runtime log :  accumulate_grad_batches = {GLOBAL_BATCHES},   enable_checkpointing = True,   max_epochs = {max_epochs}\n")   ## todo : fix enable_checkpointing log, it is hard coded 
            self.trainer = Trainer(
                num_nodes=1,
                accelerator="gpu",
                max_epochs=max_epochs,
                callbacks=training_callbacks,
                enable_checkpointing=True,
                log_every_n_steps=50,
                accumulate_grad_batches=GLOBAL_BATCHES
            )
        else:
            print(f"\nPytorch Lightning Trainer Instance Generated  (Should be Second stage training)")
            print(f"Runtime log :  accumulate_grad_batches = {GLOBAL_BATCHES},   enable_checkpointing = True,   max_epochs = {max_epochs}\n") ## todo : fix enable_checkpointing log, it is hard coded 
            training_callbacks = [
                callbacks.ModelCheckpoint(
                    #dirpath=str(checkpoints_dir_path), save_on_train_epoch_end=True
                    dirpath=str(output_dir), save_on_train_epoch_end=True
                ),
            ]

            strategy = "auto"
            if not IS_INTERACTIVE:
                strategy = (
                    "deepspeed_stage_2_offload"
                    if optimizer_name == "cpu_adam"
                    else "deepspeed_stage_2"
                )

            self.trainer = Trainer(
                num_nodes=1,
                accelerator="gpu",
                strategy=strategy,
                precision=lora_type,
                max_epochs=max_epochs,
                callbacks=training_callbacks,
                enable_checkpointing=True,
                log_every_n_steps=50,
                accumulate_grad_batches=GLOBAL_BATCHES
            )

    def fit(self):
        self.trainer.fit(self.lightning_model)
        if self.trainer.checkpoint_callback is not None:
            self.trainer.checkpoint_callback.best_model_path

    def engine(self):
        return self.lightning_model.model_engine
