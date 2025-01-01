"""Trainer for image segmentation."""

from typing import Any, Optional

import torch
import numpy as np
import PIL
import matplotlib.pyplot as plt
from torchmetrics import MetricCollection

from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryAccuracy,
    BinaryF1Score,
    BinaryCohenKappa,
    BinaryRecall,
    BinaryPrecision,
    BinaryMatthewsCorrCoef
)
import torchgeo
import torchgeo.trainers

from skimage.exposure import equalize_hist
import segmentation_models_pytorch as smp

import torch.nn as nn

import src.models
import src.models_segmentation
import src.utils

from src.losses.combine_losses import CombineLoss

src.utils.set_resources(num_threads=4)


class SegmentationTrainer(torchgeo.trainers.SemanticSegmentationTask):
    def __init__(
        self,
        segmentation_model,
        model,
        model_type="",
        weights=None,
        feature_map_indices=(5, 11, 17, 23),
        aux_loss_factor=0.5,
        input_size=224,
        patch_size=16,
        in_channels: int = 3,
        num_classes: int = 1000,
        num_filters: int = 3,
        loss: str = "ce",
        pretrained=True,
        input_res=10,
        adapter=False,
        adapter_trainable=True,
        adapter_shared=False,
        adapter_scale=1.0,
        adapter_type="lora",
        adapter_hidden_dim=16,
        norm_trainable=True,
        fixed_output_size=0,
        use_mask_token=False,
        train_patch_embed=False,
        patch_embed_adapter=False,
        patch_embed_adapter_scale=1.0,
        train_all_params=False,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: Optional[int] = None,
        lr: float = 1e-3,
        patience: int = 10,
        train_cls_mask_tokens=False,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        callbacks=None,
        only_scaler_trainable=False,
        only_bias_trainable=False,
        bands_mean=None,
        bands_std=None,
        score_threshold=None
    ) -> None:
        super().__init__()

    def configure_callbacks(self):
        return self.hparams["callbacks"]  # self.callbacks

    def configure_losses(self):
        loss = self.hparams['loss']
        ignore_index = self.hparams['ignore_index']
        if loss == 'ce':
            ignore_value = -1000 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=self.hparams['class_weights']
            )
        elif loss == 'jaccard':
            # JaccardLoss requires a list of classes to use instead of a class
            # index to ignore.
            classes = [
                i for i in range(self.hparams['num_classes']) if i != ignore_index
            ]

            self.criterion = smp.losses.JaccardLoss(mode='multiclass', classes=classes)
        elif loss == 'focal':
            self.criterion = smp.losses.FocalLoss(
                'multiclass', ignore_index=ignore_index, normalized=True
            )
        elif loss == 'combine':
            self.criterion = CombineLoss(ignore_index=ignore_index)
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def configure_models(self):
        backbone = src.models.get_model(**self.hparams)
        # add segmentation head
        if self.hparams["segmentation_model"] == "fcn":
            self.model = src.models_segmentation.ViTWithFCNHead(
                backbone,
                num_classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "upernet":
            self.model = src.models_segmentation.UPerNetWrapper(
                backbone,
                self.hparams["feature_map_indices"],
                num_classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "samseg":
            self.model = src.models_segmentation.SAMSegWrapper(
                backbone,
                self.hparams["feature_map_indices"],
                num_classes=self.hparams["num_classes"],
            )
        else:
            raise NotImplementedError(
                f"`model` must be in [fcn, upernet], not {self.hparams['model']}"
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"] + 1
        ignore_index: Optional[int] = self.hparams["ignore_index"]
        metrics = MetricCollection(
            [
                BinaryAccuracy(
                    ignore_index=ignore_index,
                ),
                BinaryJaccardIndex(
                    ignore_index=ignore_index
                ),
                BinaryF1Score(
                    ignore_index=ignore_index
                ),
                BinaryCohenKappa(
                    ignore_index=ignore_index
                ),
                BinaryRecall(
                    ignore_index=ignore_index
                ),
                BinaryPrecision(
                    ignore_index=ignore_index
                ),
                BinaryMatthewsCorrCoef(
                    ignore_index=ignore_index
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.train_aux_metrics = metrics.clone(prefix="train_aux_")
        self.val_aux_metrics = metrics.clone(prefix="val_aux_")
        self.test_aux_metrics = metrics.clone(prefix="test_aux_")

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch["image"]
        y = batch["mask"]

        if self.model.deepsup:
            y_hat, y_aux = self(x)
            # y_aux_hard = y_aux.argmax(dim=1)
            loss_aux = self.criterion(y_aux.squeeze(), y.squeeze())
            self.log("train_aux_loss", loss_aux)
            y_aux_hard = (torch.sigmoid(y_aux) > self.hparams['score_threshold']).int()
            self.train_aux_metrics(y_aux_hard.squeeze(), y.squeeze())
            self.log_dict(self.train_aux_metrics)
        else:
            y_hat = self(x)        
        # y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.log("train_loss", loss)
        y_hat_hard = (torch.sigmoid(y_hat) > self.hparams['score_threshold']).int()
        self.train_metrics(y_hat_hard.squeeze(), y.squeeze())
        self.log_dict(self.train_metrics)

        if batch_idx % 100 == 0:
            imgs = self.PIL_imgs_from_batch(x)
            target_imgs = self.PIL_masks_from_batch(y.squeeze())
            pred_imgs = self.PIL_masks_from_batch(y_hat_hard.squeeze())
            self.logger.log_image(
                key="train_imgs",
                images=imgs,
            )
            self.logger.log_image(key="train_preds", images=pred_imgs)
            self.logger.log_image(key="train_targets", images=target_imgs)

        if self.model.deepsup:
            loss = loss + self.hparams["aux_loss_factor"] * loss_aux
        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]

        if self.model.deepsup:
            y_hat, y_aux = self(x)
            # y_aux_hard = y_aux.argmax(dim=1)
            loss_aux = self.criterion(y_aux.squeeze(), y.squeeze())
            self.log("val_aux_loss", loss_aux)
            y_aux_hard = (torch.sigmoid(y_aux) > self.hparams['score_threshold']).int()
            self.val_aux_metrics(y_aux_hard.squeeze(), y.squeeze())
            self.log_dict(self.val_aux_metrics)
        else:
            y_hat = self(x)
        # y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.log("val_loss", loss)
        y_hat_hard = (torch.sigmoid(y_hat) > self.hparams['score_threshold']).int()
        self.val_metrics(y_hat_hard.squeeze(), y.squeeze())
        self.log_dict(self.val_metrics)

        if batch_idx % 100 == 0:
            imgs = self.PIL_imgs_from_batch(x)
            target_imgs = self.PIL_masks_from_batch(y.squeeze())
            pred_imgs = self.PIL_masks_from_batch(y_hat_hard.squeeze())
            self.logger.log_image(
                key="val_imgs",
                images=imgs,
            )
            self.logger.log_image(key="val_preds", images=pred_imgs)
            self.logger.log_image(key="val_targets", images=target_imgs)

        # log some figures
        if False:
            #  (
            #  batch_idx < 10
            #  and hasattr(self.trainer, "datamodule")
            #  and self.logger
            #  and hasattr(self.logger, "experiment")
            #  and hasattr(self.logger.experiment, "add_figure")
            # ):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat_hard
                for key in ["image", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = torchgeo.datasets.utils.unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    summary_writer.add_figure(
                        f"image/{batch_idx}", fig, global_step=self.global_step
                    )
                    plt.close()
            except ValueError:
                pass

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        if self.model.deepsup:
            y_hat, y_aux = self(x)
            # y_aux_hard = y_aux.argmax(dim=1)
            loss_aux = self.criterion(y_aux.squeeze(), y.squeeze())
            self.log("test_aux_loss", loss_aux)
            y_aux_hard=(torch.sigmoid(y_aux) > self.hparams['score_threshold']).int()
            self.test_aux_metrics(y_aux_hard.squeeze(), y.squeeze())
            self.log_dict(self.test_aux_metrics)
        else:
            y_hat = self(x)
        # y_hat_hard = y_hat.argmax(dim=1)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        self.log("test_loss", loss)
        y_hat_hard=(torch.sigmoid(y_hat) > self.hparams['score_threshold']).int()
        self.test_metrics(y_hat_hard.squeeze(), y.squeeze())
        self.log_dict(self.test_metrics)

        if False:  # batch_idx % 100 == 0:
            imgs = self.PIL_imgs_from_batch(x)
            target_imgs = self.PIL_masks_from_batch(y.squeeze())
            pred_imgs = self.PIL_masks_from_batch(y_hat_hard.squeeze())
            self.logger.log_image(
                key="test_imgs",
                images=imgs,
            )
            self.logger.log_image(key="test_preds", images=pred_imgs)
            self.logger.log_image(key="test_targets", images=target_imgs)

        if self.model.deepsup:
            loss = loss + self.hparams["aux_loss_factor"] * loss_aux
        return loss

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        if self.model.deepsup:
            y_hat, _ = self(x)
            # y_hat = y_hat.softmax(dim=q)
            y_hat = (torch.sigmoid(y_hat) > self.hparams['score_threshold']).int()
        else:
            # y_hat: torch.Tensor = self(x).softmax(dim=1)
            y_hat: torch.Tensor = (torch.sigmoid(self(x)) > self.hparams['score_threshold']).int()
        return y_hat

    def PIL_imgs_from_batch(self, x, n=4):
        """return list of PIL images from tensor input images"""
        imgs = []
        for img in x[:n]:
            img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
            img = (img * np.array(self.hparams['bands_std']).reshape(1, 1, -1) + np.array(self.hparams['bands_mean']).reshape(1, 1, -1))

            # assert img.shape[-1] == 3
            if img.shape[-1] not in [3, 1]:
                img = img[:, :, [3, 2, 1]]  # S2 RGB
            # img = img.detach().cpu().numpy()
            # img /= img.max(axis=(0, 1))
            img = equalize_hist(img.transpose(2, 0, 1)).transpose(1, 2, 0)
            img *= 255
            img = np.clip(img, 0, 255).astype(np.uint8)
            imgs.append(PIL.Image.fromarray(img))

        return imgs

    def PIL_masks_from_batch(self, x, n=4):
        """return list of PIL images from tensor input images"""
        palette = np.array([[255, 255, 255], [  0,   0, 128], [  128,   128, 128]])

        imgs = []
        for img in x[:n]:
            # img = np.moveaxis(img.detach().cpu().numpy(), 0, -1)
            assert len(img.shape) == 2 or img.shape[-1] == 1, f"{img.shape=}"
            img[img==255] = 2
            img = palette[img.detach().cpu().numpy()]
            img = img.astype(np.uint8) #* (255 // self.hparams["num_classes"])
            imgs.append(PIL.Image.fromarray(img)) #, mode="P"

        return imgs
