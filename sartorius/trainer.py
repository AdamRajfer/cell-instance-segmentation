import os
import pickle

import yaml
from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import segmentation_models_pytorch as smp
import torch
import wandb
from torch import nn, optim
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils import data
from torchvision.transforms import Resize
from tqdm.notebook import tqdm

from sartorius.dataset import Dataset, SubmissionDataset
from sartorius.encoder import Encoder
from sartorius.losses import (
    binary_crossentropy,
    dice_loss,
    focal_loss,
    jaccard_binary_crossentropy_loss,
    jaccard_loss,
    tversky_loss,
)
from sartorius.metrics import iou_map


class Trainer:
    def __init__(self, config: Dict[str, Any], encoder: Encoder) -> None:
        self._config = config
        self._encoder = encoder

    def cross_validate(self, folds: List[Tuple[Dataset, Dataset]], dataset: Dataset) -> Dict[str, Any]:
        if self._config["from_directory"]:
            outputs_file = os.path.join(self._config["experiments_dir"], self._config["group"], "valid_outputs.pickle")
            with open(outputs_file, "rb") as file:
                return pickle.load(file)
        valid_outputs_list = [
            self.train(f"fold-{i}", train_dataset, valid_dataset)
            for i, (train_dataset, valid_dataset) in enumerate(folds, start=1)
        ]
        return self._aggregate_outputs(valid_outputs_list, dataset, "valid")

    def train(
        self, fold_name: str, train_dataset: Dataset, valid_dataset: Optional[Dataset] = None
    ) -> Optional[Dict[str, Any]]:
        print(f"Training: {fold_name}")
        train_dataloader = data.DataLoader(
            train_dataset,
            batch_size=self._config["train_batch_size"],
            shuffle=True,
            num_workers=self._config["num_workers"],
            pin_memory=True,
            drop_last=True,
        )
        valid_dataloader = data.DataLoader(
            valid_dataset,
            batch_size=self._config["test_batch_size"],
            num_workers=self._config["num_workers"],
            pin_memory=True,
        ) if valid_dataset is not None else None
        model = self._build_model()
        optimizer, run = self._prepare_for_training(model, fold_name)
        patience = 0
        best_valid_outputs: Dict[str, Any] = {}
        checkpoints_path = self._update_directories(fold_name)
        for epoch in range(1, self._config["epochs"] + 1):
            print(f"Epoch: {epoch}/{self._config['epochs']}")
            model.train()
            train_outputs = self._epoch(
                train_dataloader, model, mode="train", desc="train", train=True, optimizer=optimizer
            )
            wandb.log({name: value.mean() for name, value in train_outputs["metrics"].items()})
            if valid_dataloader is not None:
                with torch.no_grad():
                    model.eval()
                    valid_outputs = self._epoch(valid_dataloader, model, mode="valid", desc="valid", train=False)
                    wandb.log({name: value.mean() for name, value in valid_outputs["metrics"].items()})
                    if self._is_improvement(valid_outputs, best_valid_outputs):
                        patience = 0
                        best_valid_outputs = valid_outputs
                        torch.save(model.state_dict(), checkpoints_path)
                        for name, value in chain(train_outputs["metrics"].items(), valid_outputs["metrics"].items()):
                            mode, metric_name = name.split('/')
                            run.summary[f"{mode}/best_{metric_name}"] = value.mean()
                        run.summary["best_epoch"] = epoch
                    else:
                        patience += 1
                    if patience == self._config["patience"]:
                        break
        run.finish()
        if valid_dataloader is not None:
            return self._convert_to_dataset(best_valid_outputs)
        torch.save(model.state_dict(), checkpoints_path)

    def evaluate(self, dataset: Union[Dataset, SubmissionDataset], mode: str) -> Dict[str, Any]:
        if self._config["from_directory"]:
            outputs_file = os.path.join(
                self._config["experiments_dir"], self._config["group"], f"{mode}_outputs.pickle"
            )
            with open(outputs_file, "rb") as file:
                return pickle.load(file)
        print(f"Model evaluation: {mode}")
        dataloader = data.DataLoader(
            dataset,
            batch_size=self._config["test_batch_size"],
            num_workers=self._config["num_workers"],
            pin_memory=True,
        )
        outputs_list: List[Dict[str, Any]] = []
        models_dir = os.path.join(self._config["experiments_dir"], self._config["group"], "checkpoints/")
        model = self._build_model()
        for path in os.listdir(models_dir):
            model.load_state_dict(torch.load(os.path.join(models_dir, path)))
            model.eval()
            with torch.no_grad():
                outputs_list.append(self._convert_to_dataset(
                    self._epoch(dataloader, model, mode=mode, desc=path.split(".pt")[0], train=False)
                ))
        return self._aggregate_outputs(outputs_list, dataset, mode)

    def _epoch(
        self,
        dataloader: data.DataLoader,
        model: nn.Module,
        mode: str,
        desc: str,
        train: bool,
        optimizer: Optional[Optimizer] = None,
    ) -> Dict[str, Any]:
        outputs: Dict[str, Any] = {
            "masks": [], "masks_true": [], "labels": [], "metrics": defaultdict(list), "image_ids": []
        }
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc)
        for step, batch in progress_bar:
            final_step = step == len(dataloader) - 1
            self._step(outputs, batch, model, mode, train, optimizer, final_step)
            if final_step:
                progress_bar.set_postfix(**{
                    name: f"{torch.concat(value).mean() if isinstance(value, List) else value:0.4f}"
                    for name, value in outputs["metrics"].items()
                })
        torch.cuda.empty_cache()
        outputs["masks"] = torch.concat(outputs["masks"])
        outputs["labels"] = torch.concat(outputs["labels"]) if outputs["labels"] else None
        outputs["metrics"] = {
            name: torch.concat(value) if isinstance(value, List) else value
            for name, value in outputs["metrics"].items()
        }
        outputs.pop("masks_true")
        if outputs["labels"] is None:
            outputs.pop("labels")
        if not outputs["metrics"]:
            outputs.pop("metrics")
        return outputs

    def _step(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        model: nn.Module,
        mode: str,
        train: bool,
        optimizer: Optional[Optimizer],
        final_step: bool,
    ) -> None:
        mask_true = batch["mask"].float().cuda() if "mask" in batch else None
        class_true = batch["label"].long().cuda() if "label" in batch else None
        image = batch["image"].float().cuda() / 255
        image_shape = image.shape[2:]
        image = Resize((self._config["height"], self._config["width"]))(image)
        mask_pred = model(image)
        class_pred: Optional[torch.Tensor] = None
        if not isinstance(mask_pred, torch.Tensor):
            mask_pred, class_pred = mask_pred
        mask_pred = Resize(image_shape)(torch.squeeze(mask_pred, 1))
        if self._config["multiclass"]:
            mask_pred = self._multiclass_to_binary_mask(mask_pred, class_pred, class_true, image_shape, train)
        mask_pred = mask_pred.sigmoid()
        metrics = self._calculate_metrics(mask_pred, mask_true, class_pred, class_true, mode)
        if train:
            self._update_weights(metrics, optimizer)
        outputs["masks"].append(mask_pred.detach().cpu())
        if class_pred is not None:
            outputs["labels"].append(class_pred.softmax(1).detach().cpu())
        for name, value in metrics.items():
            outputs["metrics"][name].append(value.detach().cpu())
        if mask_true is not None:
            outputs["masks_true"].append(mask_true.detach().cpu())
            if final_step:
                outputs["metrics"][f"{mode}/iou_map_score"] = iou_map(
                    torch.concat(outputs["masks_true"]), torch.concat(outputs["masks"])
                )
        outputs["image_ids"] += batch["image_id"]

    def _multiclass_to_binary_mask(
        self,
        mask_pred: torch.Tensor,
        class_pred: torch.Tensor,
        class_true: torch.Tensor,
        image_shape: torch.Size,
        train: bool,
    ) -> torch.Tensor:
        class_id = class_true if train else class_pred.argmax(1)
        class_id = F.one_hot(class_id, num_classes=len(self._encoder)).float().unsqueeze(-1).unsqueeze(-1)
        return (torch.tile(class_id, dims=image_shape) * mask_pred).sum(axis=1)

    def _calculate_metrics(
        self,
        mask_pred: torch.Tensor,
        mask_true: Optional[torch.Tensor],
        class_pred: Optional[torch.Tensor],
        class_true: Optional[torch.Tensor],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        metrics: Dict[str, torch.Tensor] = {}
        if mask_true is not None:
            metrics[f"{mode}/mask_loss"] = self._mask_loss(mask_pred, mask_true)
            metrics[f"{mode}/mask_jaccard"] = 1 - jaccard_loss((mask_pred > 0.5).float(), mask_true)
            metrics[f"{mode}/mask_dice"] = 1 - dice_loss((mask_pred > 0.5).float(), mask_true)
        if class_true is not None and class_pred is not None:
            metrics[f"{mode}/class_loss"] = F.cross_entropy(class_pred, class_true, reduction="none")
            metrics[f"{mode}/class_accuracy"] = (class_pred.argmax(1) == class_true).float()
        return metrics

    def _update_weights(self, metrics: Dict[str, torch.Tensor], optimizer: Optimizer) -> None:
        loss = metrics["train/mask_loss"]
        if self._config["classify"]:
            loss = loss + metrics["train/class_loss"] * self._config["classification_alpha"]
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _is_improvement(self, valid_outputs: Dict[str, Any], best_outputs: Dict[str, Any]) -> bool:
        if not best_outputs:
            return True
        current_iou_map_score = valid_outputs["metrics"]["valid/iou_map_score"]
        best_iou_map_score = best_outputs["metrics"]["valid/iou_map_score"]
        return current_iou_map_score > best_iou_map_score

    def _build_model(self) -> nn.Module:
        if self._config["model"] == "unet":
            model = smp.Unet
        elif self._config["model"] == "unetpp":
            model = smp.UnetPlusPlus
        elif self._config["model"] == "fpn":
            model = smp.FPN
        elif self._config["model"] == "linknet":
            model = smp.Linknet
        elif self._config["model"] == "pan":
            model = smp.PAN
        elif self._config["model"] == "pspnet":
            model = smp.PSPNet
        elif self._config["model"] == "deeplab3":
            model = smp.DeepLabV3
        elif self._config["model"] == "deeplab3p":
            model = smp.DeepLabV3Plus
        else:
            raise ValueError(f"Unknown model name: {self._config['model']}")
        return model(
            classes=len(self._encoder) if self._config["multiclass"] else 1,
            aux_params=(
                {"classes": len(self._encoder), "dropout": self._config["classification_dropout"]}
                if self._config["classify"] or self._config["multiclass"] else None
            )
        ).cuda()

    def _prepare_for_training(self, model: nn.Module, name: str) -> Tuple[Optimizer, wandb.sdk.wandb_run.Run]:
        run = wandb.init(
            config=self._config,
            project=self._config["project"],
            group=self._config["group"],
            name=name,
            anonymous=self._config["anonymous"],
        )
        optimizer = optim.Adam(model.parameters(), lr=self._config["lr"])
        wandb.watch(model, log_freq=self._config["log_freq"])
        return optimizer, run

    def _aggregate_outputs(
        self, outputs_list: List[Dict[str, Any]], dataset: Union[Dataset, SubmissionDataset], mode: str
    ) -> Dict[str, Any]:
        outputs: Dict[str, Dict[str, List[Any]]] = {}
        for x in outputs_list:
            for k, v in x.items():
                if k not in outputs:
                    outputs[k] = defaultdict(list)
                outputs[k]["masks"].append(v["mask"][None])
                if "label" in v:
                    outputs[k]["labels"].append(v["label"][None])
        outputs_aggregated: Dict[str, Any] = {}
        for k, v in outputs.items():
            outputs_aggregated[k] = {"mask": torch.mean(torch.concat(v["masks"]), 0)}
            if "labels" in v:
                outputs_aggregated[k]["label"] = torch.mean(torch.concat(v["labels"]), 0)
        iou_map_score: Optional[float] = None
        if isinstance(dataset, Dataset):
            image_ids = sorted(outputs_aggregated)
            mask_pred = torch.concat([outputs_aggregated[image_id]["mask"][None] for image_id in image_ids])
            mask_true = torch.concat([dataset[image_id]["mask"][None] for image_id in image_ids])
            if "label" in list(outputs_aggregated.values())[0]:
                class_pred = torch.concat([outputs_aggregated[image_id]["label"][None] for image_id in image_ids])
            else:
                class_pred = None
            if "label" in dataset[0]:
                class_true = torch.concat([dataset[image_id]["label"][None] for image_id in image_ids])
            else:
                class_true = None
            metrics = self._calculate_metrics(mask_pred, mask_true, class_pred, class_true, mode)
            iou_map_score = iou_map(mask_true, mask_pred).item()
            for i, image_id in enumerate(image_ids):
                outputs_aggregated[image_id]["metrics"] = {}
                for name, value in metrics.items():
                    outputs_aggregated[image_id]["metrics"][name] = value[i].item()
        for k, v in outputs_aggregated.items():
            for name, value in v.items():
                if name != "metrics":
                    outputs_aggregated[k][name] = value.numpy()
        final_outputs = {"outputs": outputs_aggregated}
        if iou_map_score is not None:
            final_outputs["iou_map_score"] = iou_map_score
        outputs_file = os.path.join(self._config["experiments_dir"], self._config["group"], f"{mode}_outputs.pickle")
        with open(outputs_file, "wb") as file:
            pickle.dump(final_outputs, file)
        return final_outputs

    def _convert_to_dataset(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs_as_dataset: Dict[str, Any] = {}
        for i, image_id in enumerate(outputs["image_ids"]):
            outputs_as_dataset[image_id] = {"mask": outputs["masks"][i]}
            if "labels" in outputs:
                outputs_as_dataset[image_id]["label"] = outputs["labels"][i]
        return outputs_as_dataset

    def _update_directories(self, fold_name: str) -> str:
        output_dir = os.path.join(self._config["experiments_dir"], self._config["group"])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        config_path = os.path.join(output_dir, "config.yaml")
        if not os.path.exists(config_path):
            with open(config_path, "w") as file:
                yaml.dump(self._config, file)
        checkpoints_dir = os.path.join(self._config["experiments_dir"], self._config["group"], "checkpoints/")
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        return os.path.join(checkpoints_dir, fold_name + ".pt")

    @property
    def _mask_loss(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self._config["loss"] == "binary_crossentropy":
            return binary_crossentropy
        elif self._config["loss"] == "jaccard":
            return jaccard_loss
        elif self._config["loss"] == "dice":
            return dice_loss
        elif self._config["loss"] == "focal":
            return focal_loss
        elif self._config["loss"] == "tversky":
            return tversky_loss
        elif self._config["loss"] == "jaccard_binary_crossentropy":
            return jaccard_binary_crossentropy_loss
        else:
            raise ValueError(f"Improper loss name: {self._config['loss']}")
