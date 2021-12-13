import os
import random
from itertools import chain
from typing import Any, Dict, List, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold, train_test_split
from torch.backends import cudnn

from sartorius.dataset import Dataset, SubmissionDataset
from sartorius.encoder import Encoder
from sartorius.trainer import Trainer
from sartorius.visualizer import Visualizer


def set_global_seed(config: Dict[str, Any]) -> None:
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    os.environ['PYTHONHASHSEED'] = str(config["seed"])
    cudnn.deterministic = True
    cudnn.benchmark = False


def prepare_config_df(config: Dict[str, Any]) -> pd.DataFrame:
    config_df = pd.read_csv(os.path.join(config["data_dir"], "train.csv"), parse_dates=["sample_date"])
    annotations = config_df.groupby("id")["annotation"].agg(tuple).rename("annotations").reset_index()
    config_df = config_df.drop_duplicates(subset=["id"]).drop(columns=["annotation"]).merge(annotations)
    config_df["path"] = config_df["id"].apply(lambda x: f"{os.path.join(config['data_dir'], 'train/', x)}.png")
    config_df["plate_time"] = pd.to_timedelta(config_df["plate_time"])
    config_df["elapsed_timedelta"] = pd.to_timedelta(config_df["elapsed_timedelta"])
    return config_df.set_index("id").rename_axis(index=None)


def load_transformations(config: Dict[str, Any], default: bool, bboxes: bool) -> A.Compose:
    transformations = []
    if not default and config["augmentation"]:
        transformations.append(A.HorizontalFlip(p=0.2))
        transformations.append(A.VerticalFlip(p=0.2))
        transformations.append(A.OneOf(
            [A.CLAHE(), A.GaussianBlur(), A.HueSaturationValue(), A.GaussNoise(), A.RandomBrightnessContrast()], p=0.25
        ))
        transformations.append(A.RandomSizedCrop(
            min_max_height=(int(config["original_height"] / 2), config["original_height"]),
            height=config["original_height"],
            width=config["original_width"],
            w2h_ratio=config["original_width"] / config["original_height"],
            p=0.25,
        ))
    transformations.append(ToTensorV2())
    return A.Compose(
        transformations, bbox_params=A.BboxParams(format="pascal_voc", label_fields=[]) if bboxes else None
    )


def prepare_datasets(
    config: Dict[str, Any], config_df_train: pd.DataFrame, config_df_test: pd.DataFrame, encoder: Encoder
) -> Tuple[List[Tuple[Dataset, Dataset]], Dataset, Dataset, Dataset, SubmissionDataset]:
    train_transformations = load_transformations(config, default=False, bboxes=True)
    test_transformations = load_transformations(config, default=True, bboxes=True)
    train_fold_transformations = load_transformations(config, default=False, bboxes=False)
    test_fold_transformations = load_transformations(config, default=True, bboxes=False)
    train_dataset = Dataset(config_df_train, encoder, train_transformations, return_bboxes=True)
    valid_dataset = Dataset(config_df_train, encoder, test_transformations, return_bboxes=True)
    test_dataset = Dataset(config_df_test, encoder, test_transformations, return_bboxes=True)
    submission_dataset = SubmissionDataset(os.path.join(config["data_dir"], "test/"), test_fold_transformations)
    folds = [
        (
            Dataset(config_df_train.iloc[train_idx], encoder, train_fold_transformations, return_bboxes=False),
            Dataset(config_df_train.iloc[valid_idx], encoder, test_fold_transformations, return_bboxes=False),
        )
        for train_idx, valid_idx in KFold(config["num_folds"]).split(config_df_train)
    ]
    return folds, train_dataset, valid_dataset, test_dataset, submission_dataset


def main(config: Dict[str, Any]) -> None:
    set_global_seed(config)
    config_df = prepare_config_df(config)
    config_df_train, config_df_test = train_test_split(config_df, test_size=config["test_size"])
    encoder = Encoder().fit(config_df_train["cell_type"])
    folds, train_dataset, valid_dataset, test_dataset, submission_dataset = prepare_datasets(
        config, config_df_train, config_df_test, encoder
    )
    train_image_ids = list(chain(*(x.sample(6).index.tolist() for i, x in config_df_train.groupby("cell_type"))))
    test_image_ids = list(chain(*(x.sample(6).index.tolist() for i, x in config_df_test.groupby("cell_type"))))
    visualizer = Visualizer(encoder)
    visualizer.visualize_config(config_df_train)
    visualizer.visualize_cell_types(config_df_train)
    visualizer.visualize_original_and_transformed_samples(valid_dataset, train_dataset, train_image_ids)
    trainer = Trainer(config, encoder)
    valid_outputs = trainer.cross_validate(folds, valid_dataset)
    train_outputs = trainer.evaluate(valid_dataset, "train")
    visualizer.visualize_train_and_valid_predictions(train_outputs, valid_outputs, valid_dataset, train_image_ids)
    test_outputs = trainer.evaluate(test_dataset, "test")
    visualizer.visualize_test_predictions(test_outputs, test_dataset, test_image_ids)
    visualizer.visualize_metrics(train_outputs, valid_outputs, test_outputs, config_df)
    submission_outputs = trainer.evaluate(submission_dataset, "submission")
