from typing import Any, Dict, List

import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from matplotlib import pyplot as plt
from matplotlib import patches

from sartorius.dataset import Dataset
from sartorius.encoder import Encoder


class Visualizer:
    def __init__(self, encoder: Encoder) -> None:
        self._encoder = encoder

    def visualize_config(self, config_df: pd.DataFrame) -> None:
        display(config_df.head())

    def visualize_cell_types(self, config_df: pd.DataFrame) -> None:
        with plt.style.context('ggplot'):
            plt.figure(figsize=(25, 8))
            plt.title("distribution of cell types", fontsize=35, pad=25)
            config_df["cell_type"].value_counts(normalize=True).plot.barh()
            plt.yticks(fontsize=27, rotation=90, va="center")
            plt.xticks(fontsize=27)
            plt.show()

    def visualize_original_and_transformed_samples(
        self,
        original_dataset: Dataset,
        transformed_dataset: Dataset,
        image_ids: List[str],
    ) -> None:
        for image_id in image_ids:
            _, axs = plt.subplots(2, 4, figsize=(40, 17))
            self._visualize_sample(axs[0], True, original_dataset[image_id])
            self._visualize_sample(axs[1], False, transformed_dataset[image_id])
            plt.show()

    def visualize_train_and_valid_predictions(
        self,
        train_predictions: Dict[str, Any],
        valid_predictions: Dict[str, Any],
        targets: Dataset,
        image_ids: List[str],
    ) -> None:
        for image_id in image_ids:
            _, axs = plt.subplots(2, 4, figsize=(40, 17))
            train_prediction = train_predictions["outputs"][image_id]
            valid_prediction = valid_predictions["outputs"][image_id]
            self._visualize_prediction(axs[0], True, "train", train_prediction, targets[image_id])
            self._visualize_prediction(axs[1], False, "valid", valid_prediction, targets[image_id])
            plt.show()
            train_metrics = pd.Series(train_prediction["metrics"]).to_frame("train").T
            train_metrics.columns = [x.split("/")[1] for x in train_metrics.columns]
            valid_metrics = pd.Series(valid_prediction["metrics"]).to_frame("valid").T
            valid_metrics.columns = [x.split("/")[1] for x in valid_metrics.columns]
            metrics = pd.concat([train_metrics, valid_metrics])
            display(metrics)

    def visualize_test_predictions(
        self, test_predictions: Dict[str, Any], targets: Dataset, image_ids: List[str]
    ) -> None:
        for image_id in image_ids:
            _, axs = plt.subplots(1, 4, figsize=(40, 17))
            test_prediction = test_predictions["outputs"][image_id]
            self._visualize_prediction(axs, True, "test", test_prediction, targets[image_id])
            plt.show()
            metrics = pd.Series(test_prediction["metrics"]).to_frame("test").T
            metrics.columns = [x.split("/")[1] for x in metrics.columns]
            display(metrics)

    def visualize_metrics(
        self,
        train_outputs: Dict[str, Any],
        valid_outputs: Dict[str, Any],
        test_outputs: Dict[str, Any],
        config_df: pd.DataFrame,
    ) -> None:
        train_df = pd.DataFrame({
            k: {k.split("/")[1]: v for k, v in x["metrics"].items()} for k, x in train_outputs["outputs"].items()
        }).T.assign(subset="train")
        valid_df = pd.DataFrame({
            k: {k.split("/")[1]: v for k, v in x["metrics"].items()} for k, x in valid_outputs["outputs"].items()
        }).T.assign(subset="valid")
        test_df = pd.DataFrame({
            k: {k.split("/")[1]: v for k, v in x["metrics"].items()} for k, x in test_outputs["outputs"].items()
        }).T.assign(subset="test")
        df = pd.concat([train_df, valid_df, test_df]).join(config_df["cell_type"])
        iou_map_score = pd.Series({
            "train": train_outputs["iou_map_score"],
            "valid": valid_outputs["iou_map_score"],
            "test": test_outputs["iou_map_score"],
        }, name="iou_map_score")
        display(df.groupby("subset").mean().rename_axis(index=None).loc[["train", "valid", "test"]].join(iou_map_score))
        display(
            df.groupby(["subset", "cell_type"]).mean().rename_axis(index=[None, None]).loc[["train", "valid", "test"]]
        )

    def _visualize_sample(self, axs: List[plt.Axes], header: bool, sample: Dict[str, Any]) -> None:
        image = sample["image"].permute(1, 2, 0).numpy()
        mask = sample["mask"].numpy()
        mask_ma = np.ma.masked_where(np.isclose(mask, 0.0), mask)
        if header:
            to_display = f"image id: <strong>{sample['image_id']}</strong>"
            to_display = f"{to_display} | label: <strong>{self._encoder.decode_label(sample['label'].item())}</strong>"
            display(Markdown(f"<br/><center><font size='5'>{to_display}</font></center><br/>"))
        self._configure_image(axs[0], "image", header)
        axs[0].imshow(image)
        axs[0].set_ylabel(f"{'original' if header else 'transformed'}", fontsize=40, labelpad=20)
        self._configure_image(axs[1], "mask (transparent)", header)
        axs[1].imshow(image)
        axs[1].imshow(mask_ma, cmap="Wistia", alpha=0.2)
        self._configure_image(axs[2], "mask", header)
        axs[2].imshow(image)
        axs[2].imshow(mask_ma, cmap="Wistia")
        self._configure_image(axs[3], "bounding boxes", header)
        axs[3].imshow(image)
        axs[3].imshow(mask_ma, cmap="Wistia")
        for x_min, y_min, x_max, y_max in sample['bboxes']:
            width = x_max - x_min
            height = y_max - y_min
            if np.isclose(width, 0.0) or np.isclose(height, 0.0):
                break
            axs[3].add_patch(
                patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor="r", facecolor="none")
            )

    def _visualize_prediction(
        self, axs: List[plt.Axes], header: bool, mode: str, prediction: Dict[str, Any], target: Dict[str, Any]
    ) -> None:
        image = target["image"].permute(1, 2, 0).numpy()
        mask_true = target["mask"].numpy()
        mask_pred = (prediction["mask"] > 0.5).astype(float)
        class_true = self._encoder.decode_label(target["label"].item())
        intersection = mask_true * mask_pred
        mask_true_fn = mask_true - intersection
        mask_pred_fp = mask_pred - intersection
        if header:
            to_display = f"image id: <strong>{target['image_id']}</strong>"
            to_display = f"{to_display} | true label: <strong>{class_true}</strong>"
            display(Markdown(f"<br/><center><font size='5'>{to_display}</font></center><br/>"))
        self._configure_image(axs[0], "image", header)
        axs[0].imshow(image)
        display_label = f"{mode}"
        if "label" in prediction:
            class_pred = self._encoder.decode_label(prediction["label"].argmax())
            bold_start = r"$\bf{"
            bold_end = r"}$"
            display_label = f"{display_label} pred: {bold_start}{class_pred}{bold_end}"
        axs[0].set_ylabel(display_label, fontsize=40, labelpad=20)
        self._configure_image(axs[1], "original mask", header)
        mask_true_ma = np.ma.masked_where(np.isclose(mask_true, 0.0), mask_true)
        axs[1].imshow(image)
        axs[1].imshow(mask_true_ma, cmap="Wistia", alpha=0.25)
        self._configure_image(axs[2], "predicted mask", header)
        mask_pred_ma = np.ma.masked_where(np.isclose(mask_pred, 0.0), mask_pred)
        axs[2].imshow(image)
        axs[2].imshow(mask_pred_ma, cmap="RdYlGn", alpha=0.25)
        self._configure_image(axs[3], "intersection", header)
        mask_true_fn_ma = np.ma.masked_where(np.isclose(mask_true_fn, 0.0), mask_true_fn)
        mask_pred_fp_ma = np.ma.masked_where(np.isclose(mask_pred_fp, 0.0), mask_pred_fp)
        intersection_ma = np.ma.masked_where(np.isclose(intersection, 0.0), intersection)
        axs[3].imshow(image)
        axs[3].imshow(mask_true_fn_ma, cmap="Wistia", alpha=0.35)
        axs[3].imshow(mask_pred_fp_ma, cmap="RdYlGn", alpha=0.35)
        axs[3].imshow(intersection_ma, cmap="summer", alpha=0.35)

    def _configure_image(self, ax: plt.Axes, title: str, header: bool) -> None:
        if header:
            ax.set_title(title, fontsize=40, pad=20)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
