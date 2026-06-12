from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from boilerwear.data.dataset import BoilerWearDataset, collate_batch
from boilerwear.data.splits import load_split_records
from boilerwear.engine.evaluator import evaluate_hog_lr, evaluate_model
from boilerwear.losses import LossBuilder
from boilerwear.models.registry import build_model
from boilerwear.utils.logger import setup_logger
from boilerwear.utils.seed import set_seed, validate_gpu_ids


def _build_data_parallel(model: torch.nn.Module, gpu_ids: list[int]) -> torch.nn.DataParallel:
    """Wrap model in DataParallel, skipping PyTorch GPU memory balance check on vGPU.

    HAMI/vGPU drivers may report ``total_memory=0``, which makes PyTorch's
    ``_check_balance`` divide by zero. Training still works; only the warning is unsafe.
    """
    import contextlib
    import importlib

    dp_mod = importlib.import_module("torch.nn.parallel.data_parallel")

    skip_balance = any(
        torch.cuda.get_device_properties(i).total_memory == 0 for i in gpu_ids
    )
    if not skip_balance:
        return torch.nn.DataParallel(model, device_ids=gpu_ids)

    orig_check = dp_mod._check_balance

    @contextlib.contextmanager
    def _no_balance_check():
        dp_mod._check_balance = lambda _device_ids: None
        try:
            yield
        finally:
            dp_mod._check_balance = orig_check

    with _no_balance_check():
        return torch.nn.DataParallel(model, device_ids=gpu_ids)


def should_show_progress(cfg: dict[str, Any]) -> bool:
    train_cfg = cfg.get("train", {})
    if train_cfg.get("show_progress") is True:
        return True
    if train_cfg.get("show_progress") is False:
        return False
    return sys.stdout.isatty()


class Trainer:
    def __init__(self, cfg: dict[str, Any], out_dir: Path) -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = Path(cfg.get("output", {}).get("checkpoints_dir", "outputs/checkpoints"))
        self.model_name = cfg["model"].get("name", "model")
        self.protocol = cfg.get("protocol", "protocol2")
        self.split_name = cfg.get("split_name", self.protocol)
        self.run_tag = cfg.get("run_tag", "")
        self.seed = cfg.get("seed", 42)
        train_cfg = cfg.get("train", {})
        if train_cfg.get("device") == "cpu":
            self.gpu_ids: list[int] = []
            self.device = torch.device("cpu")
            self.use_data_parallel = False
        else:
            self.gpu_ids = list(train_cfg.get("gpu_ids") or [])
            if not self.gpu_ids and train_cfg.get("gpu_id") is not None:
                self.gpu_ids = [int(train_cfg["gpu_id"])]
            if not self.gpu_ids:
                self.gpu_ids = [0]
            validate_gpu_ids(self.gpu_ids)
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
            self.use_data_parallel = len(self.gpu_ids) > 1
        self.show_progress = should_show_progress(cfg)
        set_seed(self.seed)
        self.logger = setup_logger(
            f"train.{self.model_name}",
            self.out_dir / "train.log",
        )

    def _device_label(self) -> str:
        if self.device.type != "cuda":
            return "cpu"
        names = [torch.cuda.get_device_name(i) for i in self.gpu_ids]
        if self.use_data_parallel:
            return f"cuda:{self.gpu_ids} DataParallel ({', '.join(names)})"
        return f"{self.device} ({names[0]})"

    def _wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        model = model.to(self.device)
        if self.use_data_parallel:
            model = _build_data_parallel(model, self.gpu_ids)
        return model

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if isinstance(model, torch.nn.DataParallel) else model

    def _log_run_header(self, train_n: int, val_n: int) -> None:
        self.logger.info(
            f"Start training model={self.model_name} protocol={self.protocol} seed={self.seed} "
            f"device={self._device_label()} train={train_n} val={val_n} "
            f"progress_bar={'on' if self.show_progress else 'off'}"
        )

    def _build_datasets(self):
        data_cfg = self.cfg["data"]
        root = Path(data_cfg["data_root"])
        if not root.is_absolute():
            root = Path(self.cfg.get("project_root", ".")) / root
        split_csv = Path(data_cfg["splits_dir"]) / f"{self.split_name}.csv"
        if not split_csv.is_absolute():
            split_csv = Path(self.cfg.get("project_root", ".")) / split_csv

        train_records = load_split_records(split_csv, "train")
        val_records = load_split_records(split_csv, "val")

        common = dict(
            auto_resize=data_cfg.get("auto_resize", True),
            data_root=root,
            num_strips=data_cfg.get("num_strips", 6),
            strip_size=data_cfg.get("strip_size", 256),
            label_mode=self.cfg["model"].get("label_mode", data_cfg.get("label_mode", "hard")),
            ldl_sigma_folders=data_cfg.get("ldl_sigma_folders", 1.0),
            normalize_mean=data_cfg.get("normalize_mean"),
            normalize_std=data_cfg.get("normalize_std"),
            aug_cfg={
                "brightness": data_cfg.get("brightness", 0.15),
                "contrast": data_cfg.get("contrast", 0.15),
                "saturation": data_cfg.get("saturation", 0.10),
            },
        )
        train_ds = BoilerWearDataset(train_records, photometric_aug=data_cfg.get("photometric_aug", True), **common)
        val_ds = BoilerWearDataset(val_records, photometric_aug=False, **common)
        return train_ds, val_ds, root, split_csv

    def _build_loader(self, dataset, shuffle: bool) -> DataLoader:
        loader_cfg = self.cfg.get("loader", {})
        return DataLoader(
            dataset,
            batch_size=loader_cfg.get("batch_size", 8),
            shuffle=shuffle,
            num_workers=loader_cfg.get("num_workers", 0),
            pin_memory=loader_cfg.get("pin_memory", True),
            collate_fn=collate_batch,
            drop_last=loader_cfg.get("drop_last", False),
        )

    def train(self) -> dict[str, Any]:
        model_cfg = self.cfg["model"]
        if model_cfg.get("family") == "hog_lr":
            return self._train_hog_lr()

        train_ds, val_ds, _, _ = self._build_datasets()
        train_loader = self._build_loader(train_ds, shuffle=True)
        val_loader = self._build_loader(val_ds, shuffle=False)
        self._log_run_header(len(train_ds), len(val_ds))

        model = self._wrap_model(build_model(self.cfg))
        loss_builder = LossBuilder(self.cfg)
        train_cfg = self.cfg.get("train", {})
        opt_cfg = train_cfg.get("optimizer", {})

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg.get("lr", 1e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
        )
        epochs = train_cfg.get("epochs", 100)
        patience = train_cfg.get("early_stopping_patience", 15)
        grad_clip = train_cfg.get("grad_clip_norm", 1.0)
        accum_steps = max(1, int(train_cfg.get("accum_steps", 1)))
        if accum_steps > 1:
            micro = self.cfg.get("loader", {}).get("batch_size", 8)
            self.logger.info(
                f"Gradient accumulation enabled: micro-batch={micro} x accum_steps={accum_steps} "
                f"-> effective batch={micro * accum_steps}"
            )

        scheduler_cfg = train_cfg.get("scheduler", {})
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - scheduler_cfg.get("warmup_epochs", 5), 1),
            eta_min=scheduler_cfg.get("min_lr", 1e-6),
        )

        best_val = float("inf")
        best_epoch = 0
        stale = 0
        history = []
        t_run = time.time()

        seed_dir = f"seed{self.seed}" + (f"_{self.run_tag}" if self.run_tag else "")
        ckpt_path = self.ckpt_dir / self.protocol / self.model_name / seed_dir / "best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        log_interval = train_cfg.get("log_batch_interval", 0)

        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            n_batches = 0
            t_epoch = time.time()

            iterator = train_loader
            pbar = None
            if self.show_progress:
                pbar = tqdm(
                    train_loader,
                    desc=f"[{self.model_name}] Epoch {epoch}/{epochs}",
                    leave=True,
                    dynamic_ncols=True,
                )
                iterator = pbar

            optimizer.zero_grad()
            for batch_idx, batch in enumerate(iterator):
                batch_t = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                outputs = model(batch_t["strips"])
                loss, _loss_logs = loss_builder(outputs, batch_t)
                (loss / accum_steps).backward()
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                train_loss += float(loss.item())
                n_batches += 1

                if pbar is not None:
                    pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{train_loss / n_batches:.4f}")
                elif log_interval > 0 and n_batches % log_interval == 0:
                    self.logger.info(
                        f"Epoch {epoch}/{epochs} batch {n_batches}/{len(train_loader)} "
                        f"loss={loss.item():.4f} avg={train_loss / n_batches:.4f}"
                    )

            if pbar is not None:
                pbar.close()

            scheduler.step()
            val_metrics = evaluate_model(model, val_loader, self.device, loss_builder)
            avg_train = train_loss / max(n_batches, 1)
            val_mae = val_metrics["mae"]
            lr = optimizer.param_groups[0]["lr"]
            epoch_sec = time.time() - t_epoch

            history.append({"epoch": epoch, "train_loss": avg_train, "lr": lr, **val_metrics})

            improved = val_mae < best_val
            if improved:
                best_val = val_mae
                best_epoch = epoch
                stale = 0
                torch.save(
                    {
                        "model_state": self._unwrap_model(model).state_dict(),
                        "cfg": self.cfg,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                    },
                    ckpt_path,
                )
                save_mark = " *best*"
            else:
                stale += 1
                save_mark = ""

            self.logger.info(
                f"Epoch {epoch}/{epochs}{save_mark} | "
                f"train_loss={avg_train:.4f} | val_mae={val_mae:.4f} | val_rmse={val_metrics['rmse']:.4f} | "
                f"val_qwk={val_metrics['qwk']:.4f} | val_acc5={val_metrics['acc_at_5']:.4f} | "
                f"lr={lr:.2e} | best={best_val:.4f}@ep{best_epoch} | stale={stale}/{patience} | "
                f"time={epoch_sec:.0f}s"
            )

            if stale >= patience:
                self.logger.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

        total_sec = time.time() - t_run
        self.logger.info(
            f"Finished model={self.model_name} best_val_mae={best_val:.4f} @ epoch {best_epoch} "
            f"total_time={total_sec:.0f}s checkpoint={ckpt_path}"
        )

        summary = {
            "model": self.model_name,
            "protocol": self.protocol,
            "seed": self.seed,
            "device": self._device_label(),
            "gpu_ids": self.gpu_ids,
            "data_parallel": self.use_data_parallel,
            "best_epoch": best_epoch,
            "best_val_mae": best_val,
            "checkpoint": str(ckpt_path),
            "total_time_sec": round(total_sec, 1),
            "history": history,
        }
        with (self.out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary

    def _train_hog_lr(self) -> dict[str, Any]:
        from boilerwear.models.baselines.hog_lr import HogLRModel
        from PIL import Image
        import numpy as np

        data_cfg = self.cfg["data"]
        root = Path(data_cfg["data_root"])
        if not root.is_absolute():
            root = Path(self.cfg.get("project_root", ".")) / root
        split_csv = Path(data_cfg["splits_dir"]) / f"{self.protocol}.csv"
        if not split_csv.is_absolute():
            split_csv = Path(self.cfg.get("project_root", ".")) / split_csv

        train_records = load_split_records(split_csv, "train")
        val_records = load_split_records(split_csv, "val")
        self._log_run_header(len(train_records), len(val_records))

        hog_cfg = self.cfg["model"]
        model = HogLRModel(
            orientations=hog_cfg.get("hog_orientations", 9),
            pixels_per_cell=tuple(hog_cfg.get("hog_pixels_per_cell", [16, 16])),
            cells_per_block=tuple(hog_cfg.get("hog_cells_per_block", [2, 2])),
        )

        self.logger.info(f"HOG+LR: extracting features from {len(train_records)} train images...")
        train_images = [np.array(Image.open(root / r.image_path).convert("RGB")) for r in train_records]
        train_y = np.array([r.wear_pct for r in train_records])
        t0 = time.time()
        model.fit(train_images, train_y)
        fit_time = time.time() - t0

        val_metrics = evaluate_hog_lr(model, val_records, root)
        self.logger.info(f"HOG+LR done fit_time={fit_time:.1f}s val_mae={val_metrics['mae']:.4f}")

        import pickle
        ckpt_path = self.ckpt_dir / self.protocol / self.model_name / f"seed{self.seed}" / "hog_lr.pkl"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        with ckpt_path.open("wb") as f:
            pickle.dump(model, f)

        summary = {
            "model": self.model_name,
            "protocol": self.protocol,
            "seed": self.seed,
            "fit_time_sec": fit_time,
            "val_metrics": val_metrics,
            "checkpoint": str(ckpt_path),
        }
        with (self.out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary
