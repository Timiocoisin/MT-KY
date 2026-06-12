#!/usr/bin/env python3
"""Train one head (reg/coral/ldl/hod) on a public ordinal benchmark (§10)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boilerwear.benchmark import (
    DATASET_SPECS,
    BenchmarkDataset,
    BenchmarkNet,
    benchmark_loss,
    collate_benchmark,
    load_benchmark_records,
)
from boilerwear.utils.logger import setup_logger
from boilerwear.utils.seed import get_device, set_seed
from boilerwear.utils.seed import project_root


def run_eval(model, loader, device, num_stages):
    import numpy as np
    from boilerwear.utils.metrics import compute_metrics

    model.eval()
    yt, yp, ft, fp = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["image"].to(device))
            yt.append(batch["wear_pct"].numpy())
            yp.append(out["wear_pct"].cpu().numpy())
            ft.append(batch["folder_id"].numpy())
            fp.append(out["folder_id_pred"].cpu().numpy())
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    m = compute_metrics(yt, yp, np.concatenate(ft), np.concatenate(fp))
    m["mae_label_units"] = round(m["mae"] * num_stages / 100.0, 4)  # e.g. years
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--head", type=str, required=True, choices=["reg", "coral", "ldl", "hod"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0, help="-1 for CPU")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", "--max-epochs", dest="epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ldl-sigma", type=float, default=2.0)
    parser.add_argument("--no-pretrain", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    root = project_root()
    set_seed(args.seed)
    spec = DATASET_SPECS[args.dataset]
    num_stages = spec["max_label"] - spec["min_label"] + 1
    device = get_device("cpu" if args.gpu < 0 else "cuda", gpu_id=None if args.gpu < 0 else args.gpu)

    csv_path = root / "outputs" / "splits" / f"benchmark_{args.dataset}.csv"
    data_root = root / "datasets_public" / args.dataset
    need_ldl = args.head in ("ldl", "hod")
    common = dict(
        data_root=data_root, min_label=spec["min_label"], max_label=spec["max_label"],
        img_size=args.img_size, ldl_sigma_stages=args.ldl_sigma, need_ldl=need_ldl,
    )
    train_ds = BenchmarkDataset(load_benchmark_records(csv_path, "train"), augment=True, **common)
    val_ds = BenchmarkDataset(load_benchmark_records(csv_path, "val"), augment=False, **common)
    mk = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, collate_fn=collate_benchmark)
    train_loader, val_loader = mk(train_ds, True), mk(val_ds, False)

    model_name = f"{args.head}_{args.backbone}"
    out_dir = root / "outputs" / "results" / f"benchmark_{args.dataset}" / model_name / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = root / "outputs" / "checkpoints" / f"benchmark_{args.dataset}" / model_name / f"seed{args.seed}" / "best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"bench.{model_name}", out_dir / "train.log")

    model = BenchmarkNet(
        backbone=args.backbone, head=args.head, num_stages=num_stages,
        num_bins=spec["num_bins"], pretrained=not args.no_pretrain,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)

    logger.info(f"Start benchmark training dataset={args.dataset} head={args.head} "
                f"backbone={args.backbone} seed={args.seed} train={len(train_ds)} val={len(val_ds)} "
                f"num_stages={num_stages} num_bins={spec['num_bins']} device={device}")

    best, best_epoch, stale, history = float("inf"), 0, 0, []
    accum = max(1, args.accum_steps)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, nb = 0.0, 0
        optimizer.zero_grad()
        for bi, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch["image"])
            loss, _ = benchmark_loss(args.head, out, batch, spec["num_bins"])
            (loss / accum).backward()
            if (bi + 1) % accum == 0 or (bi + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            total += float(loss.item()); nb += 1
        scheduler.step()
        vm = run_eval(model, val_loader, device, num_stages)
        history.append({"epoch": epoch, "train_loss": total / max(nb, 1), **vm})
        improved = vm["mae"] < best
        if improved:
            best, best_epoch, stale = vm["mae"], epoch, 0
            torch.save({"model_state": model.state_dict(),
                        "args": vars(args), "epoch": epoch, "val_metrics": vm}, ckpt_path)
        else:
            stale += 1
        logger.info(f"Epoch {epoch}/{args.epochs}{' *best*' if improved else ''} | "
                    f"train_loss={total / max(nb, 1):.4f} | val_mae={vm['mae']:.4f} "
                    f"({vm['mae_label_units']:.2f} label-units) | qwk={vm['qwk']:.4f} | "
                    f"best={best:.4f}@ep{best_epoch} | stale={stale}/{args.patience}")
        if stale >= args.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    summary = {"dataset": args.dataset, "model": model_name, "seed": args.seed,
               "best_epoch": best_epoch, "best_val_mae": best,
               "checkpoint": str(ckpt_path), "total_time_sec": round(time.time() - t0, 1),
               "history": history}
    with (out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Benchmark training done. best_val_mae={best:.4f}  ckpt={ckpt_path}")


if __name__ == "__main__":
    main()
