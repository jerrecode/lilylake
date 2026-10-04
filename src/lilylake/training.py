"""Bounded data caches, safe atomic checkpoints, and resumable CPU/GPU training."""

import importlib.metadata
import json
import os
import platform
import random
import subprocess
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .audio import features, load_audio
from .config import Config
from .data import compose, split_for
from .evaluation import _metrics, evaluate_notes, frame_metrics
from .inference import decode, select_device
from .models import EventModel, event_loss, targets
from .music.schema import Piece
from .rendering import render_piece


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        torch.save(state, file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    # Never unrestricted pickle-load model files received from outside the project.
    return torch.load(path, map_location="cpu", weights_only=True)


class MusicDataset(Dataset):
    def __init__(self, manifest, config, split="train", all_rows=False):
        self.root = Path(manifest).parent
        self.config = config
        self.rows = [
            json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()
        ]
        self.rows = [r for r in self.rows if all_rows or r["split"] == split]
        if not self.rows:
            raise ValueError(
                f"No {split} compositions in manifest; generate more or use explicit sanity-overfit mode"
            )
        self.cache = OrderedDict()
        self.cache_bytes = 0

    def __len__(self):
        return len(self.rows)

    def _load(self, index):
        row = self.rows[index]
        piece = Piece.load(self.root / row["events"])
        audio = load_audio(self.root / row["audio"], self.config.sample_rate)
        return piece, audio

    def __getitem__(self, index):
        if index in self.cache:
            value = self.cache.pop(index)
            self.cache[index] = value
            return value
        piece, audio = self._load(index)
        x = features(audio, self.config)
        y = targets(piece, x.shape[1], self.config)
        value = (x, y, piece)
        size = x.numel() * x.element_size() + sum(t.numel() * t.element_size() for t in y.values())
        budget = self.config.cache_mb * 1024 * 1024
        while self.cache and self.cache_bytes + size > budget:
            _, (_, old_targets, _) = self.cache.popitem(last=False)
            # Size tracked separately because feature length varies.
            self.cache_bytes = sum(
                v[0].numel() * v[0].element_size()
                + sum(t.numel() * t.element_size() for t in v[1].values())
                for v in self.cache.values()
            )
        if size <= budget:
            self.cache[index] = value
            self.cache_bytes += size
        return value

    @property
    def identities(self):
        return [r["composition_id"] for r in self.rows]


class OnlineDataset(MusicDataset):
    """Finite epoch view of an unbounded deterministic procedural seed space."""

    def __init__(self, count, config, split="train", level=1):
        self.config = config
        self.count = count
        self.split = split
        self.level = level
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.set_epoch(0, level)

    def set_epoch(self, epoch, level):
        if getattr(self, "epoch", None) == epoch and self.level == level:
            return
        self.epoch = epoch
        self.level = level
        self.cache.clear()
        self.cache_bytes = 0
        self.rows = []
        seed = self.config.seed + epoch * 100000
        while len(self.rows) < self.count:
            piece = compose(seed, self.level, self.config.instruments)
            pid = piece.metadata["composition_id"]
            if split_for(pid) == self.split:
                self.rows.append({"seed": seed, "composition_id": pid, "level": self.level})
            seed += 1

    def _load(self, index):
        row = self.rows[index]
        piece = compose(row["seed"], row["level"], self.config.instruments)
        with tempfile.TemporaryDirectory() as temp:
            result = render_piece(
                piece, temp, "procedural", self.config.sample_rate, seed=row["seed"]
            )
            return Piece.load(result["events"]), load_audio(
                result["audio"], self.config.sample_rate
            )


def collate(batch):
    max_frames = max(x.shape[1] for x, _, _ in batch)
    b = len(batch)
    channels = batch[0][0].shape[0]
    instruments = batch[0][1]["frame"].shape[1]
    x = torch.zeros(b, channels, max_frames, 128)
    y = {k: torch.zeros(b, max_frames, instruments, 128) for k in batch[0][1]}
    time_mask = torch.zeros(b, max_frames)
    for i, (feature, target, _) in enumerate(batch):
        frames = feature.shape[1]
        x[i, :, :frames] = feature
        time_mask[i, :frames] = 1
        for k in y:
            y[k][i, :frames] = target[k]
    return x, y, time_mask, [p for _, _, p in batch]


def evaluate_model(model, dataset, config, device="cpu"):
    model.eval()
    losses = []
    frame_counts = np.zeros(3, dtype=np.int64)
    totals = {k: np.zeros(3, dtype=np.int64) for k in ["onset", "note_with_offset"]}
    per = {}
    timing = {k: [] for k in ["onset_mae_ms", "offset_mae_ms", "velocity_mae", "velocity_rmse"]}
    with torch.inference_mode():
        for x, y, piece in dataset:
            pred = model(x.unsqueeze(0).to(device))
            target = {k: v.unsqueeze(0).to(device) for k, v in y.items()}
            loss, _ = event_loss(pred, target, model.pitch_mask)
            losses.append(float(loss))
            probs = {k: v[0].sigmoid().cpu().numpy() for k, v in pred.items()}
            mask = model.pitch_mask.cpu().numpy()[None]
            fm = frame_metrics(
                y["frame"].numpy().astype(bool),
                (probs["frame"] >= config.frame_threshold) & mask.astype(bool),
            )
            frame_counts += [fm["tp"], fm["fp"], fm["fn"]]
            decoded = decode(probs, config, duration=x.shape[1] * config.frame_seconds)
            report = evaluate_notes(piece, decoded)
            for key in totals:
                totals[key] += [report[key]["tp"], report[key]["fp"], report[key]["fn"]]
                for name, values in report["per_instrument"].items():
                    per.setdefault(name, {k: np.zeros(3, dtype=np.int64) for k in totals})
                    per[name][key] += [values[key]["tp"], values[key]["fp"], values[key]["fn"]]
            for k in timing:
                if report[k] is not None:
                    timing[k].append((report[k], report["matched_notes"]))
    return {
        "loss": float(np.mean(losses)),
        "frame": _metrics(*frame_counts),
        **{k: _metrics(*v) for k, v in totals.items()},
        "per_instrument": {
            name: {k: _metrics(*v) for k, v in values.items()} for name, values in per.items()
        },
        **{
            k: sum(v * n for v, n in values) / sum(n for _, n in values) if values else None
            for k, values in timing.items()
        },
        "compositions": len(dataset),
        "composition_ids": dataset.identities,
    }


def provenance(config):
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    versions = {
        name: importlib.metadata.version(name)
        for name in ["torch", "numpy", "scipy", "soundfile", "mido"]
    }
    return {
        "config": config.to_dict(),
        "git_revision": revision,
        "dependencies": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "time_unix": time.time(),
    }


def train(
    manifest,
    output,
    config=None,
    sanity_overfit=False,
    resume=None,
    online_count=0,
    curriculum=False,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume == "auto":
        resume = output / "last.pt" if (output / "last.pt").exists() else None
    state = load_checkpoint(resume) if resume else None
    config = config or (Config(**state["config"]) if state else Config())
    if state:
        immutable = [
            "sample_rate",
            "n_fft",
            "hop",
            "harmonics",
            "width",
            "depth",
            "instruments",
            "batch_size",
            "learning_rate",
            "seed",
            "accumulation",
        ]
        if any(state["config"][k] != config.to_dict()[k] for k in immutable):
            raise ValueError("Resume configuration differs from checkpoint training/model settings")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = select_device(config.device)
    if online_count:
        training = OnlineDataset(online_count, config, "train", 1 if curriculum else 6)
        validation = OnlineDataset(max(2, online_count // 5), config, "validation", training.level)
    else:
        training = MusicDataset(manifest, config, "train", all_rows=sanity_overfit)
        validation = MusicDataset(manifest, config, "validation", all_rows=sanity_overfit)
    if not sanity_overfit and set(training.identities) & set(validation.identities):
        raise ValueError("Composition leakage between train and validation")
    model = EventModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    amp_enabled = config.mixed_precision and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    generator = torch.Generator().manual_seed(config.seed)
    start_epoch = 0
    step = 0
    best = -1.0
    history = []
    level = training.level if online_count else None
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_epoch = state["epoch"]
        step = state["step"]
        best = state["best"]
        history = state["history"]
        level = state.get("curriculum_level", level)
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["sampler_rng"])
        if not online_count and state.get("dataset_ids") != training.identities:
            raise ValueError("Resume dataset identities differ")
        if device == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    run = provenance(config)
    run.update(
        {
            "sanity_overfit": sanity_overfit,
            "online_count": online_count,
            "train_ids": training.identities,
            "validation_ids": validation.identities,
            "resume_epoch": start_epoch,
        }
    )
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    config.save(output / "config.json")
    last_validation = state.get("validation", {}) if state else {}
    for epoch in range(start_epoch, config.epochs):
        if online_count:
            training.set_epoch(epoch, level)
            validation.set_epoch(0, level)
        loader = DataLoader(
            training,
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate,
            num_workers=config.workers,
            pin_memory=device == "cuda",
        )
        model.train()
        epoch_losses = []
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (x, y, time_mask, _) in enumerate(loader):
            x = x.to(device)
            y = {k: v.to(device) for k, v in y.items()}
            time_mask = time_mask.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=amp_enabled):
                pred = model(x)
                loss, components = event_loss(pred, y, model.pitch_mask, time_mask)
                group_size = min(
                    config.accumulation,
                    len(loader) - (batch_index // config.accumulation) * config.accumulation,
                )
                scaled_loss = loss / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss epoch {epoch + 1} batch {batch_index}")
            scaler.scale(scaled_loss).backward()
            if (batch_index + 1) % config.accumulation == 0 or batch_index + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            epoch_losses.append(float(loss.detach()))
        scheduler.step()
        validation_due = (
            (epoch + 1) % config.validation_interval == 0
            or epoch == start_epoch
            or epoch + 1 == config.epochs
        )
        if validation_due:
            last_validation = evaluate_model(model, validation, config, device)
        row = {
            "epoch": epoch + 1,
            "step": step,
            "train_loss": float(np.mean(epoch_losses)),
            "seconds": time.perf_counter() - began,
            "validation": last_validation if validation_due else None,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "curriculum_level": level,
        }
        history.append(row)
        score = last_validation.get("note_with_offset", {}).get("f1", 0)
        improved = validation_due and score > best
        if improved:
            best = score
        if (
            curriculum
            and online_count
            and validation_due
            and score >= config.curriculum_gate
            and level < 10
        ):
            level += 1
        checkpoint = {
            "format_version": 1,
            "epoch": epoch + 1,
            "step": step,
            "best": best,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config.to_dict(),
            "torch_rng": torch.get_rng_state(),
            "sampler_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else None,
            "history": history,
            "validation": last_validation,
            "dataset_ids": training.identities,
            "curriculum_level": level,
            "provenance": run,
        }
        save_checkpoint(output / "last.pt", checkpoint)
        if improved:
            save_checkpoint(output / "best.pt", checkpoint)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(
            f"epoch {epoch + 1}/{config.epochs} loss={row['train_loss']:.4f} val_note_f1={score:.4f} {row['seconds']:.2f}s",
            flush=True,
        )
    return model, history
