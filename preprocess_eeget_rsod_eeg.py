#!/usr/bin/env python3
"""Create a reproducible, train-normalized EEG copy for EEGET-RSOD.

The source pkl files are never modified.  Each valid subject snippet is
processed independently before subject aggregation:

    linear detrend -> 50/100 Hz notch -> 0.5--45 Hz band-pass -> CAR

Channel normalization is fitted only on cleaned training snippets, then
applied unchanged to train, validation and test.  The resulting directory has
the same train-EEG/val-EEG/test-EEG layout expected by the DINO data loader.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import os
import pickle
from pathlib import Path

import numpy as np
from scipy import signal


DEFAULT_ROOT = Path("/home/zgq/data/datasets/EEGET-RSOD/coco")
SPLIT_DIRS = {
    "train": "train-EEG",
    "val": "val-EEG",
    "test": "test-EEG",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_ROOT / "eeg_clean"
    )
    parser.add_argument("--lowcut", type=float, default=0.5)
    parser.add_argument("--highcut", type=float, default=45.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument("--notch-q", type=float, default=30.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of independent pkl files to filter concurrently",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing clean pkl files in the output directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="leave completed pkl files untouched and process only missing files",
    )
    parser.add_argument(
        "--reuse-statistics",
        action="store_true",
        help="reuse saved train normalization statistics instead of recomputing them",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_DIRS),
        default=tuple(SPLIT_DIRS),
        help="one or more splits to write (default: train val test)",
    )
    return parser.parse_args()


def validate_args(args):
    if not 0.0 < args.lowcut < args.highcut:
        raise ValueError("require 0 < lowcut < highcut")
    if args.filter_order < 1:
        raise ValueError("filter-order must be positive")
    if args.notch_q <= 0.0:
        raise ValueError("notch-q must be positive")
    if args.workers < 1:
        raise ValueError("workers must be positive")


def load_payload(path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or "eeg_all_subjects" not in payload:
        raise ValueError("{} has no eeg_all_subjects array".format(path))
    data = np.asarray(payload["eeg_all_subjects"], dtype=np.float64)
    if data.ndim != 3 or data.shape[1] != 32 or data.shape[2] != 1500:
        raise ValueError(
            "{} has invalid eeg_all_subjects shape {}; expected [N, 32, 1500]"
            .format(path, tuple(data.shape))
        )
    if data.shape[0] < 1:
        raise ValueError("{} has no valid subject snippets".format(path))
    sfreq = float(payload.get("sfreq", 500.0))
    if sfreq <= 0.0:
        raise ValueError("{} has invalid sfreq={}".format(path, sfreq))
    return payload, data, sfreq


def clean_subject_snippets(data, sfreq, args):
    """Filter [subjects, channels, time] snippets without cross-subject mixing."""
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    cleaned = signal.detrend(data, axis=-1, type="linear")

    nyquist = sfreq / 2.0
    for frequency in (50.0, 100.0):
        if frequency < nyquist:
            b, a = signal.iirnotch(frequency, args.notch_q, fs=sfreq)
            cleaned = signal.sosfiltfilt(
                signal.tf2sos(b, a), cleaned, axis=-1
            )

    sos = signal.butter(
        args.filter_order,
        [args.lowcut, args.highcut],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )
    cleaned = signal.sosfiltfilt(sos, cleaned, axis=-1)

    # Common-average reference per subject and time point.  Bad-channel
    # interpolation needs channel locations/raw continuous recordings and is
    # deliberately not guessed from these already-cut snippets.
    cleaned -= cleaned.mean(axis=1, keepdims=True)
    return cleaned


def statistics_for_path(path, args):
    _, data, sfreq = load_payload(Path(path))
    cleaned = clean_subject_snippets(data, sfreq, args)
    return (
        cleaned.sum(axis=(0, 2), dtype=np.float64),
        np.square(cleaned).sum(axis=(0, 2), dtype=np.float64),
        cleaned.shape[0] * cleaned.shape[2],
    )


def compute_train_statistics(train_paths, args):
    channel_sum = np.zeros(32, dtype=np.float64)
    channel_square_sum = np.zeros(32, dtype=np.float64)
    sample_count = 0
    worker = partial(statistics_for_path, args=args)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (partial_sum, partial_square_sum, count) in enumerate(
            executor.map(worker, train_paths, chunksize=4), start=1
        ):
            channel_sum += partial_sum
            channel_square_sum += partial_square_sum
            sample_count += count
            if index % 100 == 0 or index == len(train_paths):
                print(
                    "statistics: {}/{}".format(index, len(train_paths)),
                    flush=True,
                )

    mean = channel_sum / sample_count
    variance = channel_square_sum / sample_count - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean, std, sample_count


def statistics_path(output_root):
    return output_root / "normalization.json"


def save_statistics(path, mean, std, sample_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "fit_split": "train",
                "channel_sample_count": int(sample_count),
                "mean": np.asarray(mean, dtype=float).tolist(),
                "std": np.asarray(std, dtype=float).tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_saved_statistics(output_root):
    path = statistics_path(output_root)
    if not path.is_file():
        # A stopped run may already contain atomically written pkl files.  They
        # carry the exact same train-fitted statistics, so recover them without
        # recomputing the expensive first pass.
        existing = next(output_root.glob("*-EEG/*.pkl"), None)
        if existing is None:
            raise FileNotFoundError(
                "no saved statistics or completed clean pkl found under {}"
                .format(output_root)
            )
        with existing.open("rb") as handle:
            payload = pickle.load(handle)
        metadata = payload.get("preprocessing", {})
        if metadata.get("normalization_fit_split") != "train":
            raise ValueError("{} lacks train normalization metadata".format(existing))
        mean = np.asarray(metadata["normalization_mean"], dtype=np.float64)
        std = np.asarray(metadata["normalization_std"], dtype=np.float64)
        sample_count = 0
        save_statistics(path, mean, std, sample_count)
        return mean, std, sample_count
    data = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray(data["mean"], dtype=np.float64)
    std = np.asarray(data["std"], dtype=np.float64)
    if mean.shape != (32,) or std.shape != (32,) or bool((std <= 0.0).any()):
        raise ValueError("invalid saved normalization statistics: {}".format(path))
    sample_count = int(data.get("channel_sample_count", 0))
    if sample_count == 0:
        train_clean_dir = output_root / SPLIT_DIRS["train"]
        sample_count = 0
        for path in train_clean_dir.glob("*.pkl"):
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            trials = np.asarray(payload["eeg_all_subjects"])
            sample_count += int(trials.shape[0] * trials.shape[2])
        if sample_count == 0:
            raise ValueError("cannot recover train sample count from {}".format(train_clean_dir))
        save_statistics(path=statistics_path(output_root), mean=mean, std=std,
                        sample_count=sample_count)
    return mean, std, sample_count


def write_clean_payload(path, payload, cleaned, mean, std, args, split):
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    normalized = (cleaned - mean[None, :, None]) / std[None, :, None]
    normalized = np.asarray(normalized, dtype=np.float32)
    result = dict(payload)
    result["eeg_all_subjects"] = normalized
    result["eeg"] = normalized.mean(axis=0, dtype=np.float32)
    result["preprocessing"] = {
        "version": "eeget-rsod-clean-v1",
        "split": split,
        "operations": [
            "linear_detrend",
            "notch_50Hz",
            "notch_100Hz",
            "bandpass_{:.3f}-{:.3f}Hz".format(args.lowcut, args.highcut),
            "common_average_reference",
            "train_only_channel_zscore",
        ],
        "sfreq": float(payload.get("sfreq", 500.0)),
        "normalization_fit_split": "train",
        "normalization_mean": mean.astype(float).tolist(),
        "normalization_std": std.astype(float).tolist(),
    }
    temporary = path.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def process_one(source_path, output_path, mean, std, args, split):
    source_path = Path(source_path)
    output_path = Path(output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            "{} already exists; rerun with --overwrite".format(output_path)
        )
    payload, data, sfreq = load_payload(source_path)
    cleaned = clean_subject_snippets(data, sfreq, args)
    write_clean_payload(output_path, payload, cleaned, mean, std, args, split)


def process_split(split, source_dir, output_dir, mean, std, args):
    paths = sorted(source_dir.glob("*.pkl"))
    output_dir.mkdir(parents=True, exist_ok=True)
    pending_paths = []
    for source_path in paths:
        output_path = output_dir / source_path.name
        if output_path.exists():
            if args.resume:
                continue
            if not args.overwrite:
                raise FileExistsError(
                    "{} already exists; rerun with --resume or --overwrite"
                    .format(output_path)
                )
        pending_paths.append(source_path)
    jobs = [
        (source_path, output_dir / source_path.name)
        for source_path in pending_paths
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_one,
                source_path,
                output_path,
                mean,
                std,
                args,
                split,
            )
            for source_path, output_path in jobs
        ]
        for index, future in enumerate(futures, start=1):
            future.result()
            if index % 100 == 0 or index == len(pending_paths):
                print(
                    "{}: {}/{} pending ({} total)".format(
                        split, index, len(pending_paths), len(paths)
                    ),
                    flush=True,
                )
    return len(paths)


def main():
    args = parse_args()
    validate_args(args)
    source_dirs = {
        split: args.source_root / directory
        for split, directory in SPLIT_DIRS.items()
    }
    for path in source_dirs.values():
        if not path.is_dir():
            raise FileNotFoundError("missing source EEG directory: {}".format(path))

    train_paths = sorted(source_dirs["train"].glob("*.pkl"))
    if not train_paths:
        raise RuntimeError("no training EEG pkl files found")
    if args.reuse_statistics:
        mean, std, sample_count = load_saved_statistics(args.output_root)
        print("statistics: reused {}".format(statistics_path(args.output_root)), flush=True)
    else:
        mean, std, sample_count = compute_train_statistics(train_paths, args)
        save_statistics(statistics_path(args.output_root), mean, std, sample_count)
    counts = {}
    for split, directory in SPLIT_DIRS.items():
        if split not in args.splits:
            counts[split] = len(list((args.output_root / directory).glob("*.pkl")))
            continue
        counts[split] = process_split(
            split,
            source_dirs[split],
            args.output_root / directory,
            mean,
            std,
            args,
        )
    manifest = {
        "version": "eeget-rsod-clean-v1",
        "source_root": str(args.source_root),
        "counts": counts,
        "train_channel_sample_count": sample_count,
        "filter": {
            "detrend": "linear",
            "notch_hz": [50.0, 100.0],
            "notch_q": args.notch_q,
            "bandpass_hz": [args.lowcut, args.highcut],
            "butter_order": args.filter_order,
            "reference": "common_average_reference",
        },
        "normalization": {
            "fit_split": "train",
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
        },
        "usage": "Pass --eeg_root <output_root> and omit --eeg_normalize.",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("EEG_CLEAN_COMPLETE={}".format(args.output_root), flush=True)


if __name__ == "__main__":
    main()
