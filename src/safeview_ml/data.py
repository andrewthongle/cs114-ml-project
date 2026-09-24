"""ViHSD loading, split audits and aggregate-only EDA exports.

No loader shuffles, drops, deduplicates or relabels official samples. Raw files
remain untouched. Audits must be reviewed before interpreting benchmark scores.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import unicodedata

import numpy as np
import pandas as pd

from .preprocessing import PREPROCESSING_VERSION, normalize_text

LABEL_NAMES = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
SPLIT_NAMES = ("train", "validation", "test")
_SPLIT_ALIASES = {"train": ("train",), "validation": ("validation", "dev", "valid", "val"), "test": ("test",)}
_TEXT_COLUMNS = ("text", "free_text", "sentence", "comment", "content")
_LABEL_COLUMNS = ("label", "labels", "hs_label_id", "label_id", "category")
_ID_COLUMNS = ("sample_id", "id", "index")


@dataclass
class DatasetBundle:
    splits: dict[str, pd.DataFrame]
    manifest: dict


def _label_id(value) -> int:
    if isinstance(value, str):
        name = value.strip().upper()
        if name in LABEL_NAMES.values():
            return next(key for key, label in LABEL_NAMES.items() if label == name)
        if name in {"0", "1", "2"}:
            return int(name)
    elif not isinstance(value, (bool, np.bool_)) and pd.notna(value):
        if isinstance(value, (int, float, np.integer, np.floating)) and value in LABEL_NAMES:
            return int(value)
    raise ValueError("Invalid label: expected 0/CLEAN, 1/OFFENSIVE, or 2/HATE.")


def _canonical_frame(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    text_column = next((column for column in _TEXT_COLUMNS if column in frame), None)
    label_column = next((column for column in _LABEL_COLUMNS if column in frame), None)
    id_column = next((column for column in _ID_COLUMNS if column in frame), None)
    if text_column is None or label_column is None:
        raise ValueError(f"{split}: expected a text column {_TEXT_COLUMNS} and label column {_LABEL_COLUMNS}.")
    if len(frame) == 0:
        raise ValueError(f"{split}: split is empty.")
    if id_column:
        ids = frame[id_column]
        if ids.isna().any() or ids.astype(str).str.strip().eq("").any():
            raise ValueError(f"{split}: sample IDs must be non-null and non-empty.")
        ids = ids.astype(str)
        if ids.duplicated().any():
            raise ValueError(f"{split}: duplicate sample IDs; resolve IDs without dropping samples.")
    else:
        ids = pd.Series([f"{split}:{index:08d}" for index in range(len(frame))], index=frame.index)
    try:
        labels = frame[label_column].map(_label_id)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{split}: {exc}") from exc
    # Null and empty text are retained so the audit can account for every row.
    invalid_text = frame[text_column].notna() & ~frame[text_column].map(lambda value: isinstance(value, str))
    if invalid_text.any():
        raise ValueError(f"{split}: {int(invalid_text.sum())} non-string text values.")
    return pd.DataFrame({"sample_id": ids.to_numpy(), "text": frame[text_column].to_numpy(), "label": labels.to_numpy(dtype=int)})


def dataset_fingerprint(splits: dict[str, pd.DataFrame]) -> str:
    """Content fingerprint includes ordered IDs, raw text, labels and split names."""
    digest = hashlib.sha256()
    for split in SPLIT_NAMES:
        if split not in splits:
            continue
        for row in splits[split][["sample_id", "text", "label"]].itertuples(index=False, name=None):
            record = [split, str(row[0]), None if pd.isna(row[1]) else str(row[1]), int(row[2])]
            digest.update(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def bundle_from_frames(
    frames: dict[str, pd.DataFrame], *, source: str | None = None,
    revision: str | None = None, files: list[dict] | None = None,
    source_metadata: dict | None = None,
) -> DatasetBundle:
    """Apply the same schema and identity rules to local and remote data."""
    metadata = source_metadata or {}
    if set(frames) != set(SPLIT_NAMES):
        raise ValueError("Expected exactly the original train, validation and test splits.")
    splits = {split: _canonical_frame(frames[split], split) for split in SPLIT_NAMES}
    manifest = {
        "schema_version": 1,
        "dataset": metadata.get("dataset", "synthetic fixture" if metadata.get("synthetic") else "ViHSD"),
        "synthetic": bool(metadata.get("synthetic", False)),
        "source": source or "local files; provenance must be supplied by the researcher",
        "revision": revision,
        "fingerprint": dataset_fingerprint(splits),
        "label_mapping": {str(key): value for key, value in LABEL_NAMES.items()},
        "split_counts": {split: len(frame) for split, frame in splits.items()},
        "label_counts": {split: {name: int((frame.label == label).sum()) for label, name in LABEL_NAMES.items()} for split, frame in splits.items()},
        "files": files or [],
        "official_splits_preserved": True,
        "preprocessing_version": PREPROCESSING_VERSION,
        "provenance_note": "Original split assignment retained; dataset identity and license require verification against the supplied source.",
    }
    return DatasetBundle(splits=splits, manifest=manifest)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        # Literal strings such as "NA" must remain text, not be turned into NaN.
        return pd.read_csv(path, keep_default_na=False, dtype=str)
    if path.suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    if path.suffix == ".json":
        return pd.read_json(path, dtype=False, convert_dates=False)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset file format: {path.suffix}")


def _find_split(root: Path, split: str) -> tuple[pd.DataFrame, list[Path]]:
    candidates: list[list[Path]] = []
    author_pairs: list[tuple[Path, Path]] = []
    for alias in _SPLIT_ALIASES[split]:
        for directory in (root, root / "data"):
            for extension in ("csv", "jsonl", "ndjson", "json", "parquet"):
                direct = directory / f"{alias}.{extension}"
                if direct.is_file():
                    candidates.append([direct])
                shards = sorted(directory.glob(f"{alias}-*.{extension}"))
                if shards:
                    candidates.append(shards)
        # Formats distributed in the author's repository.
        for directory in (root / alias, root / "data" / alias):
            for text_name in ("sents.txt", "sentences.txt", "text.txt"):
                text_path, label_path = directory / text_name, directory / "labels.txt"
                if text_path.is_file() and label_path.is_file():
                    author_pairs.append((text_path, label_path))
        flat_text, flat_label = root / f"{alias}.txt", root / f"{alias}_labels.txt"
        if flat_text.is_file() and flat_label.is_file():
            author_pairs.append((flat_text, flat_label))
    if len(candidates) + len(author_pairs) > 1:
        raise ValueError(f"Multiple source representations found for {split}; use a directory with one official copy.")
    if candidates:
        paths = candidates[0]
        return pd.concat([_read_table(path) for path in paths], ignore_index=True), paths
    if author_pairs:
        text_path, label_path = author_pairs[0]
        texts = text_path.read_text(encoding="utf-8-sig").splitlines()
        labels = label_path.read_text(encoding="utf-8-sig").splitlines()
        if len(texts) != len(labels):
            raise ValueError(f"{split}: text and label files have different row counts.")
        return pd.DataFrame({"text": texts, "label": labels}), [text_path, label_path]
    raise FileNotFoundError(
        f"Missing official {split} split in {root}. Provide train/validation(or dev)/test "
        "CSV, JSONL or parquet files, or <split>/sents.txt + labels.txt. "
        "No synthetic or alternative dataset is substituted."
    )


def load_local_dataset(data_dir: str | Path, source: str | None = None, revision: str | None = None) -> DatasetBundle:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    source_metadata = {}
    if (root / "source.json").is_file():
        source_metadata = json.loads((root / "source.json").read_text(encoding="utf-8"))
        if not isinstance(source_metadata, dict):
            raise ValueError("source.json must be a JSON object with source/revision metadata.")
    source = source if source is not None else source_metadata.get("source")
    revision = revision if revision is not None else source_metadata.get("revision")
    frames, files = {}, []
    for split in SPLIT_NAMES:
        frame, paths = _find_split(root, split)
        frames[split] = frame
        files.extend(paths)
    return bundle_from_frames(
        frames, source=source, revision=revision, source_metadata=source_metadata,
        files=[{"path": str(path.relative_to(root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files],
    )


def validate_training_data(bundle: DatasetBundle, *, empty_text_policy: str = "error") -> dict:
    """Validate without changing rows; optionally retain zero-content strings.

    ``keep`` lets the existing model preprocessing normalize blank strings to
    ``''``. Null/non-string values remain errors under either policy.
    """
    if empty_text_policy not in ("error", "keep"):
        raise ValueError("empty_text_policy must be 'error' or 'keep'.")
    empty_counts = {}
    for split in SPLIT_NAMES:
        if split not in bundle.splits:
            raise ValueError(f"Missing {split} split.")
        frame = bundle.splits[split]
        non_string = frame.text.map(lambda text: not isinstance(text, str))
        if non_string.any():
            raise ValueError(f"{split}: {int(non_string.sum())} null/non-string texts. Review the source data; no rows were dropped.")
        empty_counts[split] = int(frame.text.map(lambda text: not normalize_text(text)).sum())
        if empty_counts[split] and empty_text_policy == "error":
            raise ValueError(
                f"{split}: {empty_counts[split]} empty texts. Review the audit and set "
                "empty_text_policy='keep' to retain blank strings with their original labels; "
                "no rows were dropped."
            )
    if set(bundle.splits["train"].label) != set(LABEL_NAMES):
        raise ValueError("Training split must contain all three labels CLEAN/OFFENSIVE/HATE.")
    return {"empty_text_policy": empty_text_policy, "empty_text_counts": empty_counts}


def save_manifest(bundle: DatasetBundle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _normalized_rows(bundle: DatasetBundle) -> pd.DataFrame:
    rows = []
    for split, frame in bundle.splits.items():
        for text, label in frame[["text", "label"]].itertuples(index=False, name=None):
            if isinstance(text, str) and normalize_text(text):
                rows.append((split, normalize_text(text), int(label)))
    return pd.DataFrame(rows, columns=["split", "normalized", "label"])


def _near_duplicate_audit(rows: pd.DataFrame, limit: int = 1500, threshold: float = 0.90) -> dict:
    """Bounded screening, never advertised as exhaustive deduplication."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    unique = rows.drop_duplicates("normalized").reset_index(drop=True)
    if len(unique) > limit:
        unique = unique.iloc[np.linspace(0, len(unique) - 1, limit, dtype=int)].reset_index(drop=True)
    result = {
        "method": "character TF-IDF cosine; up to 6 neighbors per distinct normalized text",
        "similarity_threshold": threshold,
        "sampled_distinct_texts": len(unique),
        "max_texts": limit,
        "max_characters_per_text": 2000,
        "candidate_pairs": 0,
        "cross_split_candidate_pairs": 0,
        "conflicting_label_candidate_pairs": 0,
        "exhaustive": False,
        "note": "Deterministic bounded screening; candidate counts are lower bounds, not confirmed duplicates. Exact duplicates are audited separately.",
    }
    if len(unique) < 2:
        return result
    texts = unique.normalized.str.slice(stop=2000)
    try:
        matrix = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False, max_features=20000).fit_transform(texts)
    except ValueError:  # A dataset consisting only of strings shorter than 3 chars.
        result["note"] += " No usable character n-grams in the screened texts."
        return result
    distances, neighbors = NearestNeighbors(n_neighbors=min(6, len(unique)), metric="cosine", algorithm="brute").fit(matrix).kneighbors(matrix)
    pairs = set()
    for first, (scores, indices) in enumerate(zip(distances, neighbors)):
        for distance, second in zip(scores, indices):
            if first != second and 1.0 - float(distance) >= threshold:
                pairs.add(tuple(sorted((first, int(second)))))
    result["candidate_pairs"] = len(pairs)
    result["cross_split_candidate_pairs"] = int(sum(unique.iloc[first].split != unique.iloc[second].split for first, second in pairs))
    result["conflicting_label_candidate_pairs"] = int(sum(unique.iloc[first].label != unique.iloc[second].label for first, second in pairs))
    return result


def audit_dataset(bundle: DatasetBundle) -> dict:
    rows = _normalized_rows(bundle)
    summary = {"fingerprint": bundle.manifest["fingerprint"], "normalization": "NFC and whitespace; case and punctuation preserved", "splits": {}}
    for split, frame in bundle.splits.items():
        normalized = rows[rows.split == split]
        grouped = normalized.groupby("normalized")
        counts = grouped.size()
        summary["splits"][split] = {
            "rows": len(frame),
            "null_texts": int(frame.text.isna().sum()),
            "empty_texts": int(frame.text.map(lambda value: isinstance(value, str) and not normalize_text(value)).sum()),
            "duplicate_raw_rows_beyond_first": int(frame[["text", "label"]].duplicated().sum()),
            "duplicate_normalized_texts_beyond_first": int((counts - 1).clip(lower=0).sum()),
            "duplicate_normalized_groups": int((counts > 1).sum()),
            "conflicting_label_groups": int((grouped.label.nunique() > 1).sum()),
            "label_counts": {name: int((frame.label == label).sum()) for label, name in LABEL_NAMES.items()},
        }
    grouped = rows.groupby("normalized")
    split_counts = grouped.split.nunique()
    cross_texts = split_counts[split_counts > 1].index
    summary["cross_split_normalized_duplicates"] = {
        "groups": len(cross_texts),
        "rows_involved": int(rows.normalized.isin(cross_texts).sum()),
        "conflicting_label_groups": int((grouped.label.nunique().reindex(cross_texts) > 1).sum()),
        "split_pairs": {f"{first}:{second}": len(set(rows.loc[rows.split == first, "normalized"]) & set(rows.loc[rows.split == second, "normalized"])) for index, first in enumerate(SPLIT_NAMES) for second in SPLIT_NAMES[index + 1:]},
    }
    summary["near_duplicates"] = _near_duplicate_audit(rows)
    summary["policy"] = "Report leakage on original official splits; do not silently remove or move rows. Deduplicated sensitivity experiments need separate manifests and results."
    return summary


_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\U0001F1E6-\U0001F1FF]")
_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_REPEATED = re.compile(r"(.)\1{2,}", flags=re.DOTALL)
_VI_DIACRITICS = set("ăâđêôơưĂÂĐÊÔƠƯ")


def _text_statistics(text) -> dict:
    if not isinstance(text, str):
        text = ""
    text = normalize_text(text)
    has_diacritics = any(char in _VI_DIACRITICS or unicodedata.combining(char) for char in unicodedata.normalize("NFD", text))
    return {
        "characters": len(text),
        "whitespace_tokens": len(text.split()),
        "has_emoji": bool(_EMOJI.search(text)),
        "has_url": bool(_URL.search(text)),
        "has_repeated_characters": bool(_REPEATED.search(text)),
        "latin_text_without_diacritics": bool(re.search("[A-Za-z]", text)) and not has_diacritics,
    }


def export_eda(bundle: DatasetBundle, output_dir: str | Path) -> dict[str, str]:
    """Export reproducible aggregate tables/figures without raw sample excerpts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}

    def write_csv(name: str, frame: pd.DataFrame):
        path = output / f"{name}.csv"
        frame.to_csv(path, index=False)
        artifacts[name] = str(path)

    audit_path = output / "audit.json"
    audit_path.write_text(json.dumps(audit_dataset(bundle), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifacts["audit"] = str(audit_path)
    artifacts["manifest"] = str(save_manifest(bundle, output / "manifest.json"))
    stats_records = []
    for split, frame in bundle.splits.items():
        for text, label in frame[["text", "label"]].itertuples(index=False, name=None):
            stats_records.append({"split": split, "label": LABEL_NAMES[int(label)], **_text_statistics(text)})
    stats = pd.DataFrame(stats_records)
    label_counts = stats.groupby(["split", "label"], sort=False).size().rename("count").reset_index()
    write_csv("label_distribution", label_counts)
    length_rows, signal_rows, hist_rows = [], [], []
    bins = [0, 10, 25, 50, 100, 200, 400, 800, 1600, float("inf")]
    for (split, label), group in stats.groupby(["split", "label"]):
        for metric in ("characters", "whitespace_tokens"):
            values = group[metric]
            length_rows.append({"split": split, "label": label, "metric": metric, "count": len(values), "mean": float(values.mean()), "std": float(values.std(ddof=0)), "min": int(values.min()), "p25": float(values.quantile(.25)), "median": float(values.median()), "p75": float(values.quantile(.75)), "p95": float(values.quantile(.95)), "max": int(values.max())})
        counts, _ = np.histogram(group.characters, bins=bins)
        for index, count in enumerate(counts):
            hist_rows.append({"split": split, "label": label, "min_characters": bins[index], "max_characters_exclusive": str(bins[index + 1]), "count": int(count)})
        for signal in ("has_emoji", "has_url", "has_repeated_characters", "latin_text_without_diacritics"):
            signal_rows.append({"split": split, "label": label, "signal": signal, "count": int(group[signal].sum()), "rate": float(group[signal].mean())})
    write_csv("length_summary", pd.DataFrame(length_rows))
    write_csv("length_histogram", pd.DataFrame(hist_rows))
    write_csv("text_signals", pd.DataFrame(signal_rows))

    # These are aggregate vocabulary statistics from train only; no sample rows.
    token_rows = []
    complete_texts = {normalize_text(text) for frame in bundle.splits.values() for text in frame.text if isinstance(text, str)}
    for label, name in LABEL_NAMES.items():
        for size in (1, 2):
            counts: Counter = Counter()
            documents: Counter = Counter()
            for text in bundle.splits["train"].loc[lambda frame: frame.label == label, "text"]:
                if isinstance(text, str):
                    tokens = normalize_text(text).split()
                    ngrams = [" ".join(tokens[index:index + size]) for index in range(len(tokens) - size + 1)]
                    counts.update(ngrams)
                    documents.update(set(ngrams))
            # Suppress terms seen in only one document to reduce identifying data.
            eligible = [(token, count) for token, count in counts.items() if documents[token] >= 2 and token not in complete_texts and not _URL.search(token) and "@" not in token]
            for token, count in sorted(eligible, key=lambda item: (-item[1], item[0]))[:30]:
                token_rows.append({"label": name, "ngram_size": size, "token_ngram": token, "count": count, "document_count": documents[token]})
    write_csv("train_top_ngrams", pd.DataFrame(token_rows, columns=["label", "ngram_size", "token_ngram", "count", "document_count"]))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, size in enumerate((1, 2)):
        for column, label in enumerate(LABEL_NAMES.values()):
            ax = axes[row, column]
            selected = [record for record in token_rows if record["label"] == label and record["ngram_size"] == size][:10]
            if selected:
                ax.barh([record["token_ngram"] for record in selected][::-1], [record["count"] for record in selected][::-1])
            else:
                ax.text(.5, .5, "No terms meeting document frequency >= 2", ha="center", va="center", transform=ax.transAxes, wrap=True, fontsize=8)
            ax.set(title=f"{label}: token {size}-grams", xlabel="Train occurrences")
    fig.suptitle("Frequent whitespace token n-grams (train only)")
    fig.tight_layout()
    path = output / "train_top_ngrams.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    artifacts["train_top_ngrams_figure"] = str(path)

    signal_frame = pd.DataFrame(signal_rows)
    signal_pivot = signal_frame.pivot(index=["split", "label"], columns="signal", values="rate")
    fig, ax = plt.subplots(figsize=(9, 6))
    shown = ax.imshow(signal_pivot.to_numpy(), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(signal_pivot.columns)), [name.replace("_", " ") for name in signal_pivot.columns], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(signal_pivot.index)), [f"{split} / {label}" for split, label in signal_pivot.index])
    for row in range(len(signal_pivot)):
        for column in range(len(signal_pivot.columns)):
            value = float(signal_pivot.iloc[row, column])
            ax.text(column, row, f"{value:.0%}", ha="center", va="center", color="white" if value > .5 else "black")
    ax.set_title("Text signals by split and label (heuristics)")
    fig.colorbar(shown, ax=ax, label="Fraction of samples")
    fig.tight_layout()
    path = output / "text_signals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    artifacts["text_signals_figure"] = str(path)

    pivot = label_counts.pivot(index="split", columns="label", values="count").reindex(index=SPLIT_NAMES, columns=list(LABEL_NAMES.values())).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot.bar(ax=ax, rot=0)
    ax.set(title="Class distribution by official split", xlabel="Split", ylabel="Samples")
    fig.tight_layout()
    path = output / "label_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    artifacts["label_distribution_figure"] = str(path)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, split in zip(axes, SPLIT_NAMES):
        for label in LABEL_NAMES.values():
            lengths = stats.loc[(stats.split == split) & (stats.label == label), "characters"]
            if len(lengths):
                ax.hist(np.log1p(lengths), bins=25, alpha=.45, label=label, density=True)
        ax.set(title=split, xlabel="log(1 + characters)")
    axes[0].set_ylabel("Density")
    axes[-1].legend()
    fig.suptitle("Text length by label (NFC and whitespace normalization)")
    fig.tight_layout()
    path = output / "length_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    artifacts["length_distribution_figure"] = str(path)

    notes = {
        "raw_samples_exported": False,
        "ngrams_source": "train only; whitespace tokens, not Vietnamese word segmentation; document frequency >= 2; URL/mention terms and any n-gram equal to a complete sample excluded",
        "signals": "Emoji detection is a Unicode-range approximation; unaccented Latin text is a heuristic, not a language detector.",
        "python": platform.python_version(),
    }
    notes_path = output / "eda_notes.json"
    notes_path.write_text(json.dumps(notes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifacts["notes"] = str(notes_path)
    return artifacts
