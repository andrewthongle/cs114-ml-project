"""Optional Transformer backend; importing this module does not import PyTorch.

Both training and serving use the same persisted preprocessing and tokenizer.
Hugging Face weights are downloaded only when the user starts training. Frozen
releases load locally with remote code disabled.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

from .policy import LABEL_NAMES
from .provenance import read_json, sha256, write_json


TRANSFORMER_FAMILIES = {"phobert", "bamibert"}
ID2LABEL = dict(enumerate(LABEL_NAMES))
LABEL2ID = {name: index for index, name in ID2LABEL.items()}


def _dependencies():
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError("Install safeview-ml[transformers] to train or serve PhoBERT/BamiBERT") from exc
    return torch, transformers


def prepare_texts(texts, preprocessing):
    """NFC and whitespace normalization; word segmentation only for PhoBERT."""
    if preprocessing not in {"raw", "pyvi"}:
        raise ValueError("Transformer preprocessing must be 'raw' or 'pyvi'")
    values = list(texts)
    if not all(isinstance(text, str) for text in values):
        raise ValueError("Transformer inputs must be strings")
    values = [" ".join(unicodedata.normalize("NFC", text).split()) for text in values]
    if preprocessing == "pyvi":
        try:
            from pyvi import ViTokenizer
        except ImportError as exc:
            raise ImportError("PhoBERT requires pyvi for the frozen word-segmentation pipeline") from exc
        values = [ViTokenizer.tokenize(text) for text in values]
    return values


def resolve_base_revision(model_id, revision="main"):
    """Pin floating Hub refs before loading either weights or tokenizer."""
    if re.fullmatch(r"[a-fA-F0-9]{40}", revision or ""):
        return revision.lower()
    from huggingface_hub import HfApi
    resolved = HfApi().model_info(model_id, revision=revision).sha
    if not re.fullmatch(r"[a-fA-F0-9]{40}", resolved or ""):
        raise ValueError("Could not resolve the base model to an immutable Hub commit")
    return resolved.lower()


class TransformerTextClassifier:
    """Raw text classifier with sklearn-compatible, canonical probabilities."""

    classes_ = np.asarray([0, 1, 2])

    def __init__(self, model, tokenizer, config, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = dict(config)
        self.device = str(device)
        if config.get("preprocessing") not in {"raw", "pyvi"}:
            raise ValueError("Missing or unsupported frozen Transformer preprocessing")
        if int(config.get("max_length", 0)) < 2 or int(config.get("inference_batch_size", 0)) < 1:
            raise ValueError("Invalid Transformer max_length or inference_batch_size")
        mapping = {int(index): name for index, name in model.config.id2label.items()}
        if mapping != ID2LABEL or model.config.num_labels != 3:
            raise ValueError("Transformer head must use 0=CLEAN, 1=OFFENSIVE, 2=HATE")
        self.to(device)

    def to(self, device):
        self.device = str(device)
        self.model.to(device)
        self.model.eval()
        return self

    @classmethod
    def load(cls, release_dir, device="cpu"):
        _, transformers = _dependencies()
        release = Path(release_dir)
        config = read_json(release / "transformer_config.json")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(release), local_files_only=True, trust_remote_code=False,
            use_fast=config.get("use_fast", True),
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            str(release), local_files_only=True, trust_remote_code=False, use_safetensors=True,
        )
        return cls(model, tokenizer, config, device)

    def save(self, release_dir):
        release = Path(release_dir)
        release.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(release), safe_serialization=True)
        self.tokenizer.save_pretrained(str(release))
        write_json(release / "transformer_config.json", self.config)

    def predict_proba(self, texts):
        torch, _ = _dependencies()
        values = prepare_texts(texts, self.config["preprocessing"])
        if not values:
            return np.empty((0, 3), dtype=np.float64)
        outputs = []
        batch_size = int(self.config["inference_batch_size"])
        self.model.eval()
        with torch.inference_mode():
            for start in range(0, len(values), batch_size):
                tokens = self.tokenizer(
                    values[start:start + batch_size], padding=True, truncation=True,
                    max_length=int(self.config["max_length"]), return_tensors="pt",
                )
                tokens = {name: value.to(self.device) for name, value in tokens.items()}
                logits = self.model(**tokens).logits
                # FP32 logits + FP64 softmax avoid half-precision rounding in policies.
                probabilities = torch.softmax(logits.double(), dim=-1).cpu().numpy()
                outputs.append(probabilities)
        return np.concatenate(outputs, axis=0)

    def predict(self, texts):
        return self.predict_proba(texts).argmax(axis=1)


class _EncodedDataset:
    def __init__(self, tokenizer, texts, labels, max_length):
        self.encoded = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        self.labels = [int(label) for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: values[index] for key, values in self.encoded.items()},
                "labels": self.labels[index]}


def _loss_weighting(labels, strategy=None):
    """Describe a loss using canonical counts from the fit (training) labels only."""
    if strategy is not None and strategy != "balanced":
        raise ValueError("Transformer class_weight must be null or 'balanced'")
    values = np.asarray(labels)
    if values.ndim != 1 or not len(values) or not np.isin(values, [0, 1, 2]).all():
        raise ValueError("Transformer training labels must be a nonempty vector of 0, 1, 2")
    counts = np.bincount(values.astype(np.int64), minlength=len(LABEL_NAMES))
    if strategy == "balanced" and np.any(counts == 0):
        raise ValueError("Balanced Transformer loss requires every class in the training labels")
    weights = len(values) / (len(LABEL_NAMES) * counts) if strategy == "balanced" else None
    return {
        "strategy": strategy or "none", "loss": "cross_entropy",
        "label_order": list(LABEL_NAMES), "training_class_counts": counts.tolist(),
        "class_weights": weights.tolist() if weights is not None else None,
        "weight_source": "train_labels" if strategy == "balanced" else None,
        "formula": "n_train / (n_classes * class_count)" if strategy == "balanced" else None,
        "reduction": "mean",
    }


def _make_trainer(torch, transformers, loss_weighting, **kwargs):
    """Keep the stock Trainer for unweighted runs and optional imports lazy."""
    if loss_weighting["class_weights"] is None:
        return transformers.Trainer(**kwargs)

    class WeightedCrossEntropyTrainer(transformers.Trainer):
        def __init__(self, **trainer_kwargs):
            super().__init__(**trainer_kwargs)
            self.class_weights = torch.tensor(loss_weighting["class_weights"], dtype=torch.float32)
            # Our batch mean does not consume num_items_in_batch. Trainer must
            # therefore retain its usual gradient-accumulation normalization.
            self.model_accepts_loss_kwargs = False

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            outputs = model(**{key: value for key, value in inputs.items() if key != "labels"})
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, len(LABEL_NAMES)), labels.reshape(-1),
                weight=self.class_weights.to(logits.device), reduction="mean",
            )
            return (loss, outputs) if return_outputs else loss

    return WeightedCrossEntropyTrainer(**kwargs)


class TransformerFineTuner:
    """One guarded full-fine-tuning trial, compatible with training._fit."""

    def __init__(self, settings, validation_texts, validation_labels, output_dir,
                 seed, guard, resume=False):
        self.settings = dict(settings)
        self.validation_texts = validation_texts
        self.validation_labels = validation_labels
        self.output_dir = Path(output_dir)
        self.seed = int(seed)
        self.guard = guard
        self.resume = resume

    def fit(self, texts, labels):
        loss_weighting = _loss_weighting(labels, self.settings.get("class_weight"))
        torch, transformers = _dependencies()
        from sklearn.metrics import f1_score
        from transformers.trainer_utils import get_last_checkpoint

        settings = self.settings
        model_id = settings["model_id"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.output_dir / "checkpoint_manifest.json"
        requested = {"settings": settings, "seed": self.seed, "guard": self.guard}
        if manifest_path.exists():
            previous = read_json(manifest_path)
            if not self.resume:
                raise FileExistsError("Checkpoint trial already exists; use resume=True")
            if previous["requested"] != requested:
                raise ValueError("Checkpoint config or dataset guard changed; refusing to resume")
            if "loss_weighting" not in previous and loss_weighting["strategy"] != "none":
                raise ValueError("Checkpoint has no loss-weighting metadata; start a new weighted trial")
            if previous.get("loss_weighting", loss_weighting) != loss_weighting:
                raise ValueError("Checkpoint training-label weights changed; refusing to resume")
            revision = previous["base_revision"]
        else:
            revision = resolve_base_revision(model_id, settings.get("revision", "main"))
            write_json(manifest_path, {"requested": requested, "base_revision": revision,
                                       "loss_weighting": loss_weighting})

        transformers.set_seed(self.seed)
        preprocessing = settings["preprocessing"]
        max_length = int(settings.get("max_length", 128))
        use_fast = bool(settings.get("use_fast", preprocessing != "pyvi"))
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, revision=revision, use_fast=use_fast, trust_remote_code=False,
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id, revision=revision, num_labels=3, id2label=ID2LABEL,
            label2id=LABEL2ID, trust_remote_code=False,
        )
        # All backbone parameters are trained; both models receive a new 3-way head.
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        model.config.problem_type = "single_label_classification"
        maximum = int(getattr(model.config, "max_position_embeddings", max_length + 2))
        if max_length + 2 > maximum:
            raise ValueError(f"max_length={max_length} exceeds the model position budget {maximum - 2}")
        train = _EncodedDataset(tokenizer, prepare_texts(texts, preprocessing), labels, max_length)
        validation = _EncodedDataset(tokenizer, prepare_texts(self.validation_texts, preprocessing),
                                     self.validation_labels, max_length)
        use_cuda = torch.cuda.is_available() and settings.get("device", "auto") != "cpu"
        if settings.get("device", "auto") not in {"auto", "cpu", "cuda"}:
            raise ValueError("Training device must be auto, cpu or cuda")
        if settings.get("device") == "cuda" and not use_cuda:
            raise ValueError("CUDA requested but unavailable; select a Colab GPU runtime")
        use_fp16 = bool(settings.get("fp16", True) and use_cuda)
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        arguments = transformers.TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=float(settings.get("epochs", 3)),
            learning_rate=float(settings.get("learning_rate", 2e-5)),
            per_device_train_batch_size=int(settings.get("batch_size", 8)),
            per_device_eval_batch_size=int(settings.get("eval_batch_size", 16)),
            gradient_accumulation_steps=int(settings.get("gradient_accumulation_steps", 2)),
            weight_decay=float(settings.get("weight_decay", 0.01)),
            warmup_ratio=float(settings.get("warmup_ratio", 0.1)),
            eval_strategy="epoch", save_strategy="epoch", logging_strategy="steps",
            logging_steps=int(settings.get("logging_steps", 50)),
            load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
            save_total_limit=2, save_safetensors=True,
            seed=self.seed, data_seed=self.seed, fp16=use_fp16, use_cpu=not use_cuda,
            report_to="none", dataloader_num_workers=0, dataloader_pin_memory=use_cuda,
            optim="adamw_torch", gradient_checkpointing=bool(settings.get("gradient_checkpointing", False)),
        )

        def metrics(prediction):
            logits = prediction.predictions
            if isinstance(logits, tuple):
                logits = logits[0]
            return {"macro_f1": float(f1_score(prediction.label_ids, np.argmax(logits, axis=1),
                                              labels=[0, 1, 2], average="macro", zero_division=0))}

        latest = get_last_checkpoint(str(self.output_dir)) if self.resume else None
        if latest and (Path(latest) / "scaler.pt").exists() and not use_fp16:
            raise ValueError("This checkpoint uses CUDA FP16; restore a GPU runtime to resume training")
        trainer = _make_trainer(torch, transformers, loss_weighting,
            model=model, args=arguments, train_dataset=train, eval_dataset=validation,
            processing_class=tokenizer,
            data_collator=transformers.DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if use_fp16 else None),
            compute_metrics=metrics,
        )
        trainer.train(resume_from_checkpoint=latest)
        self.history_ = trainer.state.log_history
        self.training_metadata_ = {
            "base_model": model_id, "base_revision": revision,
            "training_device": "cuda" if use_cuda else "cpu", "fp16": use_fp16,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_macro_f1": trainer.state.best_metric,
            "resumed_from_checkpoint": latest,
            "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if use_cuda else None,
            "loss_weighting": loss_weighting,
        }
        frozen = {
            "schema_version": 1, "base_model": model_id, "base_revision": revision,
            "max_length": max_length, "preprocessing": preprocessing,
            "unicode_normalization": "NFC", "whitespace_normalization": "collapse", "use_fast": use_fast,
            "inference_batch_size": int(settings.get("eval_batch_size", 16)),
            "label_mapping": {str(index): name for index, name in ID2LABEL.items()},
            "loss_weighting": loss_weighting,
        }
        self.estimator_ = TransformerTextClassifier(trainer.model, tokenizer, frozen,
                                                   device="cuda" if use_cuda else "cpu")
        # Release optimizer state before another trial is instantiated.
        del trainer
        return self


def artifact_hashes(directory):
    """Hash every local release asset, including tokenizer files and the policy."""
    directory = Path(directory)
    return {str(path.relative_to(directory)): sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "metadata.json"}
