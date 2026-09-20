"""Explicit commands: EDA -> train & freeze -> test -> saved results."""
import argparse
import sys
from pathlib import Path

import yaml

from .data import load_local_dataset, export_eda
from .provenance import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="SafeView ML research workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["eda", "train", "evaluate"]:
        p = sub.add_parser(name)
        p.add_argument("--data-dir", default="data/raw/vihsd")
        p.add_argument("--source")
        p.add_argument("--revision")
        p.add_argument("--project-dir", default=".")
        if name == "eda":
            p.add_argument("--output-dir", default="results/eda")
        if name == "train":
            p.add_argument("--config", default="configs/experiments.yaml")
            p.add_argument("--run-id", required=True)
            p.add_argument("--synthetic", action="store_true", help="Marks software-only fixture runs, forbids publishing")
        if name == "evaluate":
            p.add_argument("--run-dir", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--release-dir", required=True)
    p.add_argument("text")
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            import json
            from .inference import ReleasePredictor
            print(json.dumps(ReleasePredictor(args.release_dir).predict(args.text), ensure_ascii=False, indent=2))
            return 0
        bundle = load_local_dataset(args.data_dir, source=args.source, revision=args.revision)
        project = Path(args.project_dir)
        if args.command == "eda":
            output = export_eda(bundle, args.output_dir)
            write_json(project / "data/manifest.json", bundle.manifest)
            print(f"EDA artifacts: {args.output_dir}")
        elif args.command == "train":
            from .training import train_experiment
            config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            print(train_experiment(bundle, config, project, args.run_id, synthetic=args.synthetic))
        else:
            from .training import evaluate_locked
            print(evaluate_locked(bundle, args.run_dir, project))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"Cannot continue: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
