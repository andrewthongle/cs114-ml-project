"""Explicit commands: EDA -> train & freeze -> test -> saved results."""
import argparse
import sys
from pathlib import Path

import yaml

from .data import load_local_dataset, export_eda
from .provenance import write_json
from .remote_data import load_github_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="SafeView ML research workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["eda", "train", "evaluate"]:
        p = sub.add_parser(name)
        p.add_argument("--data-source", choices=["github", "local"], help="Defaults to GitHub; an explicit --data-dir selects local")
        p.add_argument("--data-dir", help="Directory of local official splits; implies --data-source local")
        p.add_argument("--cache-dir", help="Optional GitHub ZIP cache, e.g. /content/.cache/safeview/vihsd; default is memory only")
        p.add_argument("--source", help="Provenance URL/name for local files")
        p.add_argument("--revision", help="GitHub branch/tag/SHA (default main), or local provenance revision")
        p.add_argument("--project-dir", default=".")
        if name == "eda":
            p.add_argument("--output-dir", default="results/eda")
        if name == "train":
            p.add_argument("--config", default="configs/experiments.yaml")
            p.add_argument("--run-id", required=True)
            p.add_argument("--synthetic", action="store_true", help="Marks software-only fixture runs, forbids publishing")
            p.add_argument("--resume", action="store_true", help="Resume the same run from saved Transformer checkpoints")
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
        data_source = args.data_source or ("local" if args.data_dir else "github")
        if data_source == "github":
            if args.data_dir or args.source:
                raise ValueError("--data-dir and --source describe local files; choose --data-source local to use them.")
            bundle = load_github_dataset(revision=args.revision or "main", cache_dir=args.cache_dir)
        else:
            if not args.data_dir:
                raise ValueError("--data-source local requires --data-dir.")
            if args.cache_dir:
                raise ValueError("--cache-dir applies only to --data-source github.")
            bundle = load_local_dataset(args.data_dir, source=args.source, revision=args.revision)
        project = Path(args.project_dir)
        if args.command == "eda":
            output = export_eda(bundle, args.output_dir)
            write_json(project / "data/manifest.json", bundle.manifest)
            print(f"EDA artifacts: {args.output_dir}")
        elif args.command == "train":
            from .training import train_experiment
            config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
            print(train_experiment(bundle, config, project, args.run_id, synthetic=args.synthetic, resume=args.resume))
        else:
            from .training import evaluate_locked
            print(evaluate_locked(bundle, args.run_dir, project))
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Cannot continue: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
