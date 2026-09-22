"""Exercise the three traditional baselines with invented class markers.

Transformer software checks use tiny offline encoders in the pytest suite.
Neither run produces ViHSD research results.
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import yaml

from safeview_ml.data import load_local_dataset, export_eda
from safeview_ml.inference import ReleasePredictor
from safeview_ml.provenance import read_json, write_json
from safeview_ml.training import train_experiment, evaluate_locked


def create_fixture(path):
    path = Path(path)
    path.mkdir(parents=True)
    markers = ["nhãn_sạch lời cảm ơn hữu ích", "nhãn_xúc_phạm ký hiệu nhóm hai", "nhãn_thù_ghét ký hiệu nhóm ba"]
    for split, size in [("train", 72), ("validation", 36), ("test", 36)]:
        rows = [{"sample_id": f"synthetic-{split}-{i}",
                 "text": f"Dữ liệu giả kiểm thử {markers[i % 3]} ví dụ {split} {i}",
                 "label": i % 3} for i in range(size)]
        pd.DataFrame(rows).to_csv(path / f"{split}.csv", index=False)
    write_json(path / "source.json", {"source": "synthetic-software-fixture-v1", "revision": "1", "synthetic": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", help="New isolated output directory; default creates a temp directory")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir).resolve() if args.output_dir else Path(tempfile.mkdtemp(prefix="safeview-smoke-"))
    create_fixture(output / "data/raw/synthetic")
    bundle = load_local_dataset(output / "data/raw/synthetic", source="synthetic-software-fixture-v1", revision="1")
    bundle.manifest["dataset"] = "SYNTHETIC_SOFTWARE_FIXTURE"
    bundle.manifest["synthetic"] = True
    export_eda(bundle, output / "results/eda")
    config = yaml.safe_load((project / "configs/smoke.yaml").read_text())
    run = train_experiment(bundle, config, output, run_id="smoke", synthetic=True)
    final = evaluate_locked(bundle, run, output)
    selection = read_json(run / "selection.json")
    artifact = output / next(c["artifact_dir"] for c in selection["candidates"] if c["family"] == selection["selected_family"])
    prediction = ReleasePredictor(artifact).predict("Dữ liệu giả kiểm thử nhãn_sạch lời cảm ơn hữu ích")
    assert np.isclose(sum(prediction["scores"].values()), 1)
    assert prediction["should_hide"] == (prediction["p_harm"] >= read_json(artifact / "decision_policy.json")["threshold"])
    write_json(output / "verification.json", {"status": "passed", "synthetic": True, "trained_families": list(config["models"]),
                                               "test_report": str(final), "serving_serialization_verified": True,
                                               "notice": "Software verification only. Not ViHSD metrics or evidence of language understanding."})
    print(json.dumps({"output_dir": str(output), "status": "passed", "synthetic": True}))


if __name__ == "__main__":
    main()
