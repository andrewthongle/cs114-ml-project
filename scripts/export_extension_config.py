"""Generate the SafeView demo configuration from a verified HF bundle.

No network calls, tokens, weights deserialization, or sibling-repository edits.
Apply docs/safeview-cs114-demo.patch and copy the generated file after deploying
and testing the actual Space URL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from publish_hf import verify_prepared_bundle


INTERFACE = """export interface Cs114DemoConfig {
  spaceUrl: string;
  modelUrl: string;
  modelName: string;
  modelRevision: string;
  policyVersion: string;
  threshold: number;
  forceVietnamese: boolean;
  timeoutMs?: number;
}
"""


def export_config(bundle, output, space_url, model_repo, *, force_vietnamese=True, timeout_ms=60000):
    bundle, output = Path(bundle), Path(output)
    manifest = verify_prepared_bundle(bundle)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite extension config: {output}")
    url = urlsplit(space_url)
    if (url.scheme != "https" or not url.hostname or url.username or url.password
            or url.path not in {"", "/"} or url.query or url.fragment):
        raise ValueError("space_url must be an HTTPS origin without credentials, path or query")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_repo):
        raise ValueError("model_repo must be owner/repository")
    if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or not 1000 <= timeout_ms <= 300000:
        raise ValueError("timeout_ms must be an integer from 1000 to 300000")
    metadata = json.loads((bundle / "model/metadata.json").read_text())
    policy = json.loads((bundle / "model/decision_policy.json").read_text())
    if policy.get("model_revision") != manifest["model_revision"]:
        raise ValueError("Policy and bundle model revisions differ")
    names = {"bamibert": "BamiBERT · ViHSD", "phobert": "PhoBERT · ViHSD",
             "svm": "TF-IDF + Linear SVM", "logistic_regression": "TF-IDF + Logistic Regression",
             "complement_nb": "TF-IDF + Complement Naive Bayes"}
    config = {
        "spaceUrl": f"https://{url.netloc}", "modelUrl": f"https://huggingface.co/{model_repo}",
        "modelName": names.get(metadata.get("family"), "SafeView · ViHSD"),
        "modelRevision": policy["model_revision"], "policyVersion": policy["policy_version"],
        "threshold": policy["threshold"], "forceVietnamese": bool(force_vietnamese), "timeoutMs": timeout_ms,
    }
    content = ("// Generated from an evaluated release. Contains no credentials.\n" + INTERFACE
               + "\nexport const CS114_DEMO: Cs114DemoConfig | null = "
               + json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + ";\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--space-url", required=True)
    parser.add_argument("--model-repo", required=True)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--automatic-routing", action="store_true")
    args = parser.parse_args()
    print(export_config(args.bundle_dir, args.output, args.space_url, args.model_repo,
                        force_vietnamese=not args.automatic_routing, timeout_ms=args.timeout_ms))


if __name__ == "__main__":
    main()
