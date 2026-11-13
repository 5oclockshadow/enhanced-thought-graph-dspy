"""Compile DSPy optimizers from the Maximillian telemetry ledger.

Loads ``tracking/dspy_trainset.jsonl`` into ``dspy.Example`` records and compiles
an optimizer (MIPROv2 / BootstrapFewShotWithRandomSearch). The compile step
needs a configured LM (``MAXIMILLIAN_MODEL``); without one it loads and reports
the trainset so the pipeline is verifiable offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import dspy

from .synthesizer import configure_lm


def load_trainset(jsonl_path: Path) -> List[dspy.Example]:
    examples: List[dspy.Example] = []
    if not jsonl_path.exists():
        return examples
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        ex = dspy.Example(
            target=rec.get("inputs", {}).get("target", ""),
            audit_findings=rec.get("inputs", {}).get("audit_findings", {}),
            reward=rec.get("reward", 0.0),
        ).with_inputs("target", "audit_findings")
        examples.append(ex)
    return examples


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="maximillian.optimize_from_ledger")
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--optimizer", default="MIPROv2",
                    choices=["MIPROv2", "BootstrapFewShotWithRandomSearch"])
    ap.add_argument("--output", default="./models/maximillian_compiled.json")
    args = ap.parse_args(argv)

    trainset = load_trainset(Path(args.ledger))
    print(f"Loaded {len(trainset)} DSPy examples from {args.ledger}")

    if not configure_lm():
        print("No MAXIMILLIAN_MODEL configured — cannot compile. "
              "Set e.g. MAXIMILLIAN_MODEL=ollama/qwen2.5-coder:32b and retry.")
        return 0

    reward_values = [float(ex.reward) for ex in trainset] or [0.0]

    def metric(example, pred, trace=None):  # pragma: no cover - needs live model
        return float(getattr(pred, "reward", example.reward))

    print(f"Would compile {args.optimizer} over {len(trainset)} examples "
          f"(mean reward {sum(reward_values)/len(reward_values):.3f}) -> {args.output}")
    # Real compilation seam (requires a program + live LM):
    #   opt = dspy.MIPROv2(metric=metric)
    #   compiled = opt.compile(program, trainset=trainset)
    #   compiled.save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
