"""Run three AR objectives with shared assets/data and compare matching evaluations."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .config import Config, validate_config
from .dataset import read_json, write_json


def variants(config: Config) -> dict[str, Config]:
    if not config.dataset.require_abc or config.train.cot != "full" or config.train.abc_regularizer is None:
        raise ValueError("Comparison requires require_abc=true, cot=full and an ABC regularizer pack")
    result = {}
    for name, cot, objective, fraction in (
        ("cot_off", "off", "abc+semantic", 0.0),
        ("abc_semantic", "full", "abc+semantic", config.train.abc_regularizer_fraction),
        ("abc_only", "full", "abc-only", 1.0),
    ):
        if name == "abc_semantic" and not 0 < fraction < 1:
            raise ValueError("Comparison requires 0 < abc_regularizer_fraction < 1")
        variant = replace(
            config,
            train=replace(
                config.train,
                cot=cot,
                objective=objective,
                abc_regularizer_fraction=fraction,
                output_directory=config.train.output_directory / name,
            ),
        )
        validate_config(variant)
        result[name] = variant
    return result


def compare_runs(directories: list[Path], output: Path) -> dict:
    if len(directories) < 2:
        raise ValueError("At least two runs are required")
    runs = []
    signatures = set()
    for directory in directories:
        contract = read_json(directory / "validation.json")
        signatures.add(contract["signature"])
        metrics = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines() if line.strip()]
        evaluations = {row["step"]: row for row in metrics if "validation_signature" in row}
        if any(row["validation_signature"] != contract["signature"] for row in evaluations.values()):
            raise ValueError("Metrics contain a different validation signature")
        runs.append((directory, evaluations))
    if len(signatures) != 1:
        raise ValueError("Validation conditions differ; refusing a misleading comparison")
    common = sorted(set.intersection(*(set(rows) for _, rows in runs)))
    if not common:
        raise ValueError("No common checkpoint steps with validation results")
    report = {
        "validation_signature": signatures.pop(),
        "common_steps": common,
        "runs": [
            {
                "directory": str(directory),
                "metrics": [
                    {key: value for key, value in rows[step].items() if key == "step" or key.endswith("_validation")}
                    for step in common
                ],
            }
            for directory, rows in runs
        ],
        "note": "Teacher-forced losses; generated music quality and NAR adaptation need listening tests.",
    }
    write_json(output, report)
    return report
