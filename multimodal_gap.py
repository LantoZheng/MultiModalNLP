import argparse
import json
from collections import defaultdict
from pathlib import Path


FAMILIES = ("LLM", "VLM", "ALM")
REQUIRED_FIELDS = ("task_id", "model_family", "expected", "prediction")


def load_records(path):
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON ({error.msg})") from error
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number}: each record must be a JSON object")
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValueError(f"line {line_number}: missing required fields: {', '.join(missing)}")
            if not all(isinstance(record[field], str) for field in REQUIRED_FIELDS):
                raise ValueError(f"line {line_number}: required fields must all be strings")
            if not record["task_id"].strip() or not record["model_family"].strip():
                raise ValueError(f"line {line_number}: task_id and model_family cannot be empty")
            record["model_family"] = record["model_family"].strip().upper()
            records.append(record)
    if not records:
        raise ValueError("input contains no records")
    return records


def analyze(records):
    totals = defaultdict(lambda: [0, 0])
    task_totals = defaultdict(lambda: [0, 0])

    for record in records:
        family = record["model_family"].strip().upper()
        task = record["task_id"].strip()
        correct = record["prediction"].strip().casefold() == record["expected"].strip().casefold()
        totals[family][1] += 1
        totals[family][0] += correct
        task_totals[(task, family)][1] += 1
        task_totals[(task, family)][0] += correct

    family_accuracy = {
        family: {"correct": correct, "total": total, "accuracy": correct / total}
        for family, (correct, total) in sorted(totals.items())
    }
    task_accuracy = {}
    for (task, family), (correct, total) in sorted(task_totals.items()):
        task_accuracy.setdefault(task, {})[family] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
        }

    gaps = {}
    if "LLM" in totals:
        for family in FAMILIES:
            if family == "LLM":
                continue
            shared_tasks = [
                task for task, results in task_accuracy.items()
                if family in results and "LLM" in results
            ]
            if shared_tasks:
                differences = [
                    task_accuracy[task][family]["accuracy"]
                    - task_accuracy[task]["LLM"]["accuracy"]
                    for task in shared_tasks
                ]
                gaps[family] = {
                    "baseline": "LLM",
                    "tasks_compared": len(shared_tasks),
                    "mean_accuracy_difference": sum(differences) / len(differences),
                }

    return {
        "family_accuracy": family_accuracy,
        "task_accuracy": task_accuracy,
        "gaps_vs_llm": gaps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare exact-match results for language and multimodal model families."
    )
    parser.add_argument("input", help="JSONL file containing benchmark result records")
    args = parser.parse_args()

    try:
        report = analyze(load_records(args.input))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
