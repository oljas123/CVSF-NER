"""Command-line entry points; heavyweight model imports are loaded on demand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import read_jsonl, write_json, write_jsonl


def parser():
    cli = argparse.ArgumentParser(prog="cvsf", description="Candidate verification and score fusion for NER")
    commands = cli.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Download source corpora and restore a recorded split")
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument("--registry", default="configs/data_sources.json", type=Path)
    prepare.add_argument("--cache", default="data/raw", type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--offline", action="store_true")
    for name in ("train", "external-train"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--data", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--run", choices=["Run1", "Run2", "Run3"], default="Run1")
        sub.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        sub.add_argument("--max-steps", type=int, default=-1, help="BIO/external debug limit; omit for paper runs")
        if name == "train":
            sub.add_argument("--without-controls", action="store_true", help="Skip the partner BIO model")
        else:
            sub.add_argument("--arm", choices=["spanmarker", "gliner"], required=True)
    selection = commands.add_parser("select", help="Choose internal-system operating points on Dev-S")
    selection.add_argument("--model", required=True, type=Path)
    selection.add_argument("--data", required=True, type=Path)
    selection.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    for name in ("predict", "external-predict", "ablate"):
        sub = commands.add_parser(name)
        sub.add_argument("--model", required=True, type=Path)
        sub.add_argument("--input", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    evaluation = commands.add_parser("evaluate", help="Exact typed-span micro metrics")
    evaluation.add_argument("--gold", required=True, type=Path)
    evaluation.add_argument("--predictions", required=True, type=Path)
    evaluation.add_argument("--output", type=Path)
    consensus = commands.add_parser("consensus", help="Two-of-three exact-span voting")
    consensus.add_argument("--predictions", nargs=3, required=True, type=Path)
    consensus.add_argument("--output", required=True, type=Path)
    bootstrap = commands.add_parser("bootstrap", help="Paired sentence bootstrap over matched model runs")
    bootstrap.add_argument("--gold", required=True, type=Path)
    bootstrap.add_argument("--left", nargs="+", required=True, type=Path)
    bootstrap.add_argument("--right", nargs="+", required=True, type=Path)
    bootstrap.add_argument("--seed", type=int, default=2026)
    bootstrap.add_argument("--repetitions", type=int, default=10000)
    bootstrap.add_argument("--index-dtype", choices=["int32", "int64"], default="int32")
    bootstrap.add_argument("--output", type=Path)
    results = commands.add_parser("results", help="Verify released paper counts and display the main table")
    results.add_argument("--folder", type=Path, default="results/paper")
    results.add_argument("--markdown", type=Path)
    smoke = commands.add_parser("smoke", help="Run the complete pipeline on synthetic data, entirely on CPU")
    smoke.add_argument("--output", required=True, type=Path)
    return cli


def aligned(predictions, gold):
    if [str(row["sent_id"]) for row in predictions] != [str(row["sent_id"]) for row in gold]:
        raise ValueError("Sentence IDs and order must match the reference")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        from .data import prepare

        result = prepare(args.registry, args.manifest, args.cache, args.output, offline=args.offline)
    elif args.command == "train":
        from .pipeline import fit

        result = fit(
            args.config,
            args.data,
            args.output,
            args.run,
            device=args.device,
            controls=not args.without_controls,
            max_steps=args.max_steps,
        )
    elif args.command == "select":
        from .pipeline import select_development

        choices = select_development(args.model, args.data, device=args.device)
        result = {system: value["option"] for system, value in choices.items()}
    elif args.command == "predict":
        from .pipeline import predict_file

        values = predict_file(args.model, args.input, args.output, device=args.device)
        result = dict(systems=list(values), output=str(args.output))
    elif args.command == "external-train":
        from .external import train_external

        result = train_external(
            args.config, args.data, args.output, args.arm, args.run, device=args.device, max_steps=args.max_steps
        )
    elif args.command == "ablate":
        from .pipeline import ablate_file

        result = ablate_file(args.model, args.input, args.output, device=args.device)
    elif args.command == "external-predict":
        from .external import predict_external

        predictions = predict_external(args.model, args.input, args.output, device=args.device)
        result = dict(sentences=len(predictions), output=str(args.output))
    elif args.command == "evaluate":
        from .metrics import entities, metric

        gold, predicted = read_jsonl(args.gold), read_jsonl(args.predictions)
        aligned(predicted, gold)
        tp = total = target = 0
        for row, prediction in zip(gold, predicted):
            g, p = set(entities(row["tags_bio"], strict=True)), set(map(tuple, prediction["entities"]))
            tp += len(g & p)
            total += len(p) + len(prediction.get("unaligned_entities", []))
            target += len(g)
        result = metric(tp, total, target)
        if args.output:
            write_json(args.output, result)
    elif args.command == "consensus":
        from .metrics import consensus_rows

        inputs = [read_jsonl(path) for path in args.predictions]
        predictions = consensus_rows(inputs)
        write_jsonl(args.output, predictions)
        result = dict(sentences=len(predictions), output=str(args.output))
    elif args.command == "bootstrap":
        from .metrics import entities, member_bootstrap

        gold = read_jsonl(args.gold)
        left, right = ([read_jsonl(path) for path in paths] for paths in (args.left, args.right))
        for member in left + right:
            aligned(member, gold)

        def spans(rows):
            return [
                set(map(tuple, row["entities"])) | {("unaligned", *p) for p in row.get("unaligned_entities", [])}
                for row in rows
            ]

        result = member_bootstrap(
            [spans(member) for member in left],
            [spans(member) for member in right],
            [set(entities(row["tags_bio"], strict=True)) for row in gold],
            seed=args.seed,
            repetitions=args.repetitions,
            index_dtype=args.index_dtype,
        )
        if args.output:
            write_json(args.output, result)
    elif args.command == "results":
        from .results import main_table, verify

        result = verify(args.folder)
        table = main_table(args.folder)
        if args.markdown:
            args.markdown.parent.mkdir(parents=True, exist_ok=True)
            args.markdown.write_text(table, encoding="utf-8")
        print(table)
    else:
        from .smoke import run

        result = run(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
