"""Command line entry point: ``fakenews <prepare|baseline|transformer|liar-reference|report>``."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from fakenews import baseline, config, data
from fakenews.evaluate import metrics


def _load_corpus() -> pd.DataFrame:
    if not config.CORPUS_PARQUET.exists():
        raise SystemExit(f"{config.CORPUS_PARQUET} missing, run `fakenews prepare` first")
    return pd.read_parquet(config.CORPUS_PARQUET)


def _evaluate(predict: Callable[[list[str]], np.ndarray], test: pd.DataFrame) -> dict:
    liar = data.load_liar(config.LIAR_DIR, "test")
    return {
        "fakenewscorpus_test": metrics(test["label"], predict(test["content"].tolist())),
        "liar_test": metrics(liar["label"], predict(liar["content"].tolist())),
    }


def _write(name: str, result: dict) -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    path = config.RESULTS_DIR / f"{name}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {path}")


def cmd_prepare(args) -> None:
    df = data.sample_fakenewscorpus(
        config.FNC_CSV, per_domain=args.per_domain, per_class=args.per_class, seed=args.seed
    )
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.CORPUS_PARQUET, index=False)
    stats = {
        "rows": len(df),
        "domains": int(df["domain"].nunique()),
        "by_type": df["type"].value_counts().to_dict(),
        "by_label": {config.LABEL_NAMES[k]: v for k, v in df["label"].value_counts().items()},
    }
    _write("corpus_stats", stats)


def cmd_baseline(args) -> None:
    train, _, test = data.split(_load_corpus(), args.split, seed=args.seed)
    model = baseline.build()
    model.fit(train["content"].tolist(), train["label"].to_numpy())
    baseline.save(model, config.MODELS_DIR / f"baseline_{args.split}.joblib")
    result = {"model": "tfidf_logreg", "split": args.split, "train_rows": len(train)}
    result |= _evaluate(lambda t: baseline.predict_proba(model, t), test)
    result["top_features"] = baseline.top_features(model)
    _write(f"baseline_{args.split}", result)


def cmd_transformer(args) -> None:
    from fakenews import transformer

    train, val, test = data.split(_load_corpus(), args.split, seed=args.seed)
    if args.max_train:
        train = train.sample(min(args.max_train, len(train)), random_state=args.seed)
    cfg = transformer.TrainConfig(
        model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed
    )
    out_dir = config.MODELS_DIR / f"transformer_{args.split}"
    log = transformer.train(train, val, cfg, out_dir)
    model, tok = transformer.load(out_dir)
    result = {"model": args.model, "split": args.split, "train_rows": len(train)}
    result |= _evaluate(lambda t: transformer.predict_proba(model, tok, t, cfg.max_length), test)
    result["training"] = log
    _write(f"transformer_{args.split}", result)


def cmd_liar_reference(args) -> None:
    """In-domain LIAR baseline: what a model trained on LIAR itself reaches."""
    train = data.load_liar(config.LIAR_DIR, "train")
    test = data.load_liar(config.LIAR_DIR, "test")
    model = baseline.build(max_features=50_000)
    model.fit(train["content"].tolist(), train["label"].to_numpy())
    p = baseline.predict_proba(model, test["content"].tolist())
    _write("liar_reference", {"model": "tfidf_logreg", "liar_test": metrics(test["label"], p)})


def render_report(results_dir: Path) -> str:
    rows = []
    for path in sorted(results_dir.glob("*_random.json")) + sorted(
        results_dir.glob("*_domain.json")
    ):
        r = json.loads(path.read_text())
        f, liar = r["fakenewscorpus_test"], r["liar_test"]
        rows.append(
            f"| {r['model']} | {r['split']} | {f['accuracy']:.3f} | {f['macro_f1']:.3f} "
            f"| {f['roc_auc']:.3f} | {liar['accuracy']:.3f} | {liar['macro_f1']:.3f} "
            f"| {liar['roc_auc']:.3f} |"
        )
    ref = results_dir / "liar_reference.json"
    if ref.exists():
        liar = json.loads(ref.read_text())["liar_test"]
        rows.append(
            f"| tfidf_logreg (trained on LIAR) | - | - | - | - | {liar['accuracy']:.3f} "
            f"| {liar['macro_f1']:.3f} | {liar['roc_auc']:.3f} |"
        )
    header = (
        "| Model | Split | FNC acc | FNC macro F1 | FNC AUC "
        "| LIAR acc | LIAR macro F1 | LIAR AUC |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    return "\n".join([header, *rows]) + "\n"


def cmd_report(args) -> None:
    table = render_report(config.RESULTS_DIR)
    (config.RESULTS_DIR / "RESULTS.md").write_text(table)
    readme = config.ROOT / "README.md"
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    text = readme.read_text()
    if start in text and end in text:
        head, rest = text.split(start, 1)
        readme.write_text(f"{head}{start}\n{table}{end}{rest.split(end, 1)[1]}")
    print(table)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fakenews")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="sample FakeNewsCorpus into data/processed")
    p.add_argument("--per-domain", type=int, default=1000)
    p.add_argument("--per-class", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("baseline", help="train and evaluate TF-IDF + logistic regression")
    p.add_argument("--split", choices=data.SPLITS, default="domain")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("transformer", help="fine-tune and evaluate a transformer")
    p.add_argument("--split", choices=data.SPLITS, default="domain")
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-train", type=int, default=0, help="subsample training rows (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_transformer)

    p = sub.add_parser("liar-reference", help="in-domain LIAR baseline for comparison")
    p.set_defaults(func=cmd_liar_reference)

    p = sub.add_parser("report", help="collect results/*.json into results/RESULTS.md")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
