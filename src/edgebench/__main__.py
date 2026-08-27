"""edgebench command line interface.

    python -m edgebench run <experiment.yaml>
    python -m edgebench export <model> --to onnx|tensorrt|ncnn [--precision P]
    python -m edgebench aggregate [raw_dir]
    python -m edgebench report [raw_dir] [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from edgebench.config import load_experiment, load_model_config
from edgebench.exporters import artifact_path_for
from edgebench.models import get_detector, list_detectors
from edgebench.paths import REPO_ROOT
from edgebench.types import SupportStatus

DEFAULT_RAW_DIR = REPO_ROOT / "results" / "raw"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="edgebench", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run one experiment YAML")
    run_parser.add_argument("experiment", help="path to an experiment YAML")

    export_parser = subparsers.add_parser("export", help="export a model artifact")
    export_parser.add_argument("model", choices=list_detectors())
    export_parser.add_argument("--to", required=True, choices=["onnx", "tensorrt", "ncnn"])
    export_parser.add_argument("--precision", default="fp32")
    export_parser.add_argument(
        "--onnx-path",
        default=None,
        help="source ONNX for tensorrt/ncnn (default: the onnxruntime artifact)",
    )

    aggregate_parser = subparsers.add_parser(
        "aggregate", help="print the result tables for a raw-results directory"
    )
    aggregate_parser.add_argument(
        "raw_dir", nargs="?", default=str(DEFAULT_RAW_DIR)
    )

    report_parser = subparsers.add_parser(
        "report", help="write summary tables (and figures) from raw results"
    )
    report_parser.add_argument("raw_dir", nargs="?", default=str(DEFAULT_RAW_DIR))
    report_parser.add_argument(
        "--out", default=str(REPO_ROOT / "results" / "summaries")
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args.experiment)
    if args.command == "export":
        return _cmd_export(args.model, args.to, args.precision, args.onnx_path)
    if args.command == "aggregate":
        return _cmd_aggregate(Path(args.raw_dir))
    if args.command == "report":
        return _cmd_report(Path(args.raw_dir), Path(args.out))
    parser.error(f"unknown command {args.command}")
    return 2


def _cmd_run(experiment_path: str) -> int:
    from edgebench.benchmark import BenchmarkRunner

    experiment = load_experiment(experiment_path)
    result = BenchmarkRunner(experiment).run()
    if result.status is SupportStatus.UNSUPPORTED:
        print(f"UNSUPPORTED: {result.unsupported_reason}", file=sys.stderr)
        return 3
    print(
        f"{result.model} on {result.device} [{result.runtime}/{result.precision}]: "
        f"{result.latency_model_mean_ms:.2f} ms mean, "
        f"{result.fps_model_derived:.1f} FPS (model), "
        f"mAP50 {result.map50:.3f}, mAP50-95 {result.map50_95:.3f}"
    )
    return 0


def _cmd_export(model: str, target: str, precision: str, onnx_path: str | None) -> int:
    config = load_model_config(model)
    adapter = get_detector(model)
    adapter.configure(config, _default_settings())
    if target == "onnx":
        if precision not in {"fp16", "fp32"}:
            raise ValueError("ONNX export precision must be 'fp16' or 'fp32'")
        artifact = artifact_path_for(
            model, "onnxruntime", precision, checkpoint=config.checkpoint
        )
        adapter.export_onnx(str(artifact))
        if precision == "fp16":
            from edgebench.exporters.onnx import convert_onnx_to_fp16

            convert_onnx_to_fp16(artifact)
        print(f"wrote {artifact}")
        return 0

    if onnx_path:
        source = Path(onnx_path)
    else:
        source_precision = (
            precision
            if target == "tensorrt" and precision in {"fp16", "fp32"}
            else "fp32"
        )
        source = artifact_path_for(
            model, "onnxruntime", source_precision, checkpoint=config.checkpoint
        )
        if source_precision != "fp32" and not source.is_file():
            source = artifact_path_for(
                model, "onnxruntime", "fp32", checkpoint=config.checkpoint
            )
    if target == "tensorrt":
        from edgebench.exporters.tensorrt import export_tensorrt

        artifact = artifact_path_for(model, "tensorrt", precision, checkpoint=config.checkpoint)
        export_tensorrt(str(source), str(artifact), precision=precision,
                        input_size=config.input_size)
        print(f"wrote {artifact}")
        return 0

    from edgebench.exporters.ncnn import export_ncnn

    base = artifact_path_for(model, "ncnn", precision, checkpoint=config.checkpoint)
    param_path, bin_path = export_ncnn(str(source), str(base.parent), model_stem=base.name)
    print(f"wrote {param_path}\nwrote {bin_path}")
    return 0


def _cmd_aggregate(raw_dir: Path) -> int:
    from edgebench.reporting.aggregate import aggregate_results
    from edgebench.reporting.tables import render_tables

    rows = aggregate_results(str(raw_dir))
    if not rows:
        print(f"No result JSON files under {raw_dir}", file=sys.stderr)
        return 1
    print(render_tables(rows))
    return 0


def _cmd_report(raw_dir: Path, out_dir: Path) -> int:
    from edgebench.reporting.aggregate import aggregate_results
    from edgebench.reporting.tables import render_tables

    rows = aggregate_results(str(raw_dir))
    if not rows:
        print(f"No result JSON files under {raw_dir}", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_path = out_dir / "tables.md"
    tables_path.write_text(render_tables(rows), encoding="utf-8")
    print(f"wrote {tables_path}")
    try:
        from edgebench.reporting.plots import plot_accuracy_vs_fps

        figures_dir = REPO_ROOT / "results" / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        figure = figures_dir / "accuracy_vs_fps.png"
        plot_accuracy_vs_fps(rows, str(figure))
        print(f"wrote {figure}")
    except ImportError:
        print("matplotlib not installed; skipping figures", file=sys.stderr)
    return 0


def _default_settings():
    from edgebench.config import load_benchmark_defaults

    return load_benchmark_defaults()


if __name__ == "__main__":
    raise SystemExit(main())
