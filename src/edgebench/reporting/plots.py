"""Accuracy vs FPS and related figures.

matplotlib is an optional dependency; import errors surface only when
plotting is actually requested.
"""

from __future__ import annotations

from typing import Any


def plot_accuracy_vs_fps(rows: list[dict[str, Any]], output_path: str) -> None:
    """Scatter mAP@[.5:.95] against derived model FPS, one series per device."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("map50_95") is None or row.get("fps_model_derived") is None:
            continue
        series.setdefault(str(row.get("device", "unknown")), []).append(row)

    figure, axes = plt.subplots(figsize=(8, 6))
    for device, device_rows in sorted(series.items()):
        xs = [float(row["fps_model_derived"]) for row in device_rows]
        ys = [float(row["map50_95"]) for row in device_rows]
        labels = [
            f"{row['model']}/{row['runtime']}/{row['precision']}" for row in device_rows
        ]
        axes.scatter(xs, ys, label=device)
        for x, y, label in zip(xs, ys, labels):
            axes.annotate(label, (x, y), fontsize=6, alpha=0.75)
    axes.set_xlabel("FPS (model-only, derived from mean latency)")
    axes.set_ylabel("mAP@[0.50:0.95]")
    axes.set_title("Accuracy vs FPS")
    axes.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
