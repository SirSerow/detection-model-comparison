"""Accuracy-versus-throughput figures.

Matplotlib is optional; import errors surface only when plotting is requested.
The comparison is faceted by device because edge devices and desktop GPUs can
have throughput ranges that differ by two orders of magnitude.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

_MODEL_LABELS = {
    "damo_yolo_t": "DAMO-YOLO-T",
    "picodet_s": "PicoDet-S",
    "rfdetr_nano": "RF-DETR Nano",
    "rtdetrv2_s": "RT-DETRv2-S",
    "rtmdet_tiny": "RTMDet-Tiny",
    "yolo26n": "YOLO26-N",
    "yolox_tiny": "YOLOX-Tiny",
}

_DEVICE_LABELS = {
    "raspberry_pi_4": "Raspberry Pi 4",
    "rtx_5070_laptop": "RTX 5070 Laptop GPU",
    "jetson_orin_nano_super": "Jetson Orin Nano Super",
}

# A fixed, colorblind-friendly mapping keeps model identity consistent across panels.
_MODEL_COLORS = {
    "damo_yolo_t": "#E69F00",
    "picodet_s": "#6A3D9A",
    "rfdetr_nano": "#009E73",
    "rtdetrv2_s": "#CC79A7",
    "rtmdet_tiny": "#0072B2",
    "yolo26n": "#56B4E9",
    "yolox_tiny": "#8A8500",
}


@dataclass(frozen=True)
class _BackendStyle:
    marker: str
    filled: bool
    label: str


_BACKEND_STYLES = {
    ("pytorch", "fp32"): _BackendStyle("o", False, "PyTorch · FP32"),
    ("pytorch", "fp16"): _BackendStyle("o", True, "PyTorch · FP16"),
    ("onnxruntime", "fp32"): _BackendStyle("s", False, "ONNX Runtime · FP32"),
    ("onnxruntime", "fp16"): _BackendStyle("s", True, "ONNX Runtime · FP16"),
    ("tensorrt", "fp16"): _BackendStyle("D", True, "TensorRT · FP16"),
    ("ncnn", "fp32"): _BackendStyle("^", False, "NCNN · FP32"),
    ("ncnn", "int8"): _BackendStyle("^", True, "NCNN · INT8"),
}


def _backend_style(runtime: str, precision: str) -> _BackendStyle:
    return _BACKEND_STYLES.get(
        (runtime, precision),
        _BackendStyle("X", True, f"{runtime} · {precision.upper()}"),
    )


def _display_model(model: str) -> str:
    return _MODEL_LABELS.get(model, model.replace("_", " ").title())


def _display_device(device: str) -> str:
    return _DEVICE_LABELS.get(device, device.replace("_", " ").title())


def _spread_label_positions(
    values: list[tuple[str, float]],
    *,
    lower: float,
    upper: float,
    minimum_gap: float,
) -> dict[str, float]:
    """Return ordered y positions with enough space for readable direct labels."""
    ordered = sorted(values, key=lambda item: item[1])
    if not ordered:
        return {}

    positions: list[tuple[str, float]] = []
    for key, value in ordered:
        position = max(value, positions[-1][1] + minimum_gap) if positions else value
        positions.append((key, position))

    overflow = positions[-1][1] - upper
    if overflow > 0:
        positions = [(key, position - overflow) for key, position in positions]
    underflow = lower - positions[0][1]
    if underflow > 0:
        positions = [(key, position + underflow) for key, position in positions]
    return dict(positions)


def plot_accuracy_vs_fps(rows: list[dict[str, Any]], output_path: str) -> None:
    """Plot one independently scaled FPS panel per device with a shared mAP axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("map50_95") is None or row.get("fps_model_derived") is None:
            continue
        series[str(row.get("device", "unknown"))].append(row)

    if not series:
        raise ValueError("no successful accuracy/FPS results to plot")

    # Slowest device first makes the scale progression read naturally left-to-right.
    device_order = sorted(
        series,
        key=lambda device: sum(
            float(row["fps_model_derived"]) for row in series[device]
        )
        / len(series[device]),
    )
    all_accuracy = [
        float(row["map50_95"]) for device in device_order for row in series[device]
    ]
    y_padding = max(0.012, (max(all_accuracy) - min(all_accuracy)) * 0.08)
    y_lower = min(all_accuracy) - y_padding
    y_upper = max(all_accuracy) + y_padding
    minimum_label_gap = (y_upper - y_lower) * 0.063

    width_ratios = [max(1.0, min(1.45, len(series[device]) / 12)) for device in device_order]
    figure_width = max(10.5, 6.2 * sum(width_ratios))
    figure, axes = plt.subplots(
        1,
        len(device_order),
        figsize=(figure_width, 6.4),
        sharey=True,
        gridspec_kw={"width_ratios": width_ratios},
    )
    if len(device_order) == 1:
        axes = [axes]

    used_backends: set[tuple[str, str]] = set()
    for index, (axes_item, device) in enumerate(zip(axes, device_order)):
        device_rows = series[device]
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in device_rows:
            by_model[str(row["model"])].append(row)

        maximum_fps = max(float(row["fps_model_derived"]) for row in device_rows)
        label_x = maximum_fps * 1.08
        axes_item.set_xlim(0, maximum_fps * 1.38)
        axes_item.set_ylim(y_lower, y_upper)
        axes_item.set_facecolor("#FAFAFA")
        axes_item.grid(axis="both", color="#D9D9D9", linewidth=0.7, alpha=0.65)
        axes_item.set_axisbelow(True)
        axes_item.spines["top"].set_visible(False)
        axes_item.spines["right"].set_visible(False)

        label_anchors: list[tuple[str, float]] = []
        for model, model_rows in sorted(by_model.items()):
            color = _MODEL_COLORS.get(model, "#555555")
            ordered_rows = sorted(
                model_rows, key=lambda row: float(row["fps_model_derived"])
            )
            xs = [float(row["fps_model_derived"]) for row in ordered_rows]
            ys = [float(row["map50_95"]) for row in ordered_rows]
            if len(ordered_rows) > 1:
                axes_item.plot(xs, ys, color=color, linewidth=1.3, alpha=0.38, zorder=1)

            for row, x_value, y_value in zip(ordered_rows, xs, ys):
                backend = (str(row["runtime"]), str(row["precision"]))
                used_backends.add(backend)
                style = _backend_style(*backend)
                axes_item.scatter(
                    [x_value],
                    [y_value],
                    marker=style.marker,
                    s=46,
                    facecolors=color if style.filled else "white",
                    edgecolors=color,
                    linewidths=1.8,
                    zorder=3,
                )

            label_anchors.append((model, sum(ys) / len(ys)))

        label_positions = _spread_label_positions(
            label_anchors,
            lower=y_lower + y_padding * 0.25,
            upper=y_upper - y_padding * 0.25,
            minimum_gap=minimum_label_gap,
        )
        for model, model_rows in sorted(by_model.items()):
            color = _MODEL_COLORS.get(model, "#555555")
            anchor = max(model_rows, key=lambda row: float(row["fps_model_derived"]))
            anchor_x = float(anchor["fps_model_derived"])
            anchor_y = float(anchor["map50_95"])
            axes_item.annotate(
                _display_model(model),
                xy=(anchor_x, anchor_y),
                xytext=(label_x, label_positions[model]),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=9.3,
                fontweight="semibold",
                color="#222222",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 1.0,
                    "alpha": 0.94,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "linewidth": 1.0,
                    "linestyle": "--",
                    "alpha": 0.72,
                    "shrinkA": 2,
                    "shrinkB": 4,
                },
                zorder=4,
            )

        axes_item.set_title(
            _display_device(device), fontsize=13, fontweight="bold", pad=12
        )
        axes_item.set_xlabel("FPS", fontsize=10.5, labelpad=8)
        axes_item.tick_params(axis="both", labelsize=9.5)
        if index == 0:
            axes_item.set_ylabel("COCO mAP@[0.50:0.95]", fontsize=11, labelpad=9)

    backend_handles = []
    backend_order = [backend for backend in _BACKEND_STYLES if backend in used_backends]
    backend_order.extend(sorted(used_backends.difference(_BACKEND_STYLES)))
    for backend in backend_order:
        style = _backend_style(*backend)
        backend_handles.append(
            Line2D(
                [0],
                [0],
                marker=style.marker,
                linestyle="none",
                markerfacecolor="#555555" if style.filled else "white",
                markeredgecolor="#444444",
                markeredgewidth=1.5,
                markersize=6.5,
                label=style.label,
            )
        )

    figure.legend(
        handles=backend_handles,
        loc="lower center",
        ncol=min(5, len(backend_handles)),
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.14, wspace=0.12)
    figure.savefig(output_path, dpi=200, facecolor="white")
    plt.close(figure)
