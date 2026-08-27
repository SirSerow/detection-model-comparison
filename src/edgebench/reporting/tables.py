"""Primary result tables from the research spec.

One markdown table per device with the spec's column layout. Unsupported
combinations render as ``N/A — unsupported`` rows rather than being
omitted.
"""

from __future__ import annotations

from typing import Any

COLUMNS = [
    ("Model", "model"),
    ("Runtime", "runtime"),
    ("Precision", "precision"),
    ("FPS", "fps_model_derived"),
    ("Mean latency", "latency_model_mean_ms"),
    ("P50", "latency_model_p50_ms"),
    ("P95", "latency_model_p95_ms"),
    ("mAP50-95", "map50_95"),
    ("RAM", "ram_peak_mb"),
    ("VRAM", "vram_peak_mb"),
    ("Power", "power_w"),
]

_NUMERIC_FORMATS = {
    "fps_model_derived": "{:.1f}",
    "latency_model_mean_ms": "{:.2f}",
    "latency_model_p50_ms": "{:.2f}",
    "latency_model_p95_ms": "{:.2f}",
    "map50_95": "{:.3f}",
    "ram_peak_mb": "{:.0f}",
    "vram_peak_mb": "{:.0f}",
    "power_w": "{:.2f}",
}


def render_tables(rows: list[dict[str, Any]]) -> str:
    """Render one markdown result table per device."""
    devices: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        devices.setdefault(str(row.get("device", "unknown")), []).append(row)

    sections: list[str] = []
    for device in sorted(devices):
        device_rows = sorted(
            devices[device],
            key=lambda row: (
                str(row.get("model", "")),
                str(row.get("runtime", "")),
                str(row.get("precision", "")),
            ),
        )
        header = "| " + " | ".join(title for title, _ in COLUMNS) + " |"
        separator = "|" + "---|" * len(COLUMNS)
        lines = [f"## {device}", "", "**Input: 640 × 640, batch=1**", "", header, separator]
        for row in device_rows:
            lines.append(_render_row(row))
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def _render_row(row: dict[str, Any]) -> str:
    if row.get("status") == "unsupported":
        cells = [
            str(row.get("model", "")),
            str(row.get("runtime", "")),
            "N/A — unsupported",
        ] + ["N/A"] * (len(COLUMNS) - 3)
        return "| " + " | ".join(cells) + " |"
    cells = []
    for title, key in COLUMNS:
        value = row.get(key)
        if value is None:
            cells.append("—")
        elif key in _NUMERIC_FORMATS:
            cells.append(_NUMERIC_FORMATS[key].format(float(value)))
        else:
            cells.append(str(value))
    return "| " + " | ".join(cells) + " |"
