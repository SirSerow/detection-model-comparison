"""PP-PicoDet-S adapter for PaddleDetection deployment artifacts.

PicoDet remains Paddle-native: PyTorch is explicitly unsupported. ONNX
export runs PaddleDetection's official inference export followed by
``paddle2onnx`` and retains its postprocess/NMS output contract.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from edgebench.models._common import (
    ConfiguredDetector,
    decode_xyxy_scores,
    detections_from_arrays,
    output_sequence,
    prepare_image,
)

if TYPE_CHECKING:
    import numpy as np

    from edgebench.preprocessing import ResizeMeta
    from edgebench.types import Detection

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class PicoDetSAdapter(ConfiguredDetector):
    @property
    def name(self) -> str:
        return "picodet_s"

    def preprocess(
        self, image: np.ndarray
    ) -> tuple[dict[str, np.ndarray], ResizeMeta]:
        import numpy as np

        tensor, meta = prepare_image(
            image,
            self.input_size,
            geometry="resize",
            rgb=True,
            scale=255.0,
            mean=_IMAGENET_MEAN,
            std=_IMAGENET_STD,
        )
        # PaddleDetection's postprocessed export consumes source→input scale.
        scale_factor = np.asarray([[meta.scale_y, meta.scale_x]], dtype=np.float32)
        image_shape = np.asarray(
            [[meta.input_height, meta.input_width]], dtype=np.float32
        )
        return {
            "image": tensor,
            "im_shape": image_shape,
            "scale_factor": scale_factor,
        }, meta

    def postprocess(self, raw_output: Any, metadata: ResizeMeta) -> list[Detection]:
        import numpy as np

        outputs = output_sequence(raw_output)
        bbox = next(
            (item for item in outputs if item.ndim >= 2 and item.shape[-1] == 6),
            None,
        )
        if bbox is None:
            boxes = next(
                (item for item in outputs if item.ndim >= 2 and item.shape[-1] == 4),
                None,
            )
            scores = next(
                (
                    item
                    for item in outputs
                    if item.ndim >= 2
                    and (item.shape[-1] == 80 or item.shape[-2] == 80)
                ),
                None,
            )
            if boxes is None or scores is None:
                raise ValueError(
                    "Unexpected PicoDet output shapes; expected deployed bbox "
                    "rows (*,6) or raw boxes/scores, received "
                    f"{[item.shape for item in outputs]}"
                )
            boxes = boxes[0] if boxes.ndim == 3 else boxes
            scores = scores[0] if scores.ndim == 3 else scores
            if scores.shape[0] == 80 and scores.shape[-1] != 80:
                scores = scores.T
            return decode_xyxy_scores(
                boxes,
                scores,
                metadata,
                threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold,
            )
        if bbox.ndim == 3:
            bbox = bbox[0]
        # PaddleDetection bbox rows: [class_index, score, x1, y1, x2, y2].
        keep = np.isfinite(bbox).all(axis=1) & (
            bbox[:, 1] >= self.confidence_threshold
        )
        return detections_from_arrays(
            bbox[keep, 2:6],
            bbox[keep, 1],
            bbox[keep, 0].astype(np.int64),
            metadata,
            boxes_in_original_space=True,
        )

    def load_pytorch(self) -> Any:
        raise NotImplementedError(
            "PP-PicoDet-S is Paddle-native; PyTorch is N/A — unsupported"
        )

    def export_onnx(self, output_path: str) -> None:
        config_path = self.upstream_config_path()
        repository = _find_paddledetection_root(config_path)
        export_script = repository / "tools" / "export_model.py"
        converter = shutil.which("paddle2onnx")
        if converter is None:
            raise ImportError(
                "PicoDet ONNX export requires the `paddle2onnx` executable."
            )
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        width, height = self.input_size
        with tempfile.TemporaryDirectory(prefix="edgebench-picodet-") as directory:
            subprocess.run(
                [
                    sys.executable,
                    str(export_script),
                    "-c",
                    str(config_path),
                    "--output_dir",
                    directory,
                    "-o",
                    f"weights={self.checkpoint_path()}",
                    "use_gpu=False",
                    f"eval_size=[{height},{width}]",
                    "export.nms=False",
                    f"TestReader.inputs_def.image_shape=[1,3,{height},{width}]",
                ],
                cwd=repository,
                check=True,
            )
            model_directory = Path(directory) / config_path.stem
            model_file = model_directory / "model.pdmodel"
            params_file = model_directory / "model.pdiparams"
            if not model_file.is_file() or not params_file.is_file():
                raise FileNotFoundError(
                    f"PaddleDetection export did not create {model_file} and {params_file}"
                )
            subprocess.run(
                [
                    converter,
                    "--model_dir",
                    str(model_directory),
                    "--model_filename",
                    model_file.name,
                    "--params_filename",
                    params_file.name,
                    "--opset_version",
                    "11",
                    "--save_file",
                    str(target),
                ],
                check=True,
            )


def _find_paddledetection_root(config_path: Path) -> Path:
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "tools" / "export_model.py").is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot locate PaddleDetection/tools/export_model.py above {config_path}"
    )
