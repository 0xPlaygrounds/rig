"""Regenerate the offline image-embedding fixtures with onnx==1.17.0.

Run this file from any directory after installing that Python package.
Tests use the generated files directly and do not require Python or onnx.

The model takes RGB floats shaped [batch, 3, 2, 2] and returns channel means
shaped [batch, 3]. FastEmbed's preprocessing rescales pixels by 1/255, so
red.png and green.png produce [1, 0, 0] and [0, 1, 0], respectively, before
its output normalization. The model contains no trained weights.
"""

import json
from pathlib import Path
import struct
import zlib

import onnx
from onnx import TensorProto, helper


def png(rgb: tuple[int, int, int]) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    rows = (b"\x00" + bytes(rgb) * 2) * 2
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, level=0))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    directory = Path(__file__).resolve().parent
    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceMean",
                ["pixel_values"],
                ["image_embeds"],
                axes=[2, 3],
                keepdims=0,
            )
        ],
        "channel_mean_image_embedding",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["batch", 3, 2, 2])],
        [helper.make_tensor_value_info("image_embeds", TensorProto.FLOAT, ["batch", 3])],
    )
    model = helper.make_model(
        graph,
        producer_name="rig-fastembed-test-fixtures",
        opset_imports=[helper.make_opsetid("", 13)],
        ir_version=8,
    )
    onnx.checker.check_model(model)
    (directory / "image_mean.onnx").write_bytes(model.SerializeToString())

    preprocessor = {
        "image_processor_type": "CLIPImageProcessor",
        "do_resize": True,
        "size": {"height": 2, "width": 2},
        "do_center_crop": False,
        "do_rescale": True,
        "rescale_factor": 1 / 255,
        "do_normalize": False,
    }
    (directory / "image_preprocessor.json").write_bytes(
        (json.dumps(preprocessor, indent=2) + "\n").encode("utf-8")
    )
    (directory / "red.png").write_bytes(png((255, 0, 0)))
    (directory / "green.png").write_bytes(png((0, 255, 0)))


if __name__ == "__main__":
    main()
