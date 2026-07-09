"""Render ACT animation frames using SPR frames."""

from __future__ import annotations

import cv2
import numpy as np

from pybot.recognition.act_reader import ActFrameRef, ActSpriteLayer
from pybot.recognition.spr_reader import SprFile


def render_act_frame(spr_file: SprFile, frame: ActFrameRef) -> np.ndarray:
    """
    Composite one ACT frame to a BGRA image.

    For single-layer monster sprites this is usually one SPR frame pasted at x/y.
    """
    if not frame.layers:
        return np.zeros((1, 1, 4), dtype=np.uint8)

    min_x = min(layer.x for layer in frame.layers)
    min_y = min(layer.y for layer in frame.layers)
    max_x = max(layer.x + _layer_width(spr_file, layer) for layer in frame.layers)
    max_y = max(layer.y + _layer_height(spr_file, layer) for layer in frame.layers)

    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    for layer in frame.layers:
        sprite = _get_spr_frame(spr_file, layer.spr_frame_index)
        if sprite is None:
            continue
        sprite = _apply_layer_transform(sprite, layer)
        paste_x = layer.x - min_x
        paste_y = layer.y - min_y
        _blit_bgra(canvas, sprite, paste_x, paste_y)

    return canvas


def _get_spr_frame(spr_file: SprFile, index: int):
    frame = spr_file.get_frame(index)
    return None if frame is None else frame.rgba


def _layer_width(spr_file: SprFile, layer: ActSpriteLayer) -> int:
    sprite = _get_spr_frame(spr_file, layer.spr_frame_index)
    if sprite is None:
        return 0
    return max(1, int(round(sprite.shape[1] * abs(layer.scale_x))))


def _layer_height(spr_file: SprFile, layer: ActSpriteLayer) -> int:
    sprite = _get_spr_frame(spr_file, layer.spr_frame_index)
    if sprite is None:
        return 0
    return max(1, int(round(sprite.shape[0] * abs(layer.scale_y))))


def _apply_layer_transform(sprite: np.ndarray, layer: ActSpriteLayer) -> np.ndarray:
    src = np.fliplr(sprite) if layer.mirror else sprite
    width = max(1, int(round(src.shape[1] * abs(layer.scale_x))))
    height = max(1, int(round(src.shape[0] * abs(layer.scale_y))))
    if width != src.shape[1] or height != src.shape[0]:
        src = cv2.resize(src, (width, height), interpolation=cv2.INTER_NEAREST)

    tint_r, tint_g, tint_b, tint_a = layer.color_tint
    if (tint_r, tint_g, tint_b, tint_a) in {(255, 255, 255, 255), (255, 0, 0, 255)}:
        return src
    tinted = src.astype(np.float32)
    tinted[:, :, 0] *= tint_b / 255.0
    tinted[:, :, 1] *= tint_g / 255.0
    tinted[:, :, 2] *= tint_r / 255.0
    tinted[:, :, 3] *= tint_a / 255.0
    src = np.clip(tinted, 0, 255).astype(np.uint8)

    return src


def _blit_bgra(canvas: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
    src = sprite
    sh, sw = src.shape[:2]
    ch, cw = canvas.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(cw, x + sw)
    y2 = min(ch, y + sh)
    if x1 >= x2 or y1 >= y2:
        return

    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    src_slice = src[sy1:sy2, sx1:sx2]
    dst_slice = canvas[y1:y2, x1:x2]
    alpha = src_slice[:, :, 3:4].astype(np.float32) / 255.0
    dst_slice[:] = (src_slice.astype(np.float32) * alpha + dst_slice.astype(np.float32) * (1.0 - alpha)).astype(
        np.uint8
    )
