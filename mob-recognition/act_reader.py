"""
Ragnarok Online .act file reader.

Supports ACT versions 2.0 through 2.5 (action clips, frames, sprite layers).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from binary_reader import BinaryReader


class ActParseError(Exception):
    """Raised when a .act file cannot be parsed."""


@dataclass
class ActSpriteLayer:
    x: int
    y: int
    spr_frame_index: int
    mirror: bool
    color_tint: tuple[int, int, int, int]
    scale_x: float
    scale_y: float
    rotation: float
    image_type: int


@dataclass
class ActFrameRef:
    spr_frame_index: int
    delay_ms: int
    action_index: int
    frame_index: int
    layers: List[ActSpriteLayer]


@dataclass
class ActAction:
    name: str
    index: int
    frames: List[ActFrameRef]


@dataclass
class ActFile:
    path: Path
    version: int
    actions: List[ActAction]


class ActReader:
    def __init__(self, act_path: Path):
        self.act_path = Path(act_path)

    def load(self) -> ActFile:
        if not self.act_path.is_file():
            raise ActParseError(f"ACT file not found: {self.act_path}")

        reader = BinaryReader(self.act_path.read_bytes())
        signature = reader.read_bytes(2)
        if signature != b"AC":
            raise ActParseError(f"invalid ACT signature: {signature!r}")

        version = reader.read_uint16()
        if version < 0x200 or version > 0x205:
            raise ActParseError(f"unsupported ACT version: 0x{version:04X}")

        action_count = reader.read_uint16()
        reader.skip(10)  # reserved

        actions: List[ActAction] = []
        for action_index in range(action_count):
            actions.append(self._read_action(reader, version, action_index))

        intervals: List[float] = []
        if version >= 0x201:
            event_count = reader.read_uint32()
            for _ in range(event_count):
                reader.read_fixed_string(40)
        if version >= 0x202:
            for _ in range(action_count):
                intervals.append(reader.read_float())

        if reader.offset != len(reader.data):
            raise ActParseError(
                f"ACT layout mismatch at offset {reader.offset}, expected {len(reader.data)}"
            )

        if version >= 0x202 and len(intervals) == action_count:
            for action_index, action in enumerate(actions):
                delay_ms = int(round(intervals[action_index] * 24))
                for frame in action.frames:
                    frame.delay_ms = delay_ms

        return ActFile(
            path=self.act_path,
            version=version,
            actions=actions,
        )

    def _read_action(self, reader: BinaryReader, version: int, action_index: int) -> ActAction:
        frame_count = reader.read_uint32()
        frames: List[ActFrameRef] = []
        for frame_index in range(frame_count):
            layers = self._read_frame_layers(reader, version)
            primary_index = layers[0].spr_frame_index if layers else -1
            frames.append(
                ActFrameRef(
                    spr_frame_index=primary_index,
                    delay_ms=0,
                    action_index=action_index,
                    frame_index=frame_index,
                    layers=layers,
                )
            )
        return ActAction(
            name=f"action_{action_index}",
            index=action_index,
            frames=frames,
        )

    def _read_frame_layers(self, reader: BinaryReader, version: int) -> List[ActSpriteLayer]:
        reader.skip(32)  # attack range + fit range
        layer_count = reader.read_uint32()
        layers: List[ActSpriteLayer] = []
        for _ in range(layer_count):
            layer = self._read_sprite_layer(reader, version)
            if layer.spr_frame_index >= 0:
                layers.append(layer)

        if version >= 0x200:
            reader.skip(4)  # event id

        if version >= 0x203:
            attach_count = reader.read_uint32()
            reader.skip(attach_count * 16)

        return layers

    def _read_sprite_layer(self, reader: BinaryReader, version: int) -> ActSpriteLayer:
        x = reader.read_int32()
        y = reader.read_int32()
        spr_frame_index = reader.read_int32()
        flags = reader.read_uint32()
        color = reader.read_bytes(4)
        scale_x = reader.read_float()
        if version >= 0x204:
            scale_y = reader.read_float()
        else:
            scale_y = scale_x
        rotation = reader.read_float()
        image_type = reader.read_int32()
        if version >= 0x205:
            reader.skip(8)  # width, height

        return ActSpriteLayer(
            x=x,
            y=y,
            spr_frame_index=spr_frame_index,
            mirror=bool(flags & 1),
            color_tint=(color[0], color[1], color[2], color[3]),
            scale_x=scale_x,
            scale_y=scale_y,
            rotation=rotation,
            image_type=image_type,
        )
