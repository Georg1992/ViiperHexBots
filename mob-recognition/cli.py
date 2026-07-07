"""Mob recognition CLI for the simple descriptor heatmap detector."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

MOB_REC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MOB_REC_DIR.parent
SIMPLE_DIR = MOB_REC_DIR / "simple"
for path in (str(MOB_REC_DIR), str(SIMPLE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from act_reader import ActReader
from capture import capture_region
from debug_renderer import save_simple_debug_bundle
from descriptors.descriptor_builder import SimpleDescriptorBuilder
from detector import SimpleMobDetector, load_simple_config
from spr_reader import SprReader


def parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = [int(p.strip()) for p in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("roi must be x,y,width,height")
    return tuple(parts)


def parse_scale_range(value: str) -> tuple[float, float]:
    parts = [float(p.strip()) for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("scale range must be min,max")
    low, high = min(parts), max(parts)
    if low <= 0 or high <= 0:
        raise argparse.ArgumentTypeError("scale values must be positive")
    return low, high


def apply_scale_calibration(config: dict, scale_range: tuple[float, float] | None, enforce_size_gate: bool) -> dict:
    calibrated = dict(config)
    if scale_range is not None:
        low, high = scale_range
        mid = (low + high) / 2.0
        calibrated["scales"] = [low, mid, high]
        calibrated["centerScales"] = [low, mid, high]
    calibrated["enforceObjectSizeGate"] = enforce_size_gate
    return calibrated


def emit_json(payload: dict, output_path: Path | None = None) -> None:
    text = json.dumps(payload, separators=(",", ":"))
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(output_path)
    print(text)


def candidate_to_json(candidate, screen_offset: tuple[int, int]) -> dict:
    ox, oy = screen_offset
    payload = candidate.to_dict()
    x, y, w, h = candidate.bbox
    payload.update(
        {
            "x": x + ox,
            "y": y + oy,
            "width": w,
            "height": h,
            "centerX": candidate.center_x + ox,
            "centerY": candidate.center_y + oy,
            "confidence": round(candidate.final_score, 4),
            "living": candidate.accepted and not candidate.is_dead,
            "dead": candidate.is_dead,
        }
    )
    return payload


def _load_frame(args: argparse.Namespace) -> tuple[object, tuple[int, int], int]:
    if args.image:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = PROJECT_ROOT / image_path
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(f"failed to read image: {image_path}")
        return frame, (0, 0), 0
    if not args.roi:
        raise ValueError("--roi is required without --image")
    x, y, w, h = args.roi
    return capture_region(x, y, w, h), (x, y), 0


def cmd_build_simple_descriptor(args: argparse.Namespace) -> int:
    descriptor = SimpleDescriptorBuilder(PROJECT_ROOT).build(args.mob.lower(), force=args.force)
    print(json.dumps({"ok": True, "descriptor": descriptor.to_dict()}, indent=2))
    return 0


def build_detect_response(
    result,
    screen_offset: tuple[int, int],
    *,
    pipeline: str,
    session_id: str = "",
    scale_range: tuple[float, float] | None = None,
    enforce_size_gate: bool = False,
) -> dict:
    accepted_json = [candidate_to_json(candidate, screen_offset) for candidate in result.accepted]
    if pipeline == "scan":
        accepted_json = [item for item in accepted_json if item.get("living")]
    return {
        "ok": True,
        "pipeline": pipeline,
        "sessionId": session_id,
        "scaleCalibration": {
            "status": "locked" if scale_range else "discovering",
            "range": list(scale_range) if scale_range else None,
            "sizeGate": bool(enforce_size_gate),
        },
        "candidateCount": len(result.candidates),
        "acceptedCount": len(result.accepted),
        "elapsedS": round(result.elapsed_s, 4),
        "candidates": accepted_json,
    }


def parse_request_scale_range(value) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
        return (min(low, high), max(low, high))
    return None


def parse_request_tracks(value) -> list[dict]:
    if not value:
        return []
    tracks: list[dict] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        if "trackId" not in entry or "x" not in entry or "y" not in entry:
            continue
        tracks.append(
            {
                "trackId": int(entry["trackId"]),
                "x": int(entry["x"]),
                "y": int(entry["y"]),
            }
        )
        if "scale" in entry:
            tracks[-1]["scale"] = float(entry["scale"])
    return tracks


def build_state_response(
    track_updates: list[dict],
    *,
    session_id: str = "",
    elapsed_s: float = 0.0,
) -> dict:
    return {
        "ok": True,
        "pipeline": "state",
        "sessionId": session_id,
        "trackUpdateCount": len(track_updates),
        "elapsedS": round(elapsed_s, 4),
        "trackUpdates": track_updates,
    }


def run_detect_request(detector: SimpleMobDetector, config: dict, request: dict) -> dict:
    command = str(request.get("cmd", "")).lower()
    mob_name = str(request.get("mob", "")).lower()
    roi = request.get("roi")
    if not mob_name:
        raise ValueError("mob is required")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("roi must be [x,y,width,height]")
    x, y, w, h = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    scale_range = parse_request_scale_range(request.get("scaleRange"))
    enforce_size_gate = bool(request.get("enforceSizeGate", False))
    calibrated = apply_scale_calibration(config, scale_range, enforce_size_gate)
    detector.apply_runtime_config(calibrated)

    frame = capture_region(x, y, w, h)
    session_id = str(request.get("sessionId", ""))

    if command == "state":
        from tracking.state_recognizer import evaluate_track_state_direct, evaluate_track_states

        tracks = parse_request_tracks(request.get("tracks"))
        mode = str(request.get("mode", "")).lower()
        start = time.perf_counter()
        if mode == "direct":
            if len(tracks) != 1:
                raise ValueError("direct state requires exactly one track")
            track = tracks[0]
            scale_hint = track.get("scale")
            track_updates = [
                evaluate_track_state_direct(
                    detector,
                    frame,
                    mob_name,
                    int(track["trackId"]),
                    int(track["x"]),
                    int(track["y"]),
                    offset_x=x,
                    offset_y=y,
                    scale_hint=scale_hint,
                )
            ]
        else:
            track_updates = evaluate_track_states(detector, frame, mob_name, tracks, offset_x=x, offset_y=y)
        elapsed = time.perf_counter() - start
        return build_state_response(track_updates, session_id=session_id, elapsed_s=elapsed)

    if command == "scan":
        result = detector.detect(frame, mob_name)
        return build_detect_response(
            result,
            (x, y),
            pipeline="scan",
            session_id=session_id,
            scale_range=scale_range,
            enforce_size_gate=enforce_size_gate,
        )

    raise ValueError(f"unsupported cmd: {command}")


def write_json_file(path: Path, payload: dict) -> None:
    text = json.dumps(payload, separators=(",", ":"))
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def cmd_serve_ipc(ipc_dir: Path) -> int:
    ipc_dir.mkdir(parents=True, exist_ok=True)
    ready_path = ipc_dir / "ready.json"
    request_path = ipc_dir / "request.json"
    response_path = ipc_dir / "response.json"
    for path in (request_path, response_path):
        if path.exists():
            path.unlink()

    config = load_simple_config()
    detector = SimpleMobDetector(PROJECT_ROOT, config)
    write_json_file(ready_path, {"ok": True, "ready": True, "pipeline": "serve"})

    while True:
        while not request_path.exists():
            time.sleep(0.01)
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            write_json_file(
                response_path,
                {"ok": False, "error": f"invalid json: {exc}", "candidates": []},
            )
            request_path.unlink(missing_ok=True)
            continue
        request_path.unlink(missing_ok=True)
        if str(request.get("cmd", "")).lower() == "shutdown":
            write_json_file(response_path, {"ok": True, "shutdown": True})
            break
        try:
            response = run_detect_request(detector, config, request)
        except Exception as exc:
            response = {"ok": False, "error": str(exc), "candidates": []}
        write_json_file(response_path, response)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    if args.ipc_dir:
        return cmd_serve_ipc(Path(args.ipc_dir))

    config = load_simple_config()
    detector = SimpleMobDetector(PROJECT_ROOT, config)
    sys.stdout.write(json.dumps({"ok": True, "ready": True, "pipeline": "serve"}) + "\n")
    sys.stdout.flush()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {"ok": False, "error": f"invalid json: {exc}", "candidates": []}
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()
            continue
        if str(request.get("cmd", "")).lower() == "shutdown":
            sys.stdout.write(json.dumps({"ok": True, "shutdown": True}) + "\n")
            sys.stdout.flush()
            break
        try:
            response = run_detect_request(detector, config, request)
        except Exception as exc:
            response = {"ok": False, "error": str(exc), "candidates": []}
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    return 0


def cmd_detect_simple(args: argparse.Namespace) -> int:
    output_path = Path(args.output) if args.output else None
    try:
        frame, screen_offset, _ = _load_frame(args)
        config = load_simple_config(Path(args.config_simple) if args.config_simple else None)
        config = apply_scale_calibration(config, args.scale_range, args.enforce_size_gate)
        detector = SimpleMobDetector(PROJECT_ROOT, config)
        result = detector.detect(frame, args.mob.lower())
        accepted_json = [candidate_to_json(candidate, screen_offset) for candidate in result.accepted]
        if args.debug:
            label = Path(args.image).name if args.image else "live_capture"
            debug_root = PROJECT_ROOT / config["debugOutputDir"]
            save_simple_debug_bundle(debug_root, label, frame, result)
        emit_json(
            build_detect_response(
                result,
                screen_offset,
                pipeline="simple",
                session_id=args.session_id or "",
                scale_range=args.scale_range,
                enforce_size_gate=bool(args.enforce_size_gate),
            ),
            output_path,
        )
        return 0
    except Exception as exc:
        emit_json({"ok": False, "error": str(exc), "candidates": []}, output_path)
        return 1


def cmd_fixtures_simple(args: argparse.Namespace) -> int:
    from dataset_runner import main as fixtures_main

    argv = ["--mob", args.mob.lower()]
    if args.fixtures:
        argv.extend(["--fixtures", args.fixtures])
    if args.debug:
        argv.append("--debug")
    if args.rebuild_descriptor:
        argv.append("--rebuild-descriptor")
    return fixtures_main(argv)


def cmd_inspect(args: argparse.Namespace) -> int:
    spr_path = Path(args.spr)
    act_path = Path(args.act)
    if not spr_path.is_absolute():
        spr_path = PROJECT_ROOT / spr_path
    if not act_path.is_absolute():
        act_path = PROJECT_ROOT / act_path
    spr = SprReader(spr_path).load()
    act = ActReader(act_path).load()
    payload = {
        "spr": str(spr_path),
        "act": str(act_path),
        "sprVersion": spr.version,
        "frameCount": spr.frame_count,
        "actionCount": len(act.actions),
        "actions": [{"index": action.index, "frames": len(action.frames)} for action in act.actions],
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple mob recognition")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-simple-descriptor", help="build simple descriptor from SPR/ACT")
    build.add_argument("--mob", required=True)
    build.add_argument("--force", action="store_true")

    detect = sub.add_parser("detect-simple", help="detect mobs using descriptor heatmaps")
    detect.add_argument("--mob", required=True)
    detect.add_argument("--roi", type=parse_roi, help="screen x,y,width,height")
    detect.add_argument("--image", help="existing image path")
    detect.add_argument("--output", help="JSON output file for Python runtime")
    detect.add_argument("--debug", action="store_true")
    detect.add_argument("--config-simple", help="simple config path")
    detect.add_argument("--session-id", default="", help="bot session id for logs/calibration")
    detect.add_argument("--scale-range", type=parse_scale_range, help="locked session scale range as min,max")
    detect.add_argument("--enforce-size-gate", action="store_true", help="enforce strict object size gate")

    fixtures = sub.add_parser("fixtures-simple", help="run screenshot fixture suite with simple detector")
    fixtures.add_argument("--mob", required=True)
    fixtures.add_argument("--fixtures", help="test-fixtures root")
    fixtures.add_argument("--debug", action="store_true")
    fixtures.add_argument("--rebuild-descriptor", action="store_true")

    inspect = sub.add_parser("inspect", help="inspect SPR/ACT files")
    inspect.add_argument("--spr", required=True)
    inspect.add_argument("--act", required=True)

    serve = sub.add_parser("serve", help="persistent detector server for scan/state commands")
    serve.add_argument("--ipc-dir", help="file IPC directory for hidden parent process communication")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "build-simple-descriptor":
        return cmd_build_simple_descriptor(args)
    if args.command == "detect-simple":
        return cmd_detect_simple(args)
    if args.command == "fixtures-simple":
        return cmd_fixtures_simple(args)
    if args.command == "inspect":
        return cmd_inspect(args)
    if args.command == "serve":
        return cmd_serve(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
