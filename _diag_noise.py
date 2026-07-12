"""Quick diagnostic: what is body_work.mean() for each fixture?"""
import json, cv2, numpy as np
from pathlib import Path
from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import palette_heatmap, HeatmapDetector

FIXTURES_DIR = PROJECT_ROOT / "pybot" / "recognition" / "test-fixtures" / "game-screenshots"
DESC_DIR = PROJECT_ROOT / "assets" / "generated_descriptors"
config = json.loads((PROJECT_ROOT / "pybot/recognition/detector/detector_config.json").read_text())

def find_desc(mob_name):
    for subdir in sorted(DESC_DIR.iterdir()):
        if not subdir.is_dir(): continue
        d = subdir / "descriptor.json"
        if d.exists() and subdir.name.replace("_","") == mob_name.lower().replace("_",""):
            return MobDescriptor.load(d)
    return None

detector = HeatmapDetector(config)

for subdir in sorted(FIXTURES_DIR.iterdir()):
    if not subdir.is_dir(): continue
    mob = subdir.name
    desc = find_desc(mob)
    if not desc: continue

    for img in sorted(subdir.glob("*.png")):
        frame = cv2.imread(str(img))
        if frame is None: continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        y1, y2, x1, x2 = detector.playfield_bounds(frame.shape[:2])
        crop_hsv = hsv[y1:y2, x1:x2]
        body_work = palette_heatmap(crop_hsv, desc.body_palette)
        accent_work = palette_heatmap(crop_hsv, desc.accent_colors)
        body_mean = float(np.mean(body_work))
        accent_mean = float(np.mean(accent_work))
        noisy = "NOISY" if body_mean > 0.25 else "clean"
        print(f"{mob:15s} {img.name:25s} body_mean={body_mean:.4f}  accent_mean={accent_mean:.4f}  -> {noisy}")
