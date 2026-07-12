"""Compare: full match_palette peaks vs accent+rare only peaks for each fixture."""
import json, cv2, numpy as np
from pathlib import Path
from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import sprite_palette_heatmap, palette_heatmap, HeatmapDetector

FIXTURES_DIR = PROJECT_ROOT / "pybot/recognition/test-fixtures/game-screenshots"
DESC_DIR = PROJECT_ROOT / "assets/generated_descriptors"
config = json.loads((PROJECT_ROOT / "pybot/recognition/detector/detector_config.json").read_text())

def find_desc(mob):
    for subdir in sorted(DESC_DIR.iterdir()):
        if not subdir.is_dir(): continue
        d = subdir / "descriptor.json"
        if d.exists() and subdir.name.replace("_","") == mob.lower().replace("_",""):
            return MobDescriptor.load(d)
    return None

def peaks(heatmap, cfg):
    min_dist = int(cfg["minCenterDistancePx"])
    min_heat = float(cfg["minCenterHeat"])
    peak_rel = float(cfg["peakRelativeThreshold"])
    r = max(3, min_dist // 2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (r*2+1, r*2+1))
    lm = heatmap == cv2.dilate(heatmap, k)
    th = max(float(heatmap.max()) * peak_rel, min_heat)
    ys, xs = np.where(lm & (heatmap >= th))
    if len(xs) == 0:
        return 0
    scores = heatmap[ys, xs]
    order = np.argsort(scores)[::-1]
    kept = []
    md2 = min_dist * min_dist
    for idx in order:
        x, y = int(xs[idx]), int(ys[idx])
        if all((x-px)**2+(y-py)**2 >= md2 for px, py, _ in kept):
            kept.append((x, y, float(scores[idx])))
    return len(kept)

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
        crop = frame[y1:y2, x1:x2]
        crop_hsv = hsv[y1:y2, x1:x2]
        max_dist = float(config["maxSpritePaletteDistance"])

        # Full palette
        full_heat = sprite_palette_heatmap(crop, desc.match_palette_bgr, max_dist)

        # Accent+rare palette
        ar_palette = [(int(c.bgr[0]), int(c.bgr[1]), int(c.bgr[2])) for c in desc.accent_colors]
        for c in desc.rare_colors:
            ar_palette.append((int(c.bgr[0]), int(c.bgr[1]), int(c.bgr[2])))
        ar_heat = sprite_palette_heatmap(crop, ar_palette, max_dist)

        fp = peaks(full_heat, config)
        ap = peaks(ar_heat, config)
        diff = fp - ap
        arrow = " <<<" if diff > 2 else ""
        print(f"{mob:15s} {img.name:25s} full={fp:3d}  accent+rare={ap:3d}  removed={diff:3d}{arrow}")
