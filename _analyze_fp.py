"""Analyze what separates false-positive blobs from real sprite blobs."""
from pathlib import Path

import cv2
import numpy as np

from pybot.paths import PROJECT_ROOT
from pybot.recognition.detector.descriptors.descriptor import MobDescriptor
from pybot.recognition.detector.scoring.heatmap_detector import (
    HeatmapDetector, sprite_palette_heatmap,
)
from pybot.recognition.detector.detector import MobDetector, load_detector_config
from pybot.recognition.fixtures import MOB_FIXTURE_SUITES, fixture_search_frame

config = load_detector_config()

mspm = float(config['minSpritePaletteMatch'])
max_pd = float(config['maxSpritePaletteDistance'])
min_sim = float(config['minSilhouetteSimilarity'])


print("=" * 130)
hdr = f"{'mob':12s} {'fixture':28s} {'idx':>3s} {'heat':>6s} {'bbox':>12s} {'area':>6s} {'comp_ar':>6s} {'density':>7s} {'pal_frac':>7s} {'dim_min':>7s} {'aspect':>7s} {'sil':>6s}"
print(hdr)
print("=" * 130)

all_rows = []

for suite in MOB_FIXTURE_SUITES:
    mob_dir = suite.mob_name
    desc = MobDescriptor.load(
        PROJECT_ROOT / 'assets' / 'generated_descriptors' / mob_dir / 'descriptor.json'
    )
    desc_avg_aspect = desc.avg_width / max(desc.avg_height, 1)

    detector = MobDetector(PROJECT_ROOT, config)
    heat_det_config = dict(config)
    for k in ('playfieldTopRatio', 'playfieldBottomRatio', 'playfieldLeftRatio', 'playfieldRightRatio'):
        heat_det_config[k] = 0.0 if k.endswith('TopRatio') or k.endswith('LeftRatio') else 1.0
    heat_det = HeatmapDetector(heat_det_config)

    for image in suite.images():
        fix_name = image.path.stem
        img = cv2.imread(str(image.path))
        if img is None:
            continue
        frame = fixture_search_frame(img)
        fh, fw = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        result = detector.detect(frame, mob_dir)
        accepted_centers = set((c.center_x, c.center_y) for c in result.accepted)

        hm, _accent = heat_det.build_sprite_heatmap(frame, hsv, desc)
        blobs = heat_det.top_centers(hm, desc.avg_width, desc.avg_height)
        blobs.sort(key=lambda x: x[2], reverse=True)

        for i, (cx, cy, score, comp_bbox) in enumerate(blobs):
            bx, by, bw, bh = comp_bbox
            blob_area = bw * bh
            if blob_area < 300:
                continue

            region = frame[by:by + bh, bx:bx + bw]
            palette_hm = sprite_palette_heatmap(region, desc.match_palette_bgr, max_pd)
            binary = (palette_hm >= mspm).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.dilate(binary, kernel, iterations=1)
            nl, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

            ref_cx = bw // 2
            ref_cy = bh // 2
            best_overlap = 0
            best_area = 0
            for lbl in range(1, nl):
                sx, sy, sw, sh, area = stats[lbl]
                ol = max(sx, ref_cx - bw // 2)
                ot = max(sy, ref_cy - bh // 2)
                or_ = min(sx + sw, ref_cx + bw // 2)
                ob = min(sy + sh, ref_cy + bh // 2)
                if ol < or_ and ot < ob:
                    oa = (or_ - ol) * (ob - ot)
                    if oa > best_overlap:
                        best_overlap = oa
                        best_area = area
            comp_area = best_area
            density = comp_area / max(blob_area, 1)
            pal_above = int(np.sum(palette_hm >= mspm))
            pal_frac = pal_above / max(blob_area, 1)

            passes_sil = (
                score >= config['minDiscoveryHeatmapScore']
                and detector._passes_silhouette_gate(frame, desc, (bx, by, bw, bh), comp_bbox=comp_bbox)
            )

            _, best_bbox, sim_raw = detector.score_at(frame, hsv, desc, cx, cy)
            sil_accepted = best_bbox is not None and sim_raw >= min_sim

            dim_min_ratio = 0.0
            if desc.size.min_width and desc.size.min_height:
                dim_min_ratio = min(bw / desc.size.min_width, bh / desc.size.min_height)

            blob_aspect = bw / max(bh, 1)
            aspect_diff = abs(blob_aspect - desc_avg_aspect) / max(desc_avg_aspect, 0.01)

            is_accepted = (cx, cy) in accepted_centers

            sim_str = f"{sim_raw:.3f}" if best_bbox is not None else " N/A"

            print(
                f"{mob_dir:12s} {fix_name:28s} {i:3d}"
                f" {score:6.3f}"
                f" {bw:3d}x{bh:<7d}"
                f" {blob_area:6d}"
                f" {comp_area:6d}"
                f" {density:7.3f}"
                f" {pal_frac:7.3f}"
                f" {dim_min_ratio:7.3f}"
                f" {blob_aspect:7.2f}"
                f" {sim_str:>6s}"
            )

            all_rows.append({
                'mob': mob_dir, 'fixture': fix_name, 'blob_idx': i,
                'heat': score, 'area': blob_area, 'density': density,
                'pal_frac': pal_frac, 'dim_min': dim_min_ratio,
                'aspect': blob_aspect, 'aspect_diff': aspect_diff,
                'accepted': is_accepted, 'sim': sim_raw if best_bbox is not None else 0.0,
            })

# ============ ANALYSIS ============
print("\n\n" + "=" * 130)
print("NOXIOUS — per-fixture analysis (find the FP blob)")
print("=" * 130)
for fx in sorted(set(r['fixture'] for r in all_rows if r['mob'] == 'noxious')):
    fx_rows = [r for r in all_rows if r['mob'] == 'noxious' and r['fixture'] == fx]
    acc = [r for r in fx_rows if r['accepted']]
    rej = [r for r in fx_rows if not r['accepted']]
    print(f"\n{fx}: {len(acc)} acc, {len(rej)} rej")
    for r in sorted(fx_rows, key=lambda r: -r['area']):
        m = " OK" if r['accepted'] else " FP" if rej and r['area'] < min(a['area'] for a in acc) else " rej"
        print(f"  area={r['area']:5d}  heat={r['heat']:.3f}  density={r['density']:.3f}  "
              f"pal_frac={r['pal_frac']:.3f}  dim_min={r['dim_min']:.3f}  "
              f"aspect_diff={r['aspect_diff']:.3f}  sim={r['sim']:.3f}  {m}")

print("\n" + "=" * 130)
print("ALL MOBS — summary by mob")
print("=" * 130)
for mob in sorted(set(r['mob'] for r in all_rows)):
    mob_rows = [r for r in all_rows if r['mob'] == mob]
    acc = [r for r in mob_rows if r['accepted']]
    rej = [r for r in mob_rows if not r['accepted']]
    if acc:
        avg_acc = {k: np.mean([r[k] for r in acc]) for k in ['area','density','pal_frac','dim_min','aspect_diff','sim','heat']}
        print(f"\n{mob:15s} {len(acc)} accepted avg:   area={avg_acc['area']:.0f}  heat={avg_acc['heat']:.3f}  density={avg_acc['density']:.3f}  "
              f"pal_frac={avg_acc['pal_frac']:.3f}  dim_min={avg_acc['dim_min']:.3f}  aspect_diff={avg_acc['aspect_diff']:.3f}  sim={avg_acc['sim']:.3f}")
    if rej:
        avg_rej = {k: np.mean([r[k] for r in rej]) for k in ['area','density','pal_frac','dim_min','aspect_diff','sim','heat']}
        print(f"               {len(rej)} rejected avg:  area={avg_rej['area']:.0f}  heat={avg_rej['heat']:.3f}  density={avg_rej['density']:.3f}  "
              f"pal_frac={avg_rej['pal_frac']:.3f}  dim_min={avg_rej['dim_min']:.3f}  aspect_diff={avg_rej['aspect_diff']:.3f}  sim={avg_rej['sim']:.3f}")

print("\n" + "=" * 130)
print("NOXIOUS 3Noxious_Gray — FP vs correct")
print("=" * 130)
nox3 = [r for r in all_rows if r['mob'] == 'noxious' and r['fixture'] == '3Noxious_Gray']
sorted_nox3 = sorted(nox3, key=lambda r: r['area'])
if len(sorted_nox3) >= 2:
    fp = sorted_nox3[0]
    correct = sorted_nox3[1:]
    print(f"\nFP  (area={fp['area']}):    den={fp['density']:.3f}  pal_frac={fp['pal_frac']:.3f}  "
          f"dim_min={fp['dim_min']:.3f}  aspect_diff={fp['aspect_diff']:.3f}  sim={fp['sim']:.3f}")
    avg_c = {k: np.mean([r[k] for r in correct]) for k in ['area','density','pal_frac','dim_min','aspect_diff','sim']}
    print(f"Correct (n={len(correct)}): den={avg_c['density']:.3f}  pal_frac={avg_c['pal_frac']:.3f}  "
          f"dim_min={avg_c['dim_min']:.3f}  aspect_diff={avg_c['aspect_diff']:.3f}  sim={avg_c['sim']:.3f}")
    print(f"\nFP vs Correct ratios:")
    print(f"  area:       {fp['area']/avg_c['area']*100:.0f}%")
    print(f"  density:    {fp['density']/avg_c['density']*100:.0f}%")
    print(f"  pal_frac:   {fp['pal_frac']/avg_c['pal_frac']*100:.0f}%")
    print(f"  dim_min:    {fp['dim_min']/avg_c['dim_min']*100:.0f}%")
    print(f"  aspect_diff: {fp['aspect_diff']/avg_c['aspect_diff']*100:.0f}%")
    print(f"  sim:        {fp['sim']/avg_c['sim']*100:.0f}%")
