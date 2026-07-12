# Technical Audit: Mob Recognition System
**Date:** July 10, 2026
**Codebase state:** Commit `c9edf0e` (post v8 recovery)

---

# 1. Overall Pipeline

```
                        Screenshot (BGR, full game window)
                                    │
                         ┌──────────▼──────────┐
                         │   BGR → HSV convert  │  <1ms
                         └──────────┬──────────┘
                                    │
                         ┌──────────▼──────────┐
                         │    ROI crop to       │
                         │   playfield bounds   │  0ms (done inside heatmap)
                         │  (8%–92% H, 3%–97%W)│
                         └──────────┬──────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  Stage 1: Heatmap Build       │  ~0.60s (58%)
                    │  ┌─────────────────────────┐  │
                    │  │ sprite_palette_heatmap  │  │  1 full-frame BGR distance pass
                    │  │  (match_palette_bgr)    │  │
                    │  ├─────────────────────────┤  │
                    │  │ sprite_palette_heatmap  │  │  ×2 more (dominant + accent
                    │  │  (dominant_pixel_bgr)   │  │   structural pixels)
                    │  ├─────────────────────────┤  │
                    │  │ Multi-scale Box Blur    │  │  6 scales × ~25ms each
                    │  │  (avg_width × scale)    │  │
                    │  └─────────────────────────┘  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  Stage 2: Peak Detection       │  ~17ms (2%)
                    │  ┌─────────────────────────┐  │
                    │  │ Morphological Dilate     │  │  local maxima
                    │  │ Threshold + NMS-sort     │  │  peak_relative_threshold=0.18
                    │  └─────────────────────────┘  │
                    │  Returns: list of (x,y,score) │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  Stage 3: Candidate Scoring    │  ~0.42s (40%)
                    │  For each peak:                │
                    │  ┌─────────────────────────┐  │
                    │  │ Try 6 scales (0.35–1.1) │  │
                    │  │ For each scale:          │  │
                    │  │  ┌─────────────────────┐ │  │
                    │  │  │ RegionScorer.score() │ │  │  4 palette heatmaps per
                    │  │  │  8 gates evaluated   │ │  │  candidate per scale
                    │  │  │  (see Section 5)     │ │  │
                    │  │  └─────────────────────┘ │  │
                    │  │ Best accepted → candidate│  │
                    │  └─────────────────────────┘  │
                    │  Structural pixel gate pass    │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  Stage 4: NMS                  │  <1ms
                    │  Sort by final_score desc      │
                    │  Keep if center-distance²      │
                    │   ≥ 48² px from all kept       │
                    └───────────────┬───────────────┘
                                    │
                              Final detections
```

---

# 2. Descriptor

The `MobDescriptor` is the offline-built reference model for one mob type. It is generated by `DescriptorBuilder` from `.spr` and `.act` game asset files.

### Core fields (used by runtime detection)

| Field | Source | Used by | Purpose | Verdict |
|-------|--------|---------|---------|---------|
| `size.avg_width / avg_height` | Frame crop dimensions | `_bbox_for_size`, `_score_point_at` | Bounding box sizing | **Essential** |
| `dominant_color` (ColorCluster) | k-means on body pixels | `palette_heatmap`, `body_palette`, RegionScorer gates | Primary color match in HSV | **Essential** |
| `supporting_colors` (list) | k-means body colors after dominant | `body_palette`, `palette_heatmap`, RegionScorer | Secondary body color coverage | **Essential** |
| `accent_colors` (list) | k-means on high-contrast/saturation pixels | `palette_heatmap`, RegionScorer accent gate, local_tracker | Edge/detail color match | **Essential** |
| `rare_colors` (list) | k-means low-fraction clusters | `palette_heatmap`, rare_color_imbalance gate | Reject background noise | **Essential** |
| `match_palette_bgr` (list of BGR tuples) | Intersection of per-frame colors across all facings | `sprite_palette_heatmap` (full-frame and per-candidate) | Exact BGR pixel coverage | **Essential** — the workhorse of discovery |
| `sprite_palette_bgr` | All unique colors in all frames | Legacy fallback for `match_palette_bgr` | Not used directly in current pipeline | **Dead weight** — only used as fallback when `matchPaletteBgr` is missing from JSON |
| `hsv_histogram` | 24×16 HSV histogram of all opaque pixels | `_histogram_correlation` gate in RegionScorer | HSV distribution match | **Low value** — weak gate (0.40 threshold), often passes easily |
| `dominant_pixel_bgr` (single list[int]) | Most common pixel in first-facing frames | `build_heatmaps` structural heatmap, `_passes_structural_pixel_gate` fallback | Exact BGR pixel check | **Used** — but superseded by `dominant_pixels_bgr` |
| `accent_pixel_bgr` (single list[int]) | Most common accent pixel in first-facing frames | Same as above | Exact BGR pixel check | **Used** — but superseded by `accent_pixels_bgr` |

### v8-added fields (experimental, not used by runtime detection)

| Field | Source | Used by runtime? | Purpose | Verdict |
|-------|--------|-----------------|---------|---------|
| `dominant_pixels_bgr` (list of list[int]) | Per-facing most-common dominant pixels | **Yes** — `structural_pixel_pairs()` in detector + tracker | Multi-facing structural gate | **Essential** — handles all 4 facing directions |
| `accent_pixels_bgr` (list of list[int]) | Per-facing most-common accent pixels | **Yes** — same as above | Multi-facing structural gate | **Essential** |
| `size_stats` (SizeStats) | Per-frame width/height stats across all facings | **No** — only `effective_size_stats()` called, which falls back to estimates from `size` | Detailed size distribution | **Unused** at runtime — only used by builder |
| `occupancy_stats` (OccupancyStats) | Per-frame opaque pixel counts and density | **No** — `effective_occupancy_stats()` falls back to estimates | Density bounds | **Unused** at runtime |
| `color_stats` (list of ColorStat) | Per-color per-frame presence tracking | **No** — `stable_match_palette()` uses it but is never called in the current pipeline | Color stability metadata | **Unused** at runtime — was used by deleted v8 pipeline |
| `layout_grid` (LayoutGrid) | 5×5 occupancy grid averaged across frames | **No** | Layout similarity gate | **Dead code** — was used by deleted universal_validator |
| `silhouette_mask` (SilhouetteMask) | 16×16 silhouette averaged across frames | **No** | Silhouette similarity gate | **Dead code** — was used by deleted universal_validator |

### Key observations:
- **7 of 18 fields are unused at runtime** (`size_stats`, `occupancy_stats`, `color_stats`, `layout_grid`, `silhouette_mask`, `sprite_palette_bgr` as separate from `match_palette_bgr`, `dominant/accent_pixel_bgr` as superseded by list versions)
- `match_palette_bgr` is the single most important field — it drives both discovery heatmaps and region scoring
- The singular `dominant/accent_pixel_bgr` fields duplicate the first element of `dominant/accent_pixels_bgr`

---

# 3. Discovery

### How the discovery heatmap is built

The discovery stage lives entirely in `HeatmapDetector.build_heatmaps()`:

**Step 1: ROI crop** — Extract playfield: rows `[8%:92%]`, cols `[3%:97%]`. Downscale ×2 if `min(width,height) ≥ 1600`.

**Step 2: Full-frame BGR distance pass** — `sprite_palette_heatmap(work_bgr, descriptor.match_palette_bgr, max_distance=20.0)`

This is the most expensive operation. It computes, for every pixel in the cropped frame, the minimum L2 distance to any color in `match_palette_bgr`. The algorithm:
- Flattens the frame to N×3 pixels
- Iterates over `match_palette_bgr` in chunks of 128 colors
- For each chunk: `diff = pixels[:,None,:] - chunk[None,:,:]` → `dist² = sum(diff², axis=2)` → `min_dist² = min(chunk_min, current_min)`
- Converts: `heat = 1.0 - sqrt(min_dist²) / 20.0`
- Clips to `[0, 1]`

Complexity: **O(frame_pixels × palette_size)**. For a 4Alligator.png frame (1079×1919 ≈ 2M pixels, cropped to ~1.6M, downscaled ×2 to ~400K, with ~7 palette colors): ~2.8M distance computations.

**Step 3: Structural heatmaps** — Two more `sprite_palette_heatmap` calls for `dominant_pixel_bgr` (single color, distance=20.0) and `accent_pixel_bgr` (single color, distance=35.0). These are fast (single color each).

**Step 4: Multi-scale blur** — For each of 6 center scales (0.35, 0.45, 0.65, 0.85, 0.95, 1.1):
- Compute Gaussian box blur kernel: `(avg_width × scale, avg_height × scale)` — roughly mob-sized
- Blur the sprite heatmap → `cv2.blur(heat, window)`
- Element-wise maximum across all scales
- Same for structural heatmap (blur `sqrt(dominant × accent)`)

The multi-scale blur is the second most expensive part — 6 blurs on a ~400K pixel array, each with a kernel up to ~80×80 pixels.

**Step 5: Peak detection** — `HeatmapDetector.top_centers()`:
- Dilate with (17×17) kernel to find local maxima
- Threshold: `max(heatmap_max × 0.18, 0.015)`
- Greedy NMS: sort by score descending, keep if ≥32px from all previously kept
- Max 32 centers returned

**How many full-frame passes?** 3: one for `match_palette_bgr`, two for structural pixels. But the structural ones are on single colors so they're cheap.

**Where CPU time is spent:** 58% in heatmap building (mainly the `sprite_palette_heatmap` call + 6-scale blur), 40% in candidate scoring (which also calls `sprite_palette_heatmap` per candidate).

---

# 4. Candidate Generation

Candidates are **not** generated from connected components or mask blobs. Instead:

1. **Peaks → centers**: Each heatmap peak (x, y, score) becomes a candidate center.
2. **Bounding boxes are descriptor-sized**: For each peak, the detector generates a bbox by centering `descriptor.avg_width × scale` and `descriptor.avg_height × scale` at the peak point.
3. **Multiple scales tried**: 6 scales [0.35, 0.45, 0.65, 0.85, 0.95, 1.1] are tried per peak. The first accepted scale is used; if score ≥ 0.30, early-exit.
4. **Size statistics**: The `SizeDescriptor.min_width/max_width/min_height/max_height` fields exist but are **not used** for candidate generation. Only `avg_width` and `avg_height` matter.

### Key assumption: Every mob has roughly the same pixel size as the descriptor's average frame size, scaled by the configured scale factors. There is no data-driven bbox fitting — bboxes are fixed rectangles centered on heatmap peaks.

---

# 5. Validation

Validation happens in `RegionScorer.score()`. Each candidate bbox region is evaluated against **8 gates in fixed order**:

| # | Gate | Formula | Threshold | Fields used | Hard/Scoring | Failure case |
|---|------|---------|-----------|-------------|-------------|--------------|
| 1 | `foreign_colors` | Top-22% mean of `sprite_palette_heat` (BGR distance to `match_palette_bgr`) | ≥0.32 | `match_palette_bgr` | **Hard reject** | Background pixels dominate the bbox |
| 2 | `weak_body_palette` | Top-22% mean of `palette_heatmap(region_hsv, body_palette)` | ≥0.16 | `dominant_color`, `supporting_colors` | **Hard reject** | Mob's body colors not visible |
| 3 | `weak_accent` | Top-12% mean of `palette_heatmap(region_hsv, accent_colors)` | ≥0.16 | `accent_colors` | **Hard reject** | No edge/detail colors |
| 4 | `weak_pattern` | Top-12% mean of `accent×0.75 + body×edge_magnitude` | ≥0.14 | `accent_colors`, `body_palette` | **Hard reject** | Region is flat/no edges |
| 5 | `histogram_mismatch` | HSV histogram correlation (24×16 bins) | ≥0.40 | `hsv_histogram` | **Hard reject** | Color distribution differs |
| 6 | `rare_color_imbalance` | `rare_heat_top_8% ≤ max(body×1.15, 0.05)` | ≤ body×1.15 | `rare_colors`, `body_palette` | **Hard reject** | Too many rare/background colors |
| 7 | `insufficient_sprite_pixels` | Fraction of pixels where `sprite_palette_heat ≥ 0.32` | ≥0.06 | `match_palette_bgr` | **Hard reject** | Too few descriptor-matching pixels |
| 8 | `wrong_size` | Min(width_ratio, height_ratio) product sqrt vs expected | ≥0.34 | `avg_width`, `avg_height` | **Hard reject** | Object is wrong size |

After all gates, there is also the **structural pixel gate** (`_passes_structural_pixel_gate`):
- For each per-facing (dominant, accent) pair: check if the bbox region has ≥1.2% of pixels within distance 14 of the dominant pixel AND ≥1.0% within distance 35 of the accent pixel
- At least one pair must pass
- This is a **hard reject** — candidates failing this are discarded even if all 8 RegionScorer gates pass

### Gate importance ranking:
1. `foreign_colors` (gate 1) — most discriminating
2. `structural_pixel_gate` — second most discriminating, especially for distinguishing mobs from background
3. `weak_body_palette` (gate 2) — catches color mismatches
4. `weak_accent` (gate 3) — catches textureless regions
5. `wrong_size` (gate 8) — catches scale mismatches
6. `insufficient_sprite_pixels` (gate 7) — catches sparse regions
7. `rare_color_imbalance` (gate 6) — rarely triggers alone
8. `histogram_mismatch` (gate 5) — least discriminating, 0.40 is very permissive

---

# 6. Scoring

The **final_score** is simply `sprite_palette_heatmap` top-22% mean, clamped to [0,1]:

```python
final_score = clip(sprite_palette_heat_top22pct, 0.0, 1.0)
```

This is a single-dimensional score. All other scores (`body_palette_score`, `accent_score`, etc.) are recorded for debug but do NOT contribute to `final_score` or to acceptance.

**Acceptance is purely binary** — all 8 gates + structural gate must pass. Score does not rescue a weak candidate; score only matters for NMS tiebreaking among accepted candidates.

### Score's only purpose:
- Early-exit optimization: if `final_score ≥ 0.30`, stop trying more scales
- NMS ranking: accepted candidates sorted by `final_score` descending

---

# 7. Tracking

Tracking is a separate code path (`local_tracker.py`) used for following already-discovered mobs frame-to-frame.

### How tracking works:

1. **Single-scale re-scoring**: Uses `_score_living_only_at()` with one scale (from the track's last known scale hint). This is cheaper than discovery's multi-scale scan.

2. **Center-first check**: Score at the track's last known (x, y). If accepted → hit.

3. **Local peak search**: If center miss, crop a search window (2× `search_radius_px` + mob margin), build a local heatmap (`_build_local_follow_heatmap`), find up to 3 peaks, re-score each. This local heatmap is more expensive than the center check but cheaper than full discovery.

4. **Death detection** (`opacity_probe.py`): If enabled, measures opacity in the bbox region. Computes a weighted score from:
   - 40% informative pixel fraction (BGR match)
   - 35% purity
   - 15% body palette match
   - 10% contrast (std dev of grayscale)

   Calibrates a baseline over 2 samples, then checks if opacity drops below 90% of baseline. One confirm tick confirms death.

5. **Structural pairs in tracking**: Uses `descriptor.structural_pixel_pairs()` (per-facing) to build more selective local heatmaps.

### How tracking affects recognition: It doesn't. Tracking is a post-discovery operation. Discovery always runs on the full frame.

---

# 8. Performance

### 4Alligator.png benchmark (1079×1919, downscale ×2):

| Stage | Time | % |
|-------|------|---|
| BGR→HSV | <0.001s | 0% |
| **Heatmap build** | **0.605s** | **58%** |
| └─ sprite_palette_heatmap ×3 | ~0.30s | |
| └─ multi-scale blur ×6 | ~0.30s | |
| Peak detection | ~0.017s | 2% |
| **Candidate scoring** | **0.422s** | **40%** |
| └─ 4 candidates × up to 6 scales × 4 heatmaps each | | |
| NMS | <0.001s | 0% |
| **Total** | **1.045s** | |

### Why each stage costs what it costs:

- **Heatmap build (58%)**: The `sprite_palette_heatmap` iterates over every pixel's L2 distance to ~7 palette colors (chunked). This is the fundamental bottleneck — it's `O(pixels × colors)`. With downscale ×2 on a 1919×1079 frame, the cropped region is ~1.6M pixels, downscaled to ~400K. The multi-scale blur adds 6 box blurs on a 400K array.

- **Candidate scoring (40%)**: Each candidate runs `RegionScorer.score()` which calls `sprite_palette_heatmap` again on a small bbox region (~22×22 pixels = 484 pixels). With 4 candidates × 6 scales = up to 24 calls, but each call is cheap per-pixel. The `palette_heatmap` calls (HSV-based, per-color-cluster) add up at ~25ms per candidate.

### All mobs average (30 fixtures, 29.61s total):

| Mob | Fixtures | Avg time |
|-----|----------|----------|
| TharaFrog | 8 | 0.61s |
| Horn | 8 | 0.78s |
| Alligator | 8 | 0.97s |
| Noxious | 6 | 1.00s |

Smaller mobs (TharaFrog) are faster because their descriptor has fewer colors and the blur kernels are smaller.

---

# 9. Current Problems

### 1. Two full-frame `sprite_palette_heatmap` calls per discovery
The heatmap build calls `sprite_palette_heatmap` 3 times (match palette + dominant + accent structural). The match palette call and structural calls could share the distance computation by deriving everything from a single pass. This is the single biggest performance bottleneck.

### 2. Per-candidate `sprite_palette_heatmap` re-computation
Region scoring re-computes `sprite_palette_heatmap` on every candidate bbox. Since the bboxes are tiny (22×22), this is cheap per-call but adds up with scale iteration.

### 3. Bboxes are fixed-size descriptor rectangles, not data-driven
Candidates don't fit bboxes to the actual mob extent. The bbox is always `avg_width × scale` by `avg_height × scale` centered on the heatmap peak. This means:
- Overlapping mobs get one bbox that may cover both
- Partially visible mobs get the same size bbox as fully visible ones
- The bbox may include significant background

### 4. Multi-scale brute force
Every peak is tried at all 6 scales. There's no scale prediction from the heatmap response — it's purely brute force.

### 5. No mask-based discovery
The pipeline uses heatmap peaks, not connected components. This means:
- A single large blob in the heatmap can produce multiple peaks even if there's only one mob
- Small mobs that produce weak heatmap responses might be missed entirely

### 6. Singular final score
`final_score` is only `sprite_palette_heat` top-22% mean. Body, accent, pattern, and size scores don't contribute. A candidate with strong body but weak palette match is rejected even if it "looks right."

### 7. Descriptor field bloat
7 of 18 descriptor fields are unused at runtime. They add ~50KB to each descriptor JSON and increase build time but provide zero runtime value.

### 8. No early rejection during heatmap build
Every pixel gets a distance computed even if it's in the UI area or has already been discarded by the playfield crop.

### 9. HSV histogram correlation is weak
The `histogram_mismatch` gate (threshold 0.40, 24×16 bins) fires very rarely. It adds compute but almost never rejects anything.

### 10. Candidate explosion risk
`topCandidateCenters=32` peak limit means up to 32 candidates × 6 scales = 192 scoring calls. With 4 mobs on screen, 28 heatmap peaks could be background noise.

---

# 10. Current Strengths

### 1. The `sprite_palette_heatmap` approach is robust
BGR distance to the match palette works well for both normal-world and gray-world screenshots. It's color-space agnostic (no HSV dependency) for the discovery stage, which is why gray-world works without special handling.

### 2. Multi-scale blur is effective
Blurring the heatmap at mob-sized kernels before peak detection naturally handles mobs at different scales without explicit scale search at discovery time.

### 3. Structural pixel gate prevents false positives
The per-facing dominant/accent pixel check is fast and effective at rejecting background regions that happen to have matching colors.

### 4. Separation of discovery and tracking
Discovery scans the full frame; tracking only searches locally. This is the right architecture — it prevents O(n²) tracking costs.

### 5. Descriptor build pipeline is sound
The SPR+ACT→frame rendering→k-means clustering→match palette extraction pipeline produces good descriptors. The ACT RGBA/BGRA fix in `frame_renderer.py` ensures correct colors.

### 6. Downscaling works well
The ×2 downscale for frames ≥1600px wide reduces heatmap computation by 4× with negligible quality loss.

### 7. Simple, testable unit of work
`RegionScorer.score()` is a pure function: region + descriptor → score + accepted. Easy to test, easy to debug.

---

# 11. Failure Analysis: 3Noxious_Gray.png

**Expected:** 3 Noxious mobs
**Got:** 2 accepted candidates
**Missing:** 1 mob

### Trace:

#### Stage 1 — Heatmap peaks found:
```
sprite peaks: 4 total
  peak[0]: (1033, 594) score=0.5953  ← highest score!
  peak[1]: (582,  441) score=0.2142
  peak[2]: (1026, 658) score=0.1848
  peak[3]: (657,  531) score=0.1810

structural peaks: 2 total
  peak[0]: (1659, 868) score=0.1071
  peak[1]: (903,  391) score=0.0286
```

6 total peaks found (4 sprite + 2 structural). After deduplication (structural peaks too close to sprite peaks are dropped), all 6 enter candidate scoring.

#### Stage 2 — Candidate scoring results:

| Peak | Became candidate? | Outcome |
|------|-------------------|---------|
| (1033,594) score=0.5953 | **NO** | Disappeared during validation |
| (582,441) score=0.2142 | YES → Candidate[0] at (639,527) | Accepted, score=0.6187 |
| (1026,658) score=0.1848 | YES → Candidate[1] at (1083,744) | Accepted, score=0.5413 |
| (657,531) score=0.1810 | **NO** | Disappeared during validation |
| (1659,868) score=0.1071 | **NO** | Below structural threshold or failed validation |
| (903,391) score=0.0286 | **NO** | Below min heat threshold |

#### Stage 3 — Where the missing mob disappears:

The 3rd mob should be peak (1033,594) with the highest heatmap score (0.5953). This peak does NOT become a candidate. Since `_evaluate_discovery_center` returns an empty list, the failure is either:

1. **Region scoring**: `_score_point_at()` returned None — meaning no scale produced an accepted result through all 8 RegionScorer gates, OR
2. **Structural pixel gate**: The region scoring passed but `_passes_structural_pixel_gate()` returned False

**Diagnosis: `validation_fail`** — the heatmap found the mob (peak 0.5953), but validation rejected it. This is not a discovery failure. The mob is visible in the heatmap but doesn't pass the gates.

The likely causes for a high-score peak failing validation:
- The peak is slightly off-center, so the fixed-size bbox includes too much background → `foreign_colors` or `insufficient_sprite_pixels` gate rejects it
- The mob at that location has a different facing direction, and the structural pixel pair doesn't match
- Gray-world color shift pushed the palette match slightly below threshold

#### The other disappearing peak (657,531) score=0.1810:
This peak is close to Candidate[0] at (639,527) — only 18px apart. After Candidate[0] was accepted (shifted from peak (582,441) to (639,527)), this peak likely produced a bbox that overlapped heavily with Candidate[0]'s bbox. But since it disappears BEFORE NMS, it was rejected by validation independently. Likely a duplicate of the same mob detected from a different heatmap peak.

---

# 12. Cleanup Opportunities

Priority-ordered, impact-only, no implementation:

| # | Opportunity | Impact | Effort |
|---|------------|--------|--------|
| 1 | **Unify the duplicate `sprite_palette_heatmap` calls in heatmap build** — compute the BGR distance field once, derive both sprite and structural heatmaps from it | ~30% speedup (0.30s → ~0.20s in heatmap stage) | Medium |
| 2 | **Remove 7 unused descriptor fields** — `size_stats`, `occupancy_stats`, `color_stats`, `layout_grid`, `silhouette_mask`, and merge `dominant/accent_pixel_bgr` into their list versions | Cleaner JSON, faster load, less builder time | Low |
| 3 | **Skip HSV histogram gate** — the `histogram_mismatch` gate (threshold 0.40) almost never triggers. Removing it eliminates one `calcHist` + `compareHist` per candidate × scale | Small CPU saving (~5ms per candidate) | Trivial |
| 4 | **Add scale prediction** — the heatmap peak score already correlates with scale. A 0.5953 peak should try fewer scales than a 0.18 peak. Early-exit at first accepted scale already helps but could be smarter | ~20% speedup in scoring stage | Medium |
| 5 | **Make final_score multi-dimensional** — incorporate body_palette, accent, pattern scores into final ranking instead of only palette match. Accepted candidates with strong body but weak palette would still pass but rank lower | Better NMS ordering, fewer false negatives | Low |
| 6 | **Remove `sprite_palette_bgr` as separate field** — it's only used as a fallback when `matchPaletteBgr` is missing. All current descriptors have `matchPaletteBgr`. | Cleaner JSON | Trivial |
| 7 | **Consider mask-based discovery for small mobs** — a binary mask from `sprite_palette_heatmap ≥ threshold` followed by connected components might find mobs that heatmap peaks miss | Better recall for Noxious gray-world, lower false negative rate | High |
| 8 | **Downscale more aggressively** — current downscale is ×2 when min dimension ≥ 1600. Going to ×4 for ≥1920-wide frames would give ~4× speedup in heatmap stage with minimal quality loss | ~50% speedup for 4Alligator | Trivial |
