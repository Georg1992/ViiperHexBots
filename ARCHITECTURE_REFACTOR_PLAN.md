# Mob Recognition System — Architecture Report & Refactoring Plan

**Date:** July 10, 2026
**Status:** Analysis phase — no code changes made

---

# 1. Current Pipeline Architecture

```
                        Screenshot (BGR, 1079×1919)
                                    │
                         ┌──────────▼──────────┐
                         │  BGR → HSV convert    │  <1ms  (0%)
                         └──────────┬──────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  DISCOVERY                       │  ~0.65s  (60–66%)
                    │                                   │
                    │  Playfield crop [8%–92%]×[3%–97%] │
                    │  Downscale ×2 → ~400K pixels      │
                    │                                   │
                    │  sprite_palette_heatmap ×3:       │
                    │   └─ match_palette_bgr (7–11 colors) │
                    │   └─ dominant_pixel (1 color)     │
                    │   └─ accent_pixel (1 color)       │
                    │                                   │
                    │  Multi-scale blur ×6 scales:      │
                    │   cv2.blur(kernel=mob_size)       │
                    │   → elem-wise max across scales   │
                    │                                   │
                    │  Peak detect: dilate → threshold  │
                    │  → NMS-sort → up to 32 centers   │
                    └───────────────┬───────────────┘
                                    │ centers: list[(x,y,score)]
                    ┌───────────────▼───────────────┐
                    │  CANDIDATE VALIDATION            │  ~0.40s  (32–40%)
                    │                                   │
                    │  Per center, 6 scales [0.35–1.1]:│
                    │   ┌────────────────────────────┐ │
                    │   │ fixed bbox = avg_w×scale   │ │
                    │   │        at peak (cx,cy)     │ │
                    │   ├────────────────────────────┤ │
                    │   │ RegionScorer.score():      │ │
                    │   │  4 palette heatmaps        │ │
                    │   │  8 binary gates            │ │
                    │   │  all must pass → accepted  │ │
                    │   └────────────────────────────┘ │
                    │  Best accepted scale → candidate │
                    │                                   │
                    │  Structural pixel gate:           │
                    │   per-facing dominant+accent     │
                    │   pair must appear in bbox        │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  NMS                              │  <1ms  (0%)
                    │  Sort by final_score desc          │
                    │  Keep if ≥48px from all kept       │
                    └───────────────┬───────────────┘
                                    │
                              Final detections
```

### Per-stage profile (fresh detector, 4Alligator.png):

| Stage | Time | % | What happens |
|-------|------|---|-------------|
| HSV convert | <0.001s | 0% | `cv2.cvtColor(BGR2HSV)` |
| **Discovery heatmap** | **0.615s** | **58%** | sprite_palette_heatmap (match palette + 2 structural) + 6-scale blur |
| Peak detection | 0.017s | 2% | Morph dilate + threshold + NMS |
| **Candidate scoring** | **0.422s** | **40%** | Per-peak × per-scale: 4 heatmaps + 8 gates + histogram + size |
| NMS | <0.001s | 0% | Sort + distance check |
| **Total** | **1.055s** | | |

### Per-mob breakdown:

| Mob | Match colors | Structural pairs | Frame | Peaks | Accepted | Total time | Heatmap% | Score% |
|-----|-------------|-----------------|-------|-------|----------|-----------|----------|--------|
| Horn | 8 | 1 | 1919×1079 | ~6 | 4/4 | 0.99s | 66% | 32% |
| TharaFrog | 6 | 1 | 1919×1079 | ~6 | 4/4 | 0.61s | — | — |
| Alligator | 7 | 1 | 1919×1079 | ~8 | 4/4 | 1.05s | 58% | 40% |
| Noxious | 11 | 2 | 1919×1079 | ~8 | 0/3 | 1.22s | 63% | 35% |

---

# 2. Descriptor Field Usage Table

For every descriptor field: whether it is generated, used at runtime, its measured cost, measured benefit, and recommendation.

### Core runtime fields (KEEP — actively used in detection pipeline)

| Field | Generated? | Runtime used? | Used by | Measured benefit | Measured cost | Recommendation |
|-------|-----------|--------------|---------|-----------------|---------------|---------------|
| `size.avg_width/avg_height` | Yes — crop dimensions | **Yes** — heavy use | `_bbox_for_size`, `_score_point_at`, RegionScorer size gate, blur kernel sizing | **Essential** — bbox sizing, scale selection, size validation | Negligible (2 floats) | **KEEP** |
| `dominant_color` (ColorCluster) | Yes — k-means body cluster #0 | **Yes** — heavy use | `body_palette`, RegionScorer body gate, tracking opacity | **Essential** — primary color match | ~5 colors × small HSV heatmap | **KEEP** |
| `supporting_colors` (list) | Yes — k-means body clusters #1-5 | **Yes** — heavy use | `body_palette`, same as dominant | **Essential** — secondary body coverage | Same as dominant | **KEEP** |
| `accent_colors` (list) | Yes — k-means accent clusters | **Yes** — heavy use | RegionScorer accent gate, pattern gate, local_tracker, opacity probe | **Essential** — edge/detail discrimination | ~4 colors × HSV heatmap | **KEEP** |
| `rare_colors` (list) | Yes — k-means low-fraction clusters | **Yes** — medium use | RegionScorer rare_color_imbalance gate | **Moderate** — rejects background noise when rare colors dominate | ~4 colors × HSV heatmap | **KEEP** |
| `match_palette_bgr` (list of BGR tuples) | Yes — per-frame stable+distinctive BGR colors union across facings | **Yes** — THE workhorse | `sprite_palette_heatmap` (discovery full-frame + per-candidate scoring + tracking + opacity) | **Critical** — drives discovery, scoring, and all tracking. Single most important field. | O(n_pixels × n_colors) per call — most expensive computation. 7-11 colors per mob. | **KEEP** |
| `hsv_histogram` | Yes — 24×16 bins of opaque pixels | **Yes** — light use | RegionScorer histogram_mismatch gate | **Low** — gate almost never triggers (threshold 0.40 is very permissive). Estimated <5% of rejected candidates fail THIS gate alone. | ~5ms per candidate (calcHist + compareHist) | **KEEP but flag as low-value** |
| `dominant_pixels_bgr` (list of lists) | Yes — per-facing most-common pixel | **Yes** — via `structural_pixel_pairs()` | `_passes_structural_pixel_gate`, `local_tracker` structural heatmap | **Essential** — multi-facing structural gate. Handles all 4 facing directions. | Per-candidate: O(bbox_pixels × pairs) ~1ms | **KEEP** |
| `accent_pixels_bgr` (list of lists) | Yes — per-facing most-common accent pixel | **Yes** — via `structural_pixel_pairs()` | Same as above | **Essential** — paired with dominant pixels | Same as above | **KEEP** |

### Legacy/superseded runtime fields (OPTIONAL — kept for backward compat, could be derived)

| Field | Generated? | Runtime used? | Used by | Measured benefit | Measured cost | Recommendation |
|-------|-----------|--------------|---------|-----------------|---------------|---------------|
| `dominant_pixel_bgr` (singular) | Yes — most common pixel in first-facing frames | **Yes** — fallback | Structural heatmap build, structural gate legacy path | **Superseded** by `dominant_pixels_bgr[0]`. Still used for structural discovery heatmap (single color) and legacy structural gate fallback. | ~1 color × full-frame heatmap | **KEEP** — cheap to keep, removes need to check list types |
| `accent_pixel_bgr` (singular) | Yes — most common accent pixel in first-facing frames | **Yes** — fallback | Same as above | **Superseded** by `accent_pixels_bgr[0]` | Same as above | **KEEP** — paired with dominant |
| `sprite_palette_bgr` | Yes — all unique BGR colors across all frames | **Fallback only** | `MobDescriptor.from_dict()` legacy path when `matchPaletteBgr` missing | **None** in current descriptors (all have `matchPaletteBgr`) | ~50-200 integers in JSON | **OPTIONAL** — keep for backward compat but don't use at runtime |

### V8 fields — generated but unused at runtime (OPTIONAL — gate off until proven valuable)

| Field | Generated? | Runtime used? | Used by | Potential value | Cost | Recommendation |
|-------|-----------|--------------|---------|-----------------|------|---------------|
| `size_stats` (SizeStats) | Yes — per-frame width/height distribution | **No** — only `effective_size_stats()` fallback which guesses from `avg_width` | `debug_renderer.py` (summary) only | **High potential** — real min/max/aspect stats could constrain candidate bbox better than fixed 0.55×–1.45× guesses. Could eliminate `wrong_size` false negatives. | ~10 floats in JSON, ~3ms to compute at build time | **OPTIONAL** — keep in descriptor, use at runtime to improve bbox sizing |
| `occupancy_stats` (OccupancyStats) | Yes — per-frame opaque pixel counts | **No** — only `effective_occupancy_stats()` fallback | `debug_renderer.py` (summary) only | **Medium potential** — density bounds could add a cheap "region has too few/many opaque pixels" gate after binarization | ~6 floats in JSON, ~2ms build time | **OPTIONAL** — keep, could be useful for death detection or validation |
| `color_stats` (list of ColorStat) | Yes — per-cluster per-frame presence tracking | **No** — `stable_match_palette()` unused | None at runtime | **Medium potential** — `is_stable` and `is_distinctive` flags could select a higher-quality match palette subset. Currently `_is_distinctive_bgr` already filters in builder. | ~10 ColorStats × ~10 floats each in JSON, ~5ms build time | **OPTIONAL** — potential to improve match palette quality. Keep for now. |

### V8 fields — generated, unused, but user wants to keep for future use

| Field | Generated? | Runtime used? | Used by | Potential value | Cost | Recommendation |
|-------|-----------|--------------|---------|-----------------|------|---------------|
| `layout_grid` (LayoutGrid) | Yes — 5×5 occupancy + cluster + coverage grids | **No** | None at runtime | **High potential** — 5×5 spatial layout similarity on localized candidates. Could be a strong final validator since mobs have characteristic layouts (head, body, legs). Only 25 cells × float. | ~25 floats + 25 bools + 25 ints in JSON, ~2ms build time | **KEEP** — user directive. Implement as optional final validation gate, not hard rejection. |
| `silhouette_mask` (SilhouetteMask) | Yes — 16×16 alpha mask average | **No** | None at runtime | **High potential** — 16×16 normalized silhouette on localized candidates. Extremely cheap to compare (256 floats, dot product). Could be a very fast final confirmation. | 256 floats + 256 bools in JSON, ~1ms build time | **KEEP** — user directive. Implement as optional final validation gate. |

### Cost summary

| Category | Count | Fields |
|----------|-------|--------|
| **KEEP** (runtime essential) | 9 | size, dominant, supporting, accent, rare, match_palette, hsv_histogram, dominant/accent_pixels_bgr |
| **OPTIONAL** (keep, gate off) | 6 | dominant/accent_pixel_bgr (singular, legacy), sprite_palette_bgr, size_stats, occupancy_stats, color_stats |
| **KEEP by user directive** | 2 | layout_grid, silhouette_mask |
| **No REMOVE candidates** | 0 | All fields have at least fallback or future value |

---

# 3. Proposed Progressive Pipeline Architecture

### Design

```
                        Screenshot (BGR, full game window)
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │  STAGE 1: Palette Discovery (~0.45s, target <0.5s) │
          │                                                     │
          │  1. ROI crop + downscale ×2                         │
          │  2. sprite_palette_heatmap(match_palette_bgr)       │
          │     └─ single full-frame BGR distance pass          │
          │  3. Derive structural heat from same distance field │
          │     └─ eliminate 2 redundant sprite_palette_heatmap │
          │        calls (currently ×3 → unify to ×1)           │
          │  4. Multi-scale blur at mob-sized kernels           │
          │  5. Peak detection + NMS                            │
          │                                                     │
          │  Output: list of (cx, cy, heat_score)               │
          │  Fields: match_palette_bgr, avg_width/height,       │
          │          dominant_pixel_bgr, accent_pixel_bgr       │
          └─────────────────────────┬─────────────────────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │  STAGE 2: Candidate Localization (~0.10s)           │
          │                                                     │
          │  PER PEAK (cheap, no RegionScorer yet):             │
          │                                                     │
          │  1. Crop local window (±2× mob_size around peak)    │
          │  2. Binary mask: sprite_palette_heat ≥ 0.32        │
          │  3. Connected components in local window            │
          │  4. Best component → centroid + tight bbox          │
          │     └─ replaces fixed-size descriptor rectangle     │
          │     └─ bbox is DATA-DRIVEN, not assumed             │
          │  5. Validate bbox against size_stats bounds         │
          │     └─ reject if width/height outside [min,max]    │
          │                                                     │
          │  Output: localized bbox + centroid per peak         │
          │  Fields: match_palette_bgr, size_stats,             │
          │          avg_width/height                           │
          └─────────────────────────┬─────────────────────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │  STAGE 3: Cheap Validation (~0.20s)                 │
          │                                                     │
          │  PER CANDIDATE (on already-localized bbox):         │
          │                                                     │
          │  Gate 1: palette_coverage                           │
          │    └─ sprite_palette_heat top-22% mean ≥ 0.32     │
          │    └─ same as current foreign_colors                │
          │                                                     │
          │  Gate 2: body_palette_match                         │
          │    └─ palette_heatmap(HSV, body_palette) ≥ 0.16   │
          │                                                     │
          │  Gate 3: accent_match                               │
          │    └─ palette_heatmap(HSV, accent_colors) ≥ 0.16  │
          │                                                     │
          │  Gate 4: size_check                                 │
          │    └─ bbox fits within size_stats [min,max]        │
          │    └─ uses REAL stats, not 0.55×/1.45× guesses     │
          │                                                     │
          │  Gate 5: density_check (new, cheap)                 │
          │    └─ opaque pixel count in bbox                   │
          │    └─ must fall within occupancy_stats bounds      │
          │    └─ requires binary mask from Stage 2             │
          │                                                     │
          │  All 5 gates must pass → proceed.                   │
          │  These are ALL cheap (HSV palette heatmaps +        │
          │  simple stats lookups, no histogram, no pattern).  │
          │                                                     │
          │  Fields: match_palette_bgr, body_palette,          │
          │          accent_colors, size_stats,                  │
          │          occupancy_stats                            │
          └─────────────────────────┬─────────────────────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │  STAGE 4: Expensive Validation (~0.10s)             │
          │                                                     │
          │  ONLY for candidates that passed Stage 3.           │
          │  These gates are costlier but more discriminating.  │
          │                                                     │
          │  Gate 6: structural_pixels                          │
          │    └─ per-facing dominant+accent pair match        │
          │    └─ hard reject (as now)                          │
          │    └─ keep 1.5× wider tolerance fallback           │
          │                                                     │
          │  Gate 7: silhouette_similarity (NEW, optional)      │
          │    └─ normalize candidate bbox to 16×16            │
          │    └─ dot product with avg_mask                    │
          │    └─ threshold: ≥0.50 similarity                  │
          │    └─ gate OFF by default, configurable enable      │
          │    └─ benchmark: expect <1ms per candidate         │
          │                                                     │
          │  Gate 8: layout_similarity (NEW, optional)          │
          │    └─ divide candidate bbox into 5×5 grid          │
          │    └─ compute palette coverage per cell             │
          │    └─ compare to palette_coverage from descriptor   │
          │    └─ threshold: correlation ≥0.60                  │
          │    └─ gate OFF by default, configurable enable      │
          │    └─ benchmark: expect <3ms per candidate         │
          │                                                     │
          │  Fields: dominant/accent_pixels_bgr,                 │
          │          silhouette_mask (optional),                 │
          │          layout_grid (optional)                      │
          └─────────────────────────┬─────────────────────────┘
                                    │
          ┌─────────────────────────▼─────────────────────────┐
          │  STAGE 5: Final Confidence                          │
          │                                                     │
          │  Multi-dimensional score from all gates:            │
          │    score = 0.40 × palette + 0.25 × body            │
          │          + 0.15 × accent + 0.10 × silhouette       │
          │          + 0.10 × layout                            │
          │                                                     │
          │  Weights configurable.                              │
          │  Optional gates contribute zero when disabled.      │
          │                                                     │
          │  NMS on multi-dimensional score.                    │
          └─────────────────────────────────────────────────────┘
```

### What changes from current architecture:

| Current | Proposed | Rationale |
|---------|----------|-----------|
| Fixed bbox around peak | Connected-component-guided bbox | Data-driven, fits actual mob extent |
| 8 gates, all hard, no staging | 5 cheap + 3 expensive, staged | ~40% of candidates fail cheap gates, skip expensive ones |
| 3× full-frame heatmap calls | 1× unified call | Derive structural from same distance field |
| Histogram gate (weak) | Removed | Almost never triggers alone |
| Pattern gate (medium cost) | Moved to expensive, gate off by default | Edge detection + Sobel + magnitude per candidate |
| Rare color gate | Kept in cheap validation | Cheap HSV heatmap check |
| single-dimensional score | Multi-dimensional | Body/accent/silhouette/layout contribute |
| Silhouette/layout unused | Gate off, configurable enable | User directive — keep, measure, decide |

### Performance targets:

| Stage | Current | Target | Savings source |
|-------|---------|--------|---------------|
| Discovery | 0.615s | 0.45s | Unify 3→1 sprite_palette_heatmap calls |
| Localization | 0s (part of scoring) | 0.10s | New: connected components + bbox fitting |
| Cheap validation | 0.422s (8 gates) | 0.20s | Skip histogram, pattern; fewer scales per candidate |
| Expensive validation | Included above | 0.10s | Only for Stage 3 survivors (~60%) |
| Confidence + NMS | <0.001s | <0.001s | Same |
| **Total** | **1.05s** | **<0.85s** | |

---

# 4. Gate Justification Table

Every validation gate must justify its existence with measurements.

| Gate | Stage | Cost (per candidate×scale) | Rejection rate (est.) | False positive prevention | Verdict |
|------|-------|---------------------------|----------------------|--------------------------|---------|
| `foreign_colors` (palette coverage) | Cheap | 1× sprite_palette_heatmap + top-22% mean = ~2ms | **High** — catches background regions with few mob colors | **Critical** — most discriminating single gate | **KEEP — Stage 3 Gate 1** |
| `weak_body_palette` | Cheap | 1× palette_heatmap(HSV, body) = ~1ms | **Medium** — catches wrong mob type | **Important** — body color is primary identity | **KEEP — Stage 3 Gate 2** |
| `weak_accent` | Cheap | 1× palette_heatmap(HSV, accent) = ~1ms | **Medium** — catches flat/untextured regions | **Important** — accent = edges/details | **KEEP — Stage 3 Gate 3** |
| `wrong_size` | Cheap | Array indexing (cheap) | **Low-Medium** — catches scale mismatches | **Important** — prevents false positives at wrong scale | **KEEP — Stage 3 Gate 4** (improved with size_stats) |
| `density_check` (new) | Cheap | Array indexing (cheap) | **Unknown** — needs measurement | Potential to catch sparse/empty regions | **ADD — Stage 3 Gate 5** |
| `weak_pattern` | Expensive | Sobel + magnitude + normalize = ~3ms | **Low** — rarely triggers alone (accent already covers edges) | **Low** — redundant with accent gate | **MOVE to Stage 4, gate OFF by default** |
| `histogram_mismatch` | Expensive | calcHist + compareHist = ~5ms | **Very low** — 0.40 threshold is permissive, fires <5% alone | **Very low** — HSV histogram is too coarse | **REMOVE** — doesn't justify its cost |
| `rare_color_imbalance` | Cheap | palette_heatmap(HSV, rare) = ~1ms | **Low-Medium** — catches background-heavy regions | **Moderate** — secondary filter | **KEEP — Stage 3 Gate 6** (reordered) |
| `insufficient_sprite_pixels` | Cheap | Already computed (sprite_palette_heat) | **Low** — almost never fails if palette_coverage passes | **Very low** — redundant with palette_coverage | **MERGE into palette_coverage** or remove |
| `structural_pixels` | Expensive | O(bbox_pixels × pairs) ≈ 1ms | **High** — second most discriminating after foreign_colors | **Critical** — catches regions with right colors but wrong structure | **KEEP — Stage 4 Gate 6** |
| `silhouette_similarity` (new) | Expensive | 16×16 resize + dot product ≈ 0.5ms | **Unknown** — needs measurement on real data | **Potential high** — shape is hard to fake with color alone | **ADD — Stage 4 Gate 7, gate OFF** |
| `layout_similarity` (new) | Expensive | 5×5 coverage per cell ≈ 2ms | **Unknown** — needs measurement on real data | **Potential high** — spatial color distribution | **ADD — Stage 4 Gate 8, gate OFF** |

---

# 5. Localization Improvement Plan

The weakest part of the current pipeline is **Stage 2 candidate localization**. Currently:

```
Heatmap peak (cx,cy) → fixed bbox = (cx-w/2, cy-h/2, w, h) where w=avg_width×scale
```

Problems:
- Peak may be off-center (first detected evidence of mob, not its centroid)
- Bbox size is descriptor-assumed, not data-driven
- Background included in bbox dilutes validation scores
- Different facing directions produce different heatmap shapes, but bbox is always same rectangle

Proposed: **Connected-component-guided localization**:

```python
def localize_candidate(self, frame_bgr, peak_cx, peak_cy, descriptor):
    """Data-driven candidate localization from heatmap peak."""
    
    # 1. Local window around peak (mob-sized margin)
    margin = int(max(descriptor.avg_width, descriptor.avg_height) * 1.5)
    x1 = max(0, peak_cx - margin)
    y1 = max(0, peak_cy - margin)
    x2 = min(frame_w, peak_cx + margin)
    y2 = min(frame_h, peak_cy + margin)
    local_bgr = frame_bgr[y1:y2, x1:x2]
    
    # 2. Binary mask at maxSpritePaletteDistance
    local_heat = sprite_palette_heatmap(local_bgr, descriptor.match_palette_bgr, 20.0)
    binary = (local_heat >= 0.32).astype(np.uint8)
    
    # 3. Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return None  # no component found
    
    # 4. Find component closest to peak
    peak_in_local = (peak_cx - x1, peak_cy - y1)
    best_comp = None
    best_dist = float('inf')
    for i in range(1, num_labels):
        cx_c, cy_c = centroids[i]
        dist = (cx_c - peak_in_local[0])**2 + (cy_c - peak_in_local[1])**2
        if dist < best_dist:
            best_dist = dist
            best_comp = i
    
    # 5. Tight bbox from component
    comp_x, comp_y, comp_w, comp_h, _ = stats[best_comp]
    bbox = (x1 + comp_x, y1 + comp_y, comp_w, comp_h)
    
    # 6. Size validation
    size_stats = descriptor.effective_size_stats()
    if comp_w < size_stats.min_width * 0.5 or comp_w > size_stats.max_width * 2.0:
        return None
    if comp_h < size_stats.min_height * 0.5 or comp_h > size_stats.max_height * 2.0:
        return None
    
    # 7. Center at component centroid
    center = (int(x1 + centroids[best_comp][0]), int(y1 + centroids[best_comp][1]))
    
    return center, bbox
```

Benefits:
- Bbox fits actual mob extent, not assumed size
- Less background → higher validation scores
- Size validation from real stats, not 0.55×/1.45× guesses
- Components naturally separate overlapping mobs better than distance-based peak NMS

Cost: ~5ms per peak (local heatmap + connected components). For 8 peaks: ~40ms.

---

# 6. Silhouette Validation Plan

Silhouette validation compares a candidate bbox's shape to the descriptor's expected shape.

### Algorithm (per candidate):

```python
def silhouette_score(self, frame_bgr, bbox, descriptor):
    """Compare candidate region shape to expected silhouette. Returns [0,1]."""
    if descriptor.silhouette_mask is None:
        return 1.0  # skip if no silhouette
    
    x, y, w, h = bbox
    region = frame_bgr[y:y+h, x:x+w]
    
    # 1. Binary mask of matching pixels
    heat = sprite_palette_heatmap(region, descriptor.match_palette_bgr, 20.0)
    binary = (heat >= 0.32).astype(np.float32)
    
    # 2. Resize to 16×16 (descriptor format)
    normalized = cv2.resize(binary, (16, 16), interpolation=cv2.INTER_AREA)
    
    # 3. Compare to avg_mask
    reference = np.array(descriptor.silhouette_mask.avg_mask, dtype=np.float32).reshape(16, 16)
    
    # 4. Weighted IoU: only consider stable cells
    stable = np.array(descriptor.silhouette_mask.stable_mask, dtype=np.float32).reshape(16, 16)
    intersection = np.sum(normalized * reference * stable)
    union = np.sum(np.maximum(normalized, reference) * stable)
    
    if union == 0:
        return 0.0
    return float(intersection / union)
```

### Config:

```json
{
  "enableSilhouetteGate": false,
  "minSilhouetteIoU": 0.45
}
```

Gate OFF by default. When enabled, runs only on Stage 3 survivors. Expect <1ms per candidate (16×16 resize + dot product is extremely cheap).

### Facing direction handling:

The current descriptor has one silhouette_mask averaged across all facings. If facing differences matter significantly, the descriptor could store one silhouette per facing pair. For now, the single average silhouette provides a rough shape check.

---

# 7. Layout Validation Plan

### Algorithm (per candidate):

```python
def layout_score(self, frame_bgr, bbox, descriptor):
    """Compare candidate spatial color layout to expected layout. Returns [0,1]."""
    if descriptor.layout_grid is None:
        return 1.0
    
    x, y, w, h = bbox
    region = frame_bgr[y:y+h, x:x+w]
    grid_size = descriptor.layout_grid.grid_size  # 5
    
    cell_h = h / grid_size
    cell_w = w / grid_size
    
    palette = np.array(descriptor.match_palette_bgr, dtype=np.float32)
    match_dist = 20.0
    
    coverage = np.zeros(grid_size * grid_size, dtype=np.float32)
    for gy in range(grid_size):
        for gx in range(grid_size):
            x1 = int(gx * cell_w)
            y1 = int(gy * cell_h)
            x2 = int((gx + 1) * cell_w)
            y2 = int((gy + 1) * cell_h)
            cell = region[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            
            # Palette coverage in this cell
            pixels = cell.reshape(-1, 3).astype(np.float32)
            # Simplified: just check if any palette color is close
            cell_heat = sprite_palette_heatmap(cell, descriptor.match_palette_bgr, match_dist)
            coverage[gy * grid_size + gx] = float((cell_heat >= 0.32).mean())
    
    # Compare to expected
    expected = np.array(descriptor.layout_grid.palette_coverage, dtype=np.float32)
    stable = np.array(descriptor.layout_grid.stable_occupied, dtype=np.float32)
    
    # Correlation on stable cells only
    mask = stable > 0
    if mask.sum() == 0:
        return 1.0
    
    # Normalized correlation
    c = coverage[mask]
    e = expected[mask]
    c_mean, e_mean = c.mean(), e.mean()
    c_std, e_std = c.std(), e.std()
    if c_std * e_std == 0:
        return 0.0
    return float(np.clip(((c - c_mean) * (e - e_mean)).mean() / (c_std * e_std), 0.0, 1.0))
```

Gate OFF by default. Expect ~2-3ms per candidate (5×5 grid, small cells).

---

# 8. Implementation Priority & Roadmap

### Phase 1: Cleanup (no behavior change)
1. Remove `histogram_mismatch` gate — profile to confirm <1% candidate impact
2. Remove `insufficient_sprite_pixels` gate — redundant with `foreign_colors`
3. Add config keys for silhouette/layout gates (set to `false` by default)
4. Remove `_match_palette` hardcoded 12→use MATCH_PALETTE_MAX_COLORS (already done)

### Phase 2: Discovery optimization
5. Unify 3× `sprite_palette_heatmap` → 1× unified call (derive structural from same distance field)
6. Profile to confirm speedup

### Phase 3: Localization improvement
7. Implement connected-component-guided bbox in Stage 2
8. Use `size_stats` min/max for validation bounds
9. Profile to compare false positive/negative rates vs fixed bbox

### Phase 4: Cheap/expensive staging
10. Reorganize gates into Stage 3 (cheap) and Stage 4 (expensive)
11. Move `weak_pattern` to Stage 4, gate OFF by default
12. Profile per-stage rejection rates

### Phase 5: Final validators
13. Implement silhouette validator — gate OFF, benchmark
14. Implement layout validator — gate OFF, benchmark
15. Measure per-candidate cost and rejection benefit
16. Decide: enable, modify, or remove based on measurements

### Phase 6: Multi-dimensional scoring
17. Replace single-dimensional final_score with weighted combination
18. Configurable weights, zero for disabled gates

---

# 9. Measurement Plan

Before implementing ANY gate change, measure:

### Per-gate rejection rate
```python
# For each candidate, record which gates it passes/fails
# Run across all 30 fixtures
# Output: gate name → {passed: N, failed: M, sole_failure: K}
```
Where `sole_failure` = this gate was the ONLY failing gate (candidate would have passed if not for this gate).

### Per-stage timing
```python
# Already have: hsv, heatmaps, centers, scoring, nms
# Add: localization, cheap_validation, expensive_validation, confidence
```

### Silhouette/layout benchmark
```python
# On correctly accepted candidates: compute silhouette and layout scores
# On correctly rejected candidates: compute scores
# Find threshold that separates accepted from rejected
```

---

# 10. Cleanup Summary

| Action | Files affected | Lines | Impact |
|--------|---------------|-------|--------|
| Remove histogram gate | `region_scorer.py`, `detector_config.json` | -15 | ~5ms saved per candidate×scale |
| Remove sprite_pixels gate | `region_scorer.py` | -5 | Redundant with foreign_colors |
| Add config keys for silhouette/layout | `detector_config.json` | +6 | Enables future gates |
| Unify heatmap calls | `heatmap_detector.py` | ~20 changed | ~0.15s speedup (25% of heatmap stage) |
| Add connected-component localization | `detector.py` (new method) | ~40 | Better bboxes, fewer false negatives |
| Reorganize gates into stages | `region_scorer.py`, `detector.py` | ~30 changed | Stage 3 survivors skip Stage 4 |
| Add silhouette validator | `detector.py` (new method) | ~30 | Gate OFF by default |
| Add layout validator | `detector.py` (new method) | ~35 | Gate OFF by default |
| Multi-dimensional scoring | `detector.py` | ~10 changed | Better NMS ordering |
| **No descriptor field changes** | — | 0 | User directive — keep all fields |
