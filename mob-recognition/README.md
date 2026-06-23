# Mob Recognition

Simple SPR/ACT-driven detector:

```text
SPR + ACT -> simple descriptor -> screenshot heatmaps -> candidate centers -> fixed windows -> score -> NMS
```

The runtime detector does not use sprite blob proposals, bbox refinement scans, corpse filters, NPC/static texture gates, motion confirmation, or engaged-slot death logic.

## Commands

```powershell
# Build descriptor from assets/mobs/horn/horn.spr + horn.act
.\scripts\build-mob-descriptor.ps1 -Mob horn -Force

# Detect on a screenshot
py -3 mob-recognition\cli.py detect-simple --mob horn --image mob-recognition\test-fixtures\game-screenshots\333.png --debug

# Run fixtures
py -3 mob-recognition\cli.py fixtures-simple --mob horn --debug
```

## Runtime

| Module | Role |
|--------|------|
| `simple/descriptor_builder.py` | Render stand/walk ACT frames and build compact descriptor |
| `simple/descriptor.py` | Serializable descriptor model |
| `simple/heatmap_detector.py` | Body/accent/rare/local-pattern heatmaps and center peaks |
| `simple/region_scorer.py` | Fixed-window score breakdown |
| `simple/detector.py` | Descriptor load, heatmaps, centers, fixed-window scoring, NMS |
| `simple/dataset_runner.py` | Fixture evaluation |
| `simple/debug_renderer.py` | Heatmap/candidate/debug image output |
| `simple/config_simple.json` | Simple detector thresholds and weights |

Allowed shared support:

| Module | Role |
|--------|------|
| `spr_reader.py` | Decode `.spr` |
| `act_reader.py` | Decode `.act` |
| `frame_renderer.py` | Compose ACT frame layers |
| `capture.py` | Optional screenshot capture for CLI ROI |

## Descriptor Output

```text
generated_descriptors/<mob>/simple/
  descriptor.json
```

The bot runtime never reads `.spr` or `.act` files. It lists available mobs from `generated_descriptors/*/simple/descriptor.json`; adding a descriptor adds that mob to the UI on the next bot launch.

## Debug Output

```text
mob-recognition/debug/simple/<mob>/<image>/
  input.png
  body_palette_heatmap.png
  accent_heatmap.png
  rare_color_heatmap.png
  local_pattern_heatmap.png
  final_center_heatmap.png
  candidate_centers.png
  detected_overlay.png
  candidates.json
  timing.json
```
