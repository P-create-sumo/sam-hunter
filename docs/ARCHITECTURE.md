# SAM HUNTER Architecture

## Pipeline

```
Satellite image (Sentinel-2 optical / Sentinel-1 SAR)
        ↓
  scene_generator.py  →  synthetic scene for testing (or real image input)
        ↓
  classifier.py       →  vehicle type classification (TEL / FCR / Reload / Decoy)
        ↓
  pattern_analyzer.py →  deployment pattern analysis (radial / linear / dispersed)
        ↓
  sam_hunter.py       →  battery-level intelligence fusion + threat scoring
        ↓
  JSON report + annotated image + optional Telegram alert
```

## Vehicle classification
| Class | Description | Physical profile |
|-------|-------------|-----------------|
| TEL | Transporter-Erector-Launcher | 40t, 12-14m long, missile tubes |
| FCR | Fire Control Radar | rotating antenna, 8-10m |
| Reload | Reload vehicle | flatbed, 16m+ |
| Decoy | Inflatable replica | no tire tracks, no SAR reflection |

## Alert levels
| Level | Condition |
|-------|-----------|
| CRITICAL | Confidence > 90%, active radar, radial TEL pattern |
| HIGH | Confidence > 75%, multiple TELs |
| MEDIUM | Confidence 50-75%, partial battery |
| LOW | Isolated vehicles, no pattern |

## Decoy defeat methodology
1. **Tire track analysis** — real 40t vehicles leave visible tracks at 50cm/pixel
2. **SAR cross-check** — metal mass creates radar return; inflatables do not
3. **Thermal consistency** — engine heat signature (if thermal band available)
