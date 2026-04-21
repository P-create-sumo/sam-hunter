# SAM HUNTER — Testing Guide

## Requirements
- Python 3.10+
- pip install -r requirements.txt
- No GPU required for demo
- No API keys required for demo

## Run demo
```bash
python3 -c "
from sam_hunter.scene_generator import generate_sam_scene
from sam_hunter.sam_hunter import SAMHunter, render_annotated_image

img, gt = generate_sam_scene(scenario='active_battery', seed=42)
hunter = SAMHunter(resolution_mpp=0.5)
result = hunter.process_image(img, gt, 44.60, 33.52, 'Crimea — Sevastopol')
render_annotated_image(img, result).save('output.png')

print('Alert level:', result['overall_alert_level'])
print('Batteries:', result['summary']['batteries_identified'])
print('TELs:', result['summary']['tel_count'])
print('Decoys caught:', result['summary']['decoys_identified'])
"
```

## Scenarios available
- `active_battery` — active S-400 deployment with FCR + TELs
- `dispersed` — dispersed pattern, lower confidence
- `decoy_mix` — mix of real vehicles and inflatable decoys

## With real imagery
Replace synthetic scene with real Sentinel-2 image:
```python
from PIL import Image
img = Image.open("your_sentinel2_image.png").convert("RGB")
result = hunter.process_image(img, [], lat, lon, "Area Name")
```
