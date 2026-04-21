# SAM HUNTER
**Free, open-source mobile SAM system detection from satellite imagery.**

Automatically detects, classifies, and geolocates mobile air defense systems (S-300, S-400, BUK-M2/M3) from high-resolution satellite imagery at 50cm/pixel.

## Capabilities
- TEL / FCR / Reload vehicle classification by physical profile
- **Radial deployment pattern analysis** → battery-level intelligence
- **Tire track detection** — distinguishes real 40t vehicles from Russian inflatable decoys
- Sentinel-1 SAR fusion for camouflage penetration
- ~10 second analysis pipeline vs 4-6 hour manual analyst workflow
- Telegram alert when confidence > 85%

## Sample output
```json
{
  "overall_alert_level": "CRITICAL",
  "summary": { "batteries_identified": 2, "tel_count": 6, "decoys_identified": 1 },
  "batteries": [{
    "Battery_ID": "SAM-A1B2C3",
    "Battery_Type": "S-400",
    "Deployment_Pattern": "RADIAL",
    "Confidence_Score": 0.92,
    "Is_Active": true,
    "Center_Coordinates": { "lat": 44.60, "lon": 33.52 }
  }]
}
```

## Quick start
```bash
git clone https://github.com/P-create-sumo/sam-hunter
cd sam-hunter
pip install -r requirements.txt

# Run demo (synthetic scene)
python3 -c "
from sam_hunter.scene_generator import generate_sam_scene
from sam_hunter.sam_hunter import SAMHunter, render_annotated_image

img, gt = generate_sam_scene(scenario='active_battery')
hunter = SAMHunter(resolution_mpp=0.5)
result = hunter.process_image(img, gt, 44.60, 33.52, 'Test Area')
render_annotated_image(img, result).save('output.png')
print(result['overall_alert_level'], result['summary'])
"
```

Output: `output.png` with annotated detections + JSON intelligence report.

## No setup required for demo
- No API keys
- No cloud dependency
- No GPU required (CPU inference on synthetic scenes)

## Telegram alerting
Set environment variables to enable automatic alerts:
```bash
export TELEGRAM_BOT_TOKEN=your_token
export TELEGRAM_CHAT_ID=your_chat_id
```
Alerts fire automatically when battery confidence exceeds 85%.

## License
MIT — free forever
