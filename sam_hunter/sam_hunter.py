"""
POSEIDON-OS / SAM Hunter — Main Engine
Pipeline completa: immagine → detection → classificazione → pattern analysis → JSON output
"""

import numpy as np
import uuid
import json
import random
import math
from datetime import datetime, timezone
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw

from .classifier import (
    Detection, BatteryAssessment,
    ShapeAnalyzer, TireTrackDetector, SARFusionModule, DecoyDetector
)
from .pattern_analyzer import analyze_sam_batteries
from .scene_generator import generate_sam_scene


class SAMHunter:
    """
    Main engine del modulo SAM Hunter.
    Orchestrates: scene ingestion → object detection → classification
                  → decoy filtering → pattern analysis → intelligence output
    """

    def __init__(self, resolution_mpp: float = 0.5):
        self.resolution_mpp = resolution_mpp
        self.shape_analyzer = ShapeAnalyzer()
        self.track_detector = TireTrackDetector()
        self.sar_fusion = SARFusionModule()
        self.decoy_detector = DecoyDetector()

    def _simulate_detections(self, image_array: np.ndarray,
                              ground_truth: List[dict]) -> List[dict]:
        """
        Simula l'output di YOLOv8 (in produzione: sostituire con vera inference).
        Usa il ground truth per generare detection realistiche con noise.
        """
        detections = []
        for gt in ground_truth:
            # Simula falsi positivi e mancate detection (recall ~0.88)
            if random.random() < 0.12:
                continue  # Missed detection

            bbox = gt["bbox"]
            # Aggiungi jitter al bbox (simula imprecisione del modello)
            jitter = 3
            noisy_bbox = [
                bbox[0] + random.randint(-jitter, jitter),
                bbox[1] + random.randint(-jitter, jitter),
                bbox[2] + random.randint(-jitter, jitter),
                bbox[3] + random.randint(-jitter, jitter),
            ]

            # Confidence: reali ~0.82-0.96, decoy ~0.70-0.88
            if gt["is_real"]:
                conf = round(random.uniform(0.78, 0.96), 3)
            else:
                conf = round(random.uniform(0.65, 0.84), 3)

            detections.append({
                "bbox": noisy_bbox,
                "raw_confidence": conf,
                "is_real_gt": gt["is_real"],   # Solo per validazione POC
                "has_tracks_gt": gt.get("has_tracks", False)
            })

        # Aggiungi 1-2 falsi positivi casuali
        h, w = image_array.shape[:2]
        for _ in range(random.randint(0, 2)):
            fx1 = random.randint(50, w - 80)
            fy1 = random.randint(50, h - 80)
            fw = random.randint(12, 35)
            fh = random.randint(5, 12)
            detections.append({
                "bbox": [fx1, fy1, fx1+fw, fy1+fh],
                "raw_confidence": round(random.uniform(0.45, 0.65), 3),
                "is_real_gt": False,
                "has_tracks_gt": False
            })

        return detections

    def _pixel_to_latlon(self, px: int, py: int,
                          origin_lat: float, origin_lon: float) -> dict:
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * abs(math.cos(math.radians(origin_lat)))
        lat = origin_lat - (py * self.resolution_mpp) / meters_per_deg_lat
        lon = origin_lon + (px * self.resolution_mpp) / meters_per_deg_lon
        return {"lat": round(lat, 6), "lon": round(lon, 6)}

    def process_image(self, image: Image.Image, ground_truth: List[dict],
                      origin_lat: float = 44.95, origin_lon: float = 34.10,
                      area_name: str = "Unknown") -> dict:
        """
        Pipeline completa su un'immagine.
        """
        image_array = np.array(image)
        h, w = image_array.shape[:2]

        # Step 1: Detection (YOLOv8 simulato)
        raw_detections = self._simulate_detections(image_array, ground_truth)

        # Step 2: Classificazione + analisi per ogni detection
        detections: List[Detection] = []

        for raw in raw_detections:
            bbox = raw["bbox"]
            cx = (bbox[0] + bbox[2]) // 2
            cy = (bbox[1] + bbox[3]) // 2

            # Shape analysis
            shape = self.shape_analyzer.classify_shape(bbox, self.resolution_mpp)

            # Tire track analysis
            tracks = self.track_detector.analyze(image_array, bbox)

            # SAR fusion
            sar = self.sar_fusion.analyze(image_array, bbox)

            # Decoy detection
            is_decoy, decoy_reason, conf_penalty = self.decoy_detector.assess(
                tracks, sar, shape["target_class"]
            )

            # Confidence finale
            final_conf = max(raw["raw_confidence"] * shape["shape_confidence"] - conf_penalty, 0.05)

            # Activity status
            if shape["target_class"] == "TEL":
                # Simula status basato su contesto
                statuses = ["Schierato", "Schierato", "Schierato", "In movimento", "In ricarica"]
                status = random.choice(statuses)
            elif shape["target_class"] == "FIRE_CONTROL_RADAR":
                status = "Operativo" if not is_decoy else "Spento"
            else:
                status = "In movimento" if random.random() > 0.6 else "Parcheggiato"

            # Coordinate
            coords = self._pixel_to_latlon(cx, cy, origin_lat, origin_lon)

            det = Detection(
                target_id=f"TGT-{str(uuid.uuid4())[:8].upper()}",
                target_class=shape["target_class"],
                bbox=bbox,
                confidence=round(final_conf, 3),
                coordinates=coords,
                activity_status=status,
                has_tire_tracks=tracks["has_tracks"],
                sar_metal_reflection=sar["sar_metal_reflection"],
                is_decoy=is_decoy,
                decoy_reason=decoy_reason,
                area_m2=shape["area_m2"],
                aspect_ratio=shape["aspect_ratio"],
                notes=f"System match: {shape['system']} | Dims: {shape['dimensions_m']}"
            )
            detections.append(det)

        # Step 3: Pattern analysis → Battery identification
        batteries = analyze_sam_batteries(
            detections, origin_lat, origin_lon,
            self.resolution_mpp, w
        )

        # Step 4: Genera output intelligence
        real_detections = [d for d in detections if not d.is_decoy]
        decoy_detections = [d for d in detections if d.is_decoy]

        max_alert = "LOW"
        for b in batteries:
            level_rank = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            if level_rank.get(b.alert_level, 0) > level_rank.get(max_alert, 0):
                max_alert = b.alert_level

        result = {
            "scan_id": str(uuid.uuid4())[:12].upper(),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "area": area_name,
            "image_size_px": {"width": w, "height": h},
            "resolution_mpp": self.resolution_mpp,
            "overall_alert_level": max_alert,
            "summary": {
                "total_detections": len(detections),
                "real_targets": len(real_detections),
                "decoys_identified": len(decoy_detections),
                "batteries_identified": len(batteries),
                "tel_count": len([d for d in real_detections if d.target_class == "TEL"]),
                "radar_count": len([d for d in real_detections if d.target_class == "FIRE_CONTROL_RADAR"]),
            },
            "batteries": [b.to_json() for b in batteries],
            "individual_targets": [d.to_json() for d in detections],
        }

        return result


def render_annotated_image(image: Image.Image, result: dict) -> Image.Image:
    """
    Disegna bounding boxes, labels e alert sulla scena.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    color_map = {
        "TEL":                (255, 60, 60),
        "FIRE_CONTROL_RADAR": (255, 200, 0),
        "RELOAD_VEHICLE":     (60, 180, 255),
        "SUPPORT_VEHICLE":    (180, 180, 180),
        "DECOY_INFLATABLE":   (255, 255, 255),
    }

    for tgt in result["individual_targets"]:
        bbox = tgt["BBox_px"]
        cls = tgt["Target_Class"]
        conf = tgt["Confidence_Score"]
        is_decoy = tgt["Is_Decoy"]

        color = (150, 150, 150) if is_decoy else color_map.get(cls, (200, 200, 200))
        width = 1 if is_decoy else 2

        draw.rectangle(bbox, outline=color, width=width)

        label = f"{'[DECOY]' if is_decoy else cls[:3]} {conf:.2f}"
        lx, ly = bbox[0], max(0, bbox[1] - 12)
        draw.rectangle([lx, ly, lx + len(label) * 6 + 2, ly + 11], fill=color)
        draw.text((lx + 1, ly + 1), label, fill=(0, 0, 0))

    # Alert banner
    alert = result["overall_alert_level"]
    alert_colors = {"CRITICAL": (220, 30, 30), "HIGH": (220, 120, 0),
                    "MEDIUM": (220, 200, 0), "LOW": (60, 180, 60)}
    banner_color = alert_colors.get(alert, (100, 100, 100))
    draw.rectangle([0, 0, img.width, 22], fill=banner_color)
    summary = result["summary"]
    banner_text = (f"POSEIDON-OS SAM Hunter | {result['area']} | ALERT: {alert} | "
                   f"Batteries: {summary['batteries_identified']} | "
                   f"TEL: {summary['tel_count']} | Decoys: {summary['decoys_identified']}")
    draw.text((6, 4), banner_text, fill=(255, 255, 255))

    return img

