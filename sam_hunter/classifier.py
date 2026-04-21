"""
POSEIDON-OS / SAM Hunter — Target Classifier
Rilevamento e classificazione di sistemi SAM (S-300/S-400) e TEL
da immagini satellitari ad alta risoluzione (50cm/pixel).

Target classes:
  - TEL (Transporter Erector Launcher): rettangolo 10-15m, testata visibile
  - FIRE_CONTROL_RADAR: parabola/pannello piatto su camion, centro schieramento
  - RELOAD_VEHICLE: camion logistico adiacente a TEL
  - SUPPORT_VEHICLE: veicolo generico di supporto
  - DECOY_INFLATABLE: esca gonfiabile (nessuna traccia pneumatici)
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import uuid
import json
from datetime import datetime


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Detection:
    target_id: str
    target_class: str
    bbox: List[int]           # [x1, y1, x2, y2] in pixel
    confidence: float
    coordinates: dict         # {lat, lon}
    activity_status: str
    has_tire_tracks: bool
    sar_metal_reflection: float  # 0.0-1.0
    is_decoy: bool
    decoy_reason: Optional[str]
    area_m2: float
    aspect_ratio: float
    notes: str = ""

    def to_json(self) -> dict:
        return {
            "Target_ID": self.target_id,
            "Target_Class": self.target_class,
            "Coordinates": self.coordinates,
            "Confidence_Score": round(self.confidence, 3),
            "Activity_Status": self.activity_status,
            "Is_Decoy": self.is_decoy,
            "Decoy_Reason": self.decoy_reason,
            "SAR_Metal_Reflection": round(self.sar_metal_reflection, 3),
            "Has_Tire_Tracks": self.has_tire_tracks,
            "BBox_px": self.bbox,
            "Area_m2": round(self.area_m2, 1),
            "Notes": self.notes
        }


@dataclass
class BatteryAssessment:
    battery_id: str
    confidence: float
    alert_level: str          # LOW / MEDIUM / HIGH / CRITICAL
    battery_type: str         # S-300 / S-400 / BUK / UNKNOWN
    tel_count: int
    radar_count: int
    reload_count: int
    deployment_pattern: str   # RADIAL / LINEAR / DISPERSED / INCOMPLETE
    center_coordinates: dict
    detections: List[Detection]
    is_active: bool
    timestamp: str

    def to_json(self) -> dict:
        return {
            "Battery_ID": self.battery_id,
            "Battery_Type": self.battery_type,
            "Alert_Level": self.alert_level,
            "Confidence_Score": round(self.confidence, 3),
            "Is_Active": self.is_active,
            "Deployment_Pattern": self.deployment_pattern,
            "TEL_Count": self.tel_count,
            "Radar_Count": self.radar_count,
            "Reload_Count": self.reload_count,
            "Center_Coordinates": self.center_coordinates,
            "Timestamp_UTC": self.timestamp,
            "Targets": [d.to_json() for d in self.detections]
        }


# ─────────────────────────────────────────────
# SHAPE ANALYZER
# ─────────────────────────────────────────────

class ShapeAnalyzer:
    """
    Analizza forme geometriche nei blob rilevati per classificarli.
    Alla risoluzione 50cm/pixel:
      - TEL S-300/400: ~24-30px di lunghezza (12-15m)
      - Radar NEBO/TOMBSTONE: ~16-20px (8-10m), forma quasi quadrata
      - Camion logistico: ~14-18px (7-9m)
    """

    # Parametri fisici reali (in metri) per sistema
    SYSTEM_PROFILES = {
        "S-400_TEL": {
            "length_m": (14.0, 16.0), "width_m": (3.0, 3.5),
            "aspect_ratio": (3.8, 5.5), "area_m2": (42, 56)
        },
        "S-300_TEL": {
            "length_m": (12.0, 14.0), "width_m": (2.8, 3.2),
            "aspect_ratio": (3.5, 5.0), "area_m2": (34, 45)
        },
        "30N6E2_RADAR": {
            "length_m": (8.0, 10.0), "width_m": (7.5, 9.5),
            "aspect_ratio": (0.8, 1.4), "area_m2": (60, 95)
        },
        "BUK_TEL": {
            "length_m": (9.0, 10.5), "width_m": (3.0, 3.3),
            "aspect_ratio": (2.8, 3.5), "area_m2": (27, 35)
        },
        "RELOAD_TRUCK": {
            "length_m": (9.0, 12.0), "width_m": (2.5, 3.0),
            "aspect_ratio": (3.0, 4.5), "area_m2": (22, 36)
        }
    }

    def classify_shape(self, bbox: List[int], resolution_mpp: float = 0.5) -> dict:
        """
        Classifica un bounding box basandosi su dimensioni fisiche e aspect ratio.
        resolution_mpp: metri per pixel (0.5 = 50cm/pixel)
        """
        x1, y1, x2, y2 = bbox
        w_px = x2 - x1
        h_px = y2 - y1
        w_m = w_px * resolution_mpp
        h_m = h_px * resolution_mpp

        long_side = max(w_m, h_m)
        short_side = min(w_m, h_m)
        aspect = long_side / max(short_side, 0.1)
        area = w_m * h_m

        candidates = []
        for system, profile in self.SYSTEM_PROFILES.items():
            l_min, l_max = profile["length_m"]
            ar_min, ar_max = profile["aspect_ratio"]
            a_min, a_max = profile["area_m2"]

            length_match = l_min <= long_side <= l_max
            ar_match = ar_min <= aspect <= ar_max
            area_match = a_min <= area <= a_max

            score = sum([length_match, ar_match, area_match]) / 3.0
            if score > 0.33:
                candidates.append((system, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        best = candidates[0] if candidates else ("UNKNOWN", 0.0)

        # Mappa sistema → classe target
        class_map = {
            "S-400_TEL": "TEL", "S-300_TEL": "TEL", "BUK_TEL": "TEL",
            "30N6E2_RADAR": "FIRE_CONTROL_RADAR",
            "RELOAD_TRUCK": "RELOAD_VEHICLE",
            "UNKNOWN": "SUPPORT_VEHICLE"
        }

        return {
            "system": best[0],
            "target_class": class_map.get(best[0], "SUPPORT_VEHICLE"),
            "shape_confidence": best[1],
            "dimensions_m": {"length": round(long_side, 1), "width": round(short_side, 1)},
            "aspect_ratio": round(aspect, 2),
            "area_m2": round(area, 1)
        }


# ─────────────────────────────────────────────
# TIRE TRACK DETECTOR
# ─────────────────────────────────────────────

class TireTrackDetector:
    """
    Analizza l'area attorno a un veicolo rilevato per tracce di pneumatici.
    Un TEL S-400 pesa ~40 tonnellate — lascia solchi profondi nel terreno.
    Un gonfiabile non lascia tracce.

    Metodo: analisi gradiente direzionale nell'area perimetrale del bbox.
    In immagini reali: cerca pattern lineari scuri paralleli (solchi nel terreno).
    Nel POC: usa analisi di texture e varianza dell'immagine.
    """

    def analyze(self, image_array: np.ndarray, bbox: List[int],
                expansion_factor: float = 1.8) -> dict:
        """
        Cerca tracce di pneumatici nell'area espansa attorno al bbox.
        expansion_factor: quanto espandere il bbox per cercare le tracce
        """
        h, w = image_array.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        # Espandi area di ricerca
        margin_x = int(bw * (expansion_factor - 1) / 2)
        margin_y = int(bh * (expansion_factor - 1) / 2)

        rx1 = max(0, x1 - margin_x)
        ry1 = max(0, y1 - margin_y)
        rx2 = min(w, x2 + margin_x)
        ry2 = min(h, y2 + margin_y)

        region = image_array[ry1:ry2, rx1:rx2]
        if region.size == 0:
            return {"has_tracks": False, "track_confidence": 0.0, "reason": "no_region"}

        # Converti in grayscale
        if len(region.shape) == 3:
            gray = np.mean(region, axis=2)
        else:
            gray = region.astype(float)

        # Analisi gradiente — le tracce creano discontinuità lineari
        from scipy import ndimage
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Calcola metriche
        mean_gradient = np.mean(gradient_magnitude)
        variance = np.var(gray)
        dark_pixels = np.sum(gray < np.mean(gray) * 0.7)
        dark_ratio = dark_pixels / max(gray.size, 1)

        # Logica euristica per tracce
        # Terreno con tracce: alta varianza, gradienti direzionali elevati, pixel scuri lineari
        track_score = 0.0
        reasons = []

        if mean_gradient > 15:
            track_score += 0.3
            reasons.append("high_directional_gradient")
        if variance > 200:
            track_score += 0.25
            reasons.append("high_texture_variance")
        if 0.05 < dark_ratio < 0.35:
            track_score += 0.25
            reasons.append("linear_dark_patterns")
        if mean_gradient > 25 and variance > 300:
            track_score += 0.2
            reasons.append("combined_track_signature")

        has_tracks = track_score >= 0.45

        return {
            "has_tracks": has_tracks,
            "track_confidence": round(min(track_score, 1.0), 3),
            "mean_gradient": round(float(mean_gradient), 2),
            "texture_variance": round(float(variance), 2),
            "dark_pixel_ratio": round(float(dark_ratio), 3),
            "indicators": reasons
        }


# ─────────────────────────────────────────────
# SAR FUSION MODULE
# ─────────────────────────────────────────────

class SARFusionModule:
    """
    Confronta immagine ottica con dati SAR per:
    1. Rilevare riflessione metallica (alta backscatter SAR = oggetto metallico reale)
    2. Smascherare esche gonfiabili (nessuna riflessione metallica in SAR)
    3. Rilevare oggetti sotto reti mimetiche

    In produzione: usa Sentinel-1 GRD (Ground Range Detected) in banda C.
    Nel POC: simula i valori SAR basandosi su caratteristiche ottiche.
    """

    def analyze(self, optical_patch: np.ndarray, bbox: List[int]) -> dict:
        """
        Analizza un patch per riflessione metallica simulata.
        In produzione: riceve il corrispondente patch SAR Sentinel-1.
        """
        x1, y1, x2, y2 = bbox
        patch = optical_patch[y1:y2, x1:x2]

        if patch.size == 0:
            return {"sar_metal_reflection": 0.0, "confirmed_metal": False}

        if len(patch.shape) == 3:
            gray = np.mean(patch, axis=2)
        else:
            gray = patch.astype(float)

        # Simula backscatter SAR:
        # Oggetti metallici reali: alta riflettività ottica + alta varianza locale
        # Gonfiabili: superficie omogenea, bassa varianza locale
        mean_brightness = np.mean(gray)
        local_variance = np.var(gray)
        high_reflect_ratio = np.sum(gray > 180) / max(gray.size, 1)

        # Modello semplificato di backscatter
        # In produzione: sigma0 da Sentinel-1 in dB
        simulated_backscatter = (
            0.4 * (mean_brightness / 255.0) +
            0.4 * min(local_variance / 2000.0, 1.0) +
            0.2 * high_reflect_ratio
        )

        confirmed_metal = simulated_backscatter > 0.45

        return {
            "sar_metal_reflection": round(float(simulated_backscatter), 3),
            "confirmed_metal": confirmed_metal,
            "mean_brightness": round(float(mean_brightness), 1),
            "local_variance": round(float(local_variance), 1),
            "high_reflect_ratio": round(float(high_reflect_ratio), 3),
            "note": "Simulated SAR. In production: use Sentinel-1 GRD sigma0 in dB"
        }


# ─────────────────────────────────────────────
# DECOY DETECTOR
# ─────────────────────────────────────────────

class DecoyDetector:
    """
    Combina analisi tracce + SAR per determinare se un oggetto è un'esca.
    Russia impiega sistemi gonfiabili S-300/S-400 scala 1:1 per ingannare ISR.
    """

    def assess(self, track_result: dict, sar_result: dict,
               target_class: str) -> Tuple[bool, Optional[str], float]:
        """
        Ritorna: (is_decoy, reason, adjusted_confidence_penalty)
        """
        if target_class in ["FIRE_CONTROL_RADAR", "SUPPORT_VEHICLE"]:
            # Radar hanno meno probabilità di essere gonfiabili
            return False, None, 0.0

        no_tracks = not track_result.get("has_tracks", False)
        no_metal = not sar_result.get("confirmed_metal", True)
        low_track_conf = track_result.get("track_confidence", 0) < 0.2
        low_sar = sar_result.get("sar_metal_reflection", 1.0) < 0.25

        # Logica decoy
        if no_tracks and no_metal:
            return True, "No tire tracks + No SAR metal reflection — likely inflatable decoy", 0.6
        elif no_tracks and low_sar:
            return True, "No tire tracks detected — possible inflatable (weight ~40t expected)", 0.4
        elif low_track_conf and no_metal:
            return True, "Weak track signature + No metal reflection — decoy probability HIGH", 0.35
        elif no_metal and target_class == "TEL":
            return False, None, 0.15  # Penalità parziale — potrebbe essere sotto copertura
        else:
            return False, None, 0.0

