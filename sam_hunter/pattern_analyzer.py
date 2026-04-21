"""
POSEIDON-OS / SAM Hunter — Deployment Pattern Analyzer
Analisi del "Pattern di Schieramento" SAM.

Un TEL isolato è un avvistamento. Quattro TEL disposti radialmente
attorno a un radar di fuoco è una batteria S-400 attiva.

Logica:
  1. Trova cluster spaziali di Detection
  2. Verifica composizione (TEL + RADAR + RELOAD)
  3. Analizza geometria di schieramento (radiale vs lineare)
  4. Assegna alert level e confidence
"""

import numpy as np
import math
import uuid
from datetime import datetime, timezone
from typing import List, Tuple, Optional
from .classifier import Detection, BatteryAssessment


# ─────────────────────────────────────────────
# SPATIAL CLUSTERING
# ─────────────────────────────────────────────

def pixel_distance(d1: Detection, d2: Detection) -> float:
    cx1 = (d1.bbox[0] + d1.bbox[2]) / 2
    cy1 = (d1.bbox[1] + d1.bbox[3]) / 2
    cx2 = (d2.bbox[0] + d2.bbox[2]) / 2
    cy2 = (d2.bbox[1] + d2.bbox[3]) / 2
    return math.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)


def cluster_detections(detections: List[Detection],
                        max_distance_px: float = 300.0) -> List[List[Detection]]:
    """
    Raggruppa detections spazialmente vicine (DBSCAN semplificato).
    max_distance_px: raggio massimo del cluster (300px @ 50cm = 150m)
    """
    if not detections:
        return []

    assigned = [False] * len(detections)
    clusters = []

    for i, det in enumerate(detections):
        if assigned[i]:
            continue
        cluster = [det]
        assigned[i] = True
        for j, other in enumerate(detections):
            if assigned[j] or i == j:
                continue
            if pixel_distance(det, other) <= max_distance_px:
                cluster.append(other)
                assigned[j] = True
        clusters.append(cluster)

    return clusters


# ─────────────────────────────────────────────
# GEOMETRY ANALYZER
# ─────────────────────────────────────────────

def analyze_deployment_geometry(cluster: List[Detection]) -> dict:
    """
    Analizza la geometria di un cluster per determinare se è un
    deployment SAM valido.

    Pattern SAM tipici:
      RADIAL: TEL disposti a raggiera attorno al radar centrale
      LINEAR: TEL in linea (meno comune, posizione di transito)
      DISPERSED: Dispersione tattica (area boschiva)
    """
    tels = [d for d in cluster if d.target_class == "TEL"]
    radars = [d for d in cluster if d.target_class == "FIRE_CONTROL_RADAR"]
    reloads = [d for d in cluster if d.target_class == "RELOAD_VEHICLE"]

    if not tels:
        return {"pattern": "INCOMPLETE", "score": 0.1, "reason": "No TEL detected"}

    centers = []
    for d in cluster:
        cx = (d.bbox[0] + d.bbox[2]) / 2
        cy = (d.bbox[1] + d.bbox[3]) / 2
        centers.append((cx, cy))

    cx_mean = np.mean([c[0] for c in centers])
    cy_mean = np.mean([c[1] for c in centers])

    # Analisi radiale: i TEL dovrebbero essere equidistanti dal radar
    if radars:
        radar = radars[0]
        radar_cx = (radar.bbox[0] + radar.bbox[2]) / 2
        radar_cy = (radar.bbox[1] + radar.bbox[3]) / 2

        tel_distances = []
        for tel in tels:
            tel_cx = (tel.bbox[0] + tel.bbox[2]) / 2
            tel_cy = (tel.bbox[1] + tel.bbox[3]) / 2
            dist = math.sqrt((tel_cx - radar_cx)**2 + (tel_cy - radar_cy)**2)
            tel_distances.append(dist)

        if tel_distances:
            dist_mean = np.mean(tel_distances)
            dist_std = np.std(tel_distances)
            radial_uniformity = 1.0 - min(dist_std / max(dist_mean, 1), 1.0)

            if radial_uniformity > 0.65 and len(tels) >= 3:
                pattern = "RADIAL"
                geo_score = radial_uniformity
            else:
                pattern = "DISPERSED"
                geo_score = 0.5
        else:
            pattern = "INCOMPLETE"
            geo_score = 0.2
    else:
        # Nessun radar — verifica allineamento lineare dei TEL
        if len(tels) >= 2:
            tel_centers = [(((d.bbox[0]+d.bbox[2])//2), ((d.bbox[1]+d.bbox[3])//2)) for d in tels]
            xs = [c[0] for c in tel_centers]
            ys = [c[1] for c in tel_centers]
            # Collinearità approssimativa
            if len(xs) > 2:
                coeffs = np.polyfit(xs, ys, 1)
                residuals = [abs(y - (coeffs[0]*x + coeffs[1])) for x, y in zip(xs, ys)]
                linearity = 1.0 - min(np.mean(residuals) / 50.0, 1.0)
                pattern = "LINEAR" if linearity > 0.7 else "DISPERSED"
                geo_score = linearity * 0.6  # Senza radar, score ridotto
            else:
                pattern = "INCOMPLETE"
                geo_score = 0.2
        else:
            pattern = "INCOMPLETE"
            geo_score = 0.15

    return {
        "pattern": pattern,
        "geometry_score": round(geo_score, 3),
        "tel_count": len(tels),
        "radar_count": len(radars),
        "reload_count": len(reloads),
        "cluster_center": {"x": round(cx_mean, 1), "y": round(cy_mean, 1)}
    }


# ─────────────────────────────────────────────
# BATTERY IDENTIFIER
# ─────────────────────────────────────────────

def identify_battery_type(tel_count: int, radar_count: int,
                            tel_dimensions: List[dict]) -> Tuple[str, float]:
    """
    Identifica il tipo di batteria SAM basandosi sulla composizione e dimensioni.
    S-400 Triumf: 4-8 TEL + 1-2 radar 92N6E + 1-2 radar 96L6
    S-300 PMU2:   4-6 TEL + 1 radar 30N6E2
    BUK-M3:       4-6 TEL + 1 radar 9S36M
    """
    avg_tel_length = np.mean([d.get("length", 12) for d in tel_dimensions]) if tel_dimensions else 12.0

    if avg_tel_length >= 13.5 and tel_count >= 3:
        return "S-400 Triumf", 0.78
    elif 11.0 <= avg_tel_length < 13.5 and radar_count >= 1:
        return "S-300 PMU2", 0.72
    elif avg_tel_length < 11.0 and tel_count >= 3:
        return "BUK-M3", 0.65
    elif tel_count >= 4:
        return "S-300/S-400 (unconfirmed)", 0.55
    else:
        return "UNKNOWN SAM System", 0.35


# ─────────────────────────────────────────────
# MAIN PATTERN ANALYZER
# ─────────────────────────────────────────────

def analyze_sam_batteries(detections: List[Detection],
                           image_origin_lat: float = 44.95,
                           image_origin_lon: float = 34.10,
                           resolution_mpp: float = 0.5,
                           image_width_px: int = 1280) -> List[BatteryAssessment]:
    """
    Analisi completa: raggruppa detections → identifica batterie → genera alert.

    image_origin_lat/lon: coordinate dell'angolo top-left dell'immagine
    resolution_mpp: metri per pixel
    """
    if not detections:
        return []

    # Filtra decoy
    real_targets = [d for d in detections if not d.is_decoy]

    # Clustering spaziale
    clusters = cluster_detections(real_targets, max_distance_px=350)

    batteries = []

    for cluster in clusters:
        if len(cluster) < 2:
            continue

        geo = analyze_deployment_geometry(cluster)
        tel_count = geo["tel_count"]
        radar_count = geo["radar_count"]

        if tel_count == 0:
            continue

        tels = [d for d in cluster if d.target_class == "TEL"]
        tel_dims = [{"length": d.area_m2 / max(d.area_m2 / d.aspect_ratio, 1)} for d in tels]

        battery_type, type_confidence = identify_battery_type(
            tel_count, radar_count, tel_dims
        )

        # Calcola confidence complessiva
        base_confidence = np.mean([d.confidence for d in cluster])
        pattern_bonus = {
            "RADIAL": 0.20, "LINEAR": 0.05,
            "DISPERSED": 0.08, "INCOMPLETE": -0.10
        }.get(geo["pattern"], 0.0)

        composition_bonus = (
            0.15 if radar_count >= 1 else 0.0 +
            0.10 if geo["reload_count"] >= 1 else 0.0
        )

        tel_count_bonus = min((tel_count - 1) * 0.04, 0.16)

        final_confidence = min(
            base_confidence + pattern_bonus + composition_bonus + tel_count_bonus,
            0.97
        )

        # Alert level
        if final_confidence >= 0.85:
            alert_level = "CRITICAL"
        elif final_confidence >= 0.70:
            alert_level = "HIGH"
        elif final_confidence >= 0.50:
            alert_level = "MEDIUM"
        else:
            alert_level = "LOW"

        # Override: pattern radiale con radar = CRITICAL se >= 4 TEL
        if geo["pattern"] == "RADIAL" and radar_count >= 1 and tel_count >= 4:
            alert_level = "CRITICAL"
            final_confidence = max(final_confidence, 0.92)

        # Calcola coordinate del centro
        cx_px = geo["cluster_center"]["x"]
        cy_px = geo["cluster_center"]["y"]

        # Converti pixel → lat/lon
        # 1 grado lat ≈ 111,320m; 1 grado lon ≈ 111,320 * cos(lat)m
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * abs(math.cos(math.radians(image_origin_lat)))

        delta_lat = -(cy_px * resolution_mpp) / meters_per_deg_lat
        delta_lon = (cx_px * resolution_mpp) / meters_per_deg_lon

        center_lat = round(image_origin_lat + delta_lat, 6)
        center_lon = round(image_origin_lon + delta_lon, 6)

        # Determina activity status dalla media
        statuses = [d.activity_status for d in cluster]
        if "In movimento" in statuses:
            is_active = True
        elif statuses.count("Schierato") > len(statuses) / 2:
            is_active = True
        else:
            is_active = bool(tel_count >= 2 and radar_count >= 1)

        battery = BatteryAssessment(
            battery_id=f"SAM-{str(uuid.uuid4())[:8].upper()}",
            confidence=round(final_confidence, 3),
            alert_level=alert_level,
            battery_type=battery_type,
            tel_count=tel_count,
            radar_count=radar_count,
            reload_count=geo["reload_count"],
            deployment_pattern=geo["pattern"],
            center_coordinates={"lat": center_lat, "lon": center_lon},
            detections=cluster,
            is_active=is_active,
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        batteries.append(battery)

    # Ordina per confidence decrescente
    batteries.sort(key=lambda b: b.confidence, reverse=True)
    return batteries

