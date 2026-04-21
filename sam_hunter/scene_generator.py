"""
POSEIDON-OS / SAM Hunter — Synthetic Scene Generator
Genera sceni satellite sintetici realistici per test e demo.
Include: terreno, vegetazione, strade, veicoli, reti mimetiche, decoy.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Tuple
import random
import math


def generate_sam_scene(
    width: int = 1280,
    height: int = 1280,
    scenario: str = "active_battery",  # active_battery | transit | decoy_field
    resolution_mpp: float = 0.5,
    seed: int = 42
) -> Tuple[Image.Image, List[dict]]:
    """
    Genera un'immagine satellite sintetica con sistema SAM.
    Ritorna (immagine, lista di ground_truth_objects)
    """
    random.seed(seed)
    np.random.seed(seed)

    # ── Sfondo terreno ──
    # Prato/campo (verde-giallo tipico satellitare)
    base_color = (145 + random.randint(-10, 10),
                  148 + random.randint(-10, 10),
                  98 + random.randint(-10, 10))
    img = Image.new("RGB", (width, height), base_color)
    draw = ImageDraw.Draw(img)

    # Texture terreno (rumore)
    terrain_noise = np.random.randint(-18, 18, (height, width, 3), dtype=np.int16)
    img_array = np.array(img).astype(np.int16) + terrain_noise
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # Vegetazione sparsa (macchie verdi scure)
    for _ in range(40):
        x = random.randint(0, width)
        y = random.randint(0, height)
        r = random.randint(15, 55)
        color = (random.randint(60, 100), random.randint(100, 140), random.randint(40, 80))
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color)

    # Strade/piste di accesso
    road_color = (160, 150, 130)
    # Strada principale
    draw.line([(0, height//2 + 80), (width, height//2 + 80)], fill=road_color, width=6)
    # Pista di accesso al sito
    draw.line([(width//2, height//2 + 80), (width//2, height//3)], fill=road_color, width=4)

    ground_truth = []

    if scenario == "active_battery":
        # S-400 battery — schieramento radiale classico
        # Centro: radar al centro, 4 TEL disposti radialmente
        cx, cy = width // 2, height // 2 - 60

        # Aree di vegetazione/mimetismo attorno al sito
        for _ in range(8):
            angle = random.uniform(0, 2 * math.pi)
            r = random.randint(120, 200)
            x = int(cx + r * math.cos(angle))
            y = int(cy + r * math.sin(angle))
            vr = random.randint(20, 45)
            draw.ellipse([x-vr, y-vr, x+vr, y+vr],
                         fill=(random.randint(55,90), random.randint(95,130), random.randint(35,70)))

        # Rete mimetica (area rettangolare verdastra sul sito)
        camo_color = (110, 125, 75)
        draw.rectangle([cx-180, cy-180, cx+180, cy+180], fill=camo_color)

        # 92N6E Radar (centro) — forma quadrata/rettangolare grande
        radar_w, radar_h = 22, 20
        radar_x1, radar_y1 = cx - radar_w//2, cy - radar_h//2
        radar_x2, radar_y2 = cx + radar_w//2, cy + radar_h//2
        draw.rectangle([radar_x1, radar_y1, radar_x2, radar_y2], fill=(185, 180, 170))
        # Parabola radar (cerchio piccolo sopra)
        draw.ellipse([cx-8, cy-radar_h//2-14, cx+8, cy-radar_h//2-2], fill=(200, 195, 190))
        ground_truth.append({
            "class": "FIRE_CONTROL_RADAR", "bbox": [radar_x1-2, radar_y1-16, radar_x2+2, radar_y2],
            "is_real": True, "has_tracks": False
        })

        # 4x TEL S-400 (radiali, ~150px dal centro)
        tel_radius = 155
        tel_angles = [45, 135, 225, 315]
        tel_length_px = int(15.0 / resolution_mpp)   # 15m → 30px
        tel_width_px  = int(3.2 / resolution_mpp)    # 3.2m → 6px

        for angle_deg in tel_angles:
            angle_rad = math.radians(angle_deg)
            tx = int(cx + tel_radius * math.cos(angle_rad))
            ty = int(cy + tel_radius * math.sin(angle_rad))

            # Ruota il rettangolo del TEL
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            half_l = tel_length_px // 2
            half_w = tel_width_px // 2

            corners = [
                (tx + cos_a*half_l - sin_a*half_w, ty + sin_a*half_l + cos_a*half_w),
                (tx - cos_a*half_l - sin_a*half_w, ty - sin_a*half_l + cos_a*half_w),
                (tx - cos_a*half_l + sin_a*half_w, ty - sin_a*half_l - cos_a*half_w),
                (tx + cos_a*half_l + sin_a*half_w, ty + sin_a*half_l - cos_a*half_w),
            ]
            draw.polygon(corners, fill=(175, 170, 162))

            # Tracce pneumatici (linee scure verso il centro)
            track_end_x = int(tx + (cx - tx) * 0.4)
            track_end_y = int(ty + (cy - ty) * 0.4)
            track_color = (100, 95, 80)
            draw.line([(tx, ty), (track_end_x, track_end_y)], fill=track_color, width=2)

            bbox_x1 = int(min(c[0] for c in corners)) - 2
            bbox_y1 = int(min(c[1] for c in corners)) - 2
            bbox_x2 = int(max(c[0] for c in corners)) + 2
            bbox_y2 = int(max(c[1] for c in corners)) + 2
            ground_truth.append({
                "class": "TEL", "bbox": [bbox_x1, bbox_y1, bbox_x2, bbox_y2],
                "is_real": True, "has_tracks": True
            })

        # Veicolo di ricarica (adiacente a un TEL)
        reload_x = int(cx + (tel_radius + 45) * math.cos(math.radians(45)))
        reload_y = int(cy + (tel_radius + 45) * math.sin(math.radians(45)))
        rl_w, rl_h = 18, 7
        draw.rectangle([reload_x-rl_w//2, reload_y-rl_h//2,
                         reload_x+rl_w//2, reload_y+rl_h//2], fill=(165, 160, 152))
        ground_truth.append({
            "class": "RELOAD_VEHICLE",
            "bbox": [reload_x-rl_w//2-1, reload_y-rl_h//2-1,
                     reload_x+rl_w//2+1, reload_y+rl_h//2+1],
            "is_real": True, "has_tracks": True
        })

        # DECOY — esca gonfiabile S-300 (stessa forma, nessuna traccia)
        decoy_x = cx + 280
        decoy_y = cy - 120
        d_l, d_w = 28, 6
        draw.rectangle([decoy_x-d_l//2, decoy_y-d_w//2,
                         decoy_x+d_l//2, decoy_y+d_w//2], fill=(178, 174, 168))
        ground_truth.append({
            "class": "TEL", "bbox": [decoy_x-d_l//2-1, decoy_y-d_w//2-1,
                                      decoy_x+d_l//2+1, decoy_y+d_w//2+1],
            "is_real": False, "has_tracks": False,
            "note": "Inflatable decoy — no tracks, no SAR reflection"
        })

    # ── Rumore finale ──
    img_arr = np.array(img).astype(np.int16)
    sensor_noise = np.random.randint(-6, 7, img_arr.shape, dtype=np.int16)
    img_arr = np.clip(img_arr + sensor_noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)

    # Leggero blur atmosferico (simula diffrazione)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    return img, ground_truth

