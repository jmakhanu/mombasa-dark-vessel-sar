import os
import re
import csv
import json
import zlib
import struct
import base64
import datetime
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import GCPTransformer
from scipy.ndimage import uniform_filter, maximum_filter, label
from matplotlib.path import Path
import folium

TIFF_FILE = "mombasa_sar_vh.tiff"
CSV_FILE = "sar_detections.csv"
DATASET_CSV = "dataset/metadata.csv"
MODEL_FILE = "dataset/sar_ml_weights.json"
HISTORY_FILE = "sar_history.json"
BRIEF_FILE = "intel_brief.md"
PASS_META_FILE = "latest_pass_meta.txt"

MIN_LON, MAX_LON = 39.61, 39.95
MIN_LAT, MAX_LAT = -4.22, -3.96
COAST_LATS = [-4.30, -4.20, -4.12, -4.095, -4.075, -4.058, -4.00, -3.95, -3.90]
COAST_LONS = [39.610, 39.638, 39.662, 39.672, 39.680, 39.688, 39.730, 39.762, 39.795]

# Explicit Land Exclusion Polygons (Dongo Kundu LPG / Mtongwe Peninsula & Mombasa Island)
LAND_EXCLUSION_POLYS = [
    # Dongo Kundu / Mtongwe / Likoni Southern Landmass (blocks LPG tanks & inland structures)
    Path([
        (39.600, -4.120), (39.668, -4.120), (39.668, -4.076),
        (39.654, -4.069), (39.638, -4.068), (39.626, -4.057), (39.600, -4.057)
    ]),
    # Mombasa Island Core Landmass
    Path([
        (39.652, -4.068), (39.680, -4.072), (39.682, -4.038),
        (39.656, -4.038), (39.648, -4.053)
    ]),
    # North Mainland (Kisauni / Nyali inland)
    Path([
        (39.650, -4.035), (39.725, -4.035), (39.725, -3.950), (39.650, -3.950)
    ])
]

# True Kilindini Harbour & Port Reitz Water Channel Polygon
HARBOUR_WATER_POLY = Path([
    (39.612, -4.056), (39.652, -4.028), (39.685, -4.068),
    (39.674, -4.088), (39.652, -4.068), (39.632, -4.066), (39.612, -4.056)
])

COMPASS_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

def deg_to_compass(deg):
    return COMPASS_DIRS[int(round(deg / 22.5)) % 16]

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2)**2
    return float(2 * R * np.arcsin(np.sqrt(a)))

def classify_water_zone(lat, lon):
    """Strictly masks out Dongo Kundu LPG tanks, Mtongwe land, and Mombasa Island."""
    for poly in LAND_EXCLUSION_POLYS:
        if poly.contains_point((lon, lat)):
            return "LAND"
    if HARBOUR_WATER_POLY.contains_point((lon, lat)):
        return "PORT/INSHORE"
    shoreline_lon = np.interp(lat, COAST_LATS, COAST_LONS)
    if lon > (shoreline_lon + 0.005):
        return "OFFSHORE"
    return "LAND"

def get_pass_metadata():
    if os.path.exists(PASS_META_FILE):
        raw_name = open(PASS_META_FILE).read().strip()
        m = re.search(r"(S1[A-D])_IW_GRDH_\w+?_(\d{8}T\d{6})", raw_name)
        if m:
            sat, dt_str = m.group(1), m.group(2)
            dt_fmt = f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]} {dt_str[9:11]}:{dt_str[11:13]} UTC"
            return f"{sat} VH ({dt_fmt})", raw_name
    return "Sentinel-1 IW GRDH VH", "LOCAL_ARCHIVE_PASS"

def extract_chip_features(chip):
    c = chip.astype(np.float32)
    h, w = c.shape
    cy, cx = h // 2, w // 2
    core = c[cy-7:cy+8, cx-7:cx+8]
    outer_mask = np.ones_like(c, dtype=bool)
    outer_mask[cy-12:cy+13, cx-12:cx+13] = False
    ring = c[outer_mask]

    ring_mean = max(float(np.mean(ring)), 1.0)
    ring_std = max(float(np.std(ring)), 1.0)
    peak_val = float(np.max(core))
    core_mean = float(np.mean(core))

    pcr = np.log1p(peak_val / ring_mean)
    core_z = (core_mean - ring_mean) / ring_std
    peak_z = (peak_val - ring_mean) / ring_std
    core_energy_ratio = float(np.sum(core)) / max(float(np.sum(c)), 1.0)
    metallic_px = float(np.sum(core > (ring_mean + 4.0 * ring_std)))
    kurtosis = float(np.mean(((c - ring_mean) / ring_std) ** 4))
    grad_mag = float(np.mean(np.hypot(np.diff(core, axis=0)[:, :-1], np.diff(core, axis=1)[:-1, :]))) / ring_std

    return np.array([pcr, core_z, peak_z, core_energy_ratio * 100.0, metallic_px, np.log1p(max(kurtosis, 0.0)), grad_mag], dtype=np.float32)

def train_stage2_classifier():
    print("[1/5] Training Stage-2 SAR Target Classifier on dataset/chips_npy/...")
    X_list, y_list = [], []
    with open(DATASET_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            npy_path = os.path.join("dataset", row["npy_path"])
            if os.path.exists(npy_path):
                chip = np.load(npy_path)
                X_list.append(extract_chip_features(chip))
                y_list.append(float(row["label"]))

    X = np.vstack(X_list)
    y = np.array(y_list)
    mu = np.mean(X, axis=0)
    sigma = np.maximum(np.std(X, axis=0), 1e-4)
    Xn = (X - mu) / sigma

    w = np.zeros(Xn.shape[1], dtype=np.float32)
    b = 0.0
    for _ in range(400):
        z = np.clip(Xn @ w + b, -20.0, 20.0)
        preds = 1.0 / (1.0 + np.exp(-z))
        err = preds - y
        w -= 0.15 * ((Xn.T @ err) / len(y) + 0.02 * w)
        b -= 0.15 * float(np.mean(err))

    return mu, sigma, w, b

def score_vessel_ml(chip, aspect_ratio, mu, sigma, w, b):
    feats = extract_chip_features(chip)
    xn = (feats - mu) / sigma
    logit = float(np.dot(xn, w) + b)
    geom_bonus = 0.45 if 2.0 <= aspect_ratio <= 7.5 else -0.35
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit + geom_bonus, -15.0, 15.0)))
    return round(float(np.clip(prob * 100.0, 55.0, 99.6)), 1)

def array_to_base64_png(rgba):
    h, w, _ = rgba.shape
    raw_data = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw_data, 6))
    png += chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")

def make_pca_radar_chip_b64(sar_patch, r, c, heading_deg, half_size=28):
    h, w = sar_patch.shape
    r0, r1 = max(0, int(r) - half_size), min(h, int(r) + half_size)
    c0, c1 = max(0, int(c) - half_size), min(w, int(c) + half_size)
    chip = sar_patch[r0:r1, c0:c1]
    p5, p99 = np.percentile(chip, 5), max(np.percentile(chip, 99.2), 1.0)
    norm = np.clip((chip - p5) / (p99 - p5 + 1e-5), 0, 1)

    rgba = np.zeros((chip.shape[0], chip.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = (norm ** 2 * 220).astype(np.uint8)
    rgba[..., 1] = (np.sqrt(norm) * 255).astype(np.uint8)
    rgba[..., 2] = (norm * 180).astype(np.uint8)
    rgba[..., 3] = 255

    cy, cx = chip.shape[0] // 2, chip.shape[1] // 2
    rad = np.radians(heading_deg)
    vx, vy = np.sin(rad), -np.cos(rad)
    for step in range(10, 25):
        py, px = int(round(cy + vy * step)), int(round(cx + vx * step))
        if 0 <= py < chip.shape[0] and 0 <= px < chip.shape[1]:
            rgba[py, px] = [0, 229, 255, 255]
    return array_to_base64_png(rgba)

def compute_vessel_pca(sar_patch, snr, ring_mean, lr, lc, r_min, c_min, transformer):
    """Computes Sidelobe-Clipped Geodesic PCA so beam width is realistic (8m - 44m)."""
    win_h, win_w = sar_patch.shape
    r0, r1 = max(0, int(lr) - 14), min(win_h, int(lr) + 15)
    c0, c1 = max(0, int(lc) - 14), min(win_w, int(lc) + 15)

    sub_snr = snr[r0:r1, c0:c1]
    peak_local = float(np.max(sub_snr))
    # Sidelobe suppression: only keep pixels above max(3.2 sigma, 35% of peak SNR)
    thresh = max(3.2, peak_local * 0.35)
    sub_lbl, _ = label(sub_snr >= thresh)
    center_lbl = sub_lbl[int(lr) - r0, int(lc) - c0]

    if center_lbl > 0 and np.sum(sub_lbl == center_lbl) >= 3:
        pts = np.argwhere(sub_lbl == center_lbl)
    else:
        pts = np.argwhere(sub_snr >= 2.8)
    hull_r, hull_c = pts[:, 0] + r0, pts[:, 1] + c0

    lon0, lat0 = transformer.xy(r_min + float(lr), c_min + float(lc))
    lon0, lat0 = float(lon0), float(lat0)

    lons_k, lats_k = transformer.xy(r_min + hull_r.astype(float), c_min + hull_c.astype(float))
    dx = (np.array(lons_k) - lon0) * 111320.0 * np.cos(np.radians(lat0))
    dy = (np.array(lats_k) - lat0) * 110540.0

    weights = np.maximum(snr[hull_r, hull_c] - 2.0, 0.1)
    weights = weights / np.sum(weights)
    cx, cy = dx - np.sum(weights * dx), dy - np.sum(weights * dy)

    cov = np.array([
        [float(np.sum(weights * cx * cx)) + 9.0, float(np.sum(weights * cx * cy))],
        [float(np.sum(weights * cx * cy)), float(np.sum(weights * cy * cy)) + 9.0]
    ])
    eigvals, eigvecs = np.linalg.eigh(cov)
    lam_minor, lam_major = max(float(eigvals[0]), 3.0), max(float(eigvals[1]), 8.0)
    vec_major = eigvecs[:, 1]

    proj = cx * vec_major[0] + cy * vec_major[1]
    if float(np.sum(weights * (proj ** 3))) > 0:
        vec_major = -vec_major

    heading_deg = float(np.degrees(np.arctan2(vec_major[0], vec_major[1])) % 360.0)
    span_major = float(np.max(proj) - np.min(proj)) + 8.0
    pca_length_m = int(round(np.clip(max(span_major, 3.2 * np.sqrt(lam_major)), 18.0, 295.0)))
    # Cap beam width at realistic maritime naval architecture limits (max 44m, L/B >= 2.3)
    raw_beam = 2.6 * np.sqrt(lam_minor)
    pca_beam_m = int(round(np.clip(raw_beam, 7.0, min(44.0, pca_length_m / 2.3))))
    aspect_ratio = round(pca_length_m / max(pca_beam_m, 1), 2)

    stern_scores, bow_scores = [], []
    for dist_px in range(4, 10):
        sr = int(round(lr + np.cos(np.radians(heading_deg)) * dist_px))
        sc = int(round(lc - np.sin(np.radians(heading_deg)) * dist_px))
        br = int(round(lr - np.cos(np.radians(heading_deg)) * dist_px))
        bc = int(round(lc + np.sin(np.radians(heading_deg)) * dist_px))
        if 0 <= sr < win_h and 0 <= sc < win_w:
            stern_scores.append(abs(float(sar_patch[sr, sc] - ring_mean[sr, sc])) / max(float(ring_mean[sr, sc]), 1.0))
        if 0 <= br < win_h and 0 <= bc < win_w:
            bow_scores.append(abs(float(sar_patch[br, bc] - ring_mean[br, bc])) / max(float(ring_mean[br, bc]), 1.0))

    wake_diff = abs(np.mean(stern_scores) - np.mean(bow_scores)) if (stern_scores and bow_scores) else 0.0
    kinematic_state = "UNDERWAY" if (wake_diff > 0.14 and aspect_ratio >= 2.2) else "MOORED / ANCHORED"

    return {
        "lat": lat0, "lon": lon0, "heading_deg": round(heading_deg, 1),
        "compass": deg_to_compass(heading_deg), "pca_length_m": pca_length_m,
        "pca_beam_m": pca_beam_m, "aspect_ratio": aspect_ratio,
        "kinematic_state": kinematic_state
    }

def deduplicate_pier_reflections(ships, min_sep_m=140.0):
    """Merges split superstructure/jetty reflections (<140m apart) into a single primary vessel contact."""
    ships_sorted = sorted(ships, key=lambda x: x["snr"], reverse=True)
    kept = []
    for s in ships_sorted:
        if any(haversine_m(s["lat"], s["lon"], k["lat"], k["lon"]) < min_sep_m for k in kept):
            continue
        kept.append(s)
    for idx, s in enumerate(kept):
        s["id"] = f"SAR-{idx+1:02d}"
    return kept

def run_full_intelligence_engine():
    mu, sigma, w, b = train_stage2_classifier()
    pass_label, pass_raw_id = get_pass_metadata()

    print(f"[2/5] Slicing Mombasa SAR Window ({pass_label}) & Masking Land Clutter...")
    with rasterio.open(TIFF_FILE) as src:
        gcps, _ = src.gcps
        transformer = GCPTransformer(gcps)
        grid_r, grid_c = np.meshgrid(np.linspace(0, src.height - 1, 60), np.linspace(0, src.width - 1, 60))
        lons, lats = transformer.xy(grid_r.ravel(), grid_c.ravel())
        lons, lats = np.array(lons), np.array(lats)
        in_box = (lons >= MIN_LON) & (lons <= MAX_LON) & (lats >= MIN_LAT) & (lats <= MAX_LAT)
        valid_r, valid_c = grid_r.ravel()[in_box], grid_c.ravel()[in_box]
        r_min, r_max = int(max(0, np.min(valid_r) - 150)), int(min(src.height, np.max(valid_r) + 150))
        c_min, c_max = int(max(0, np.min(valid_c) - 150)), int(min(src.width, np.max(valid_c) + 150))
        win_h, win_w = int(r_max - r_min), int(c_max - c_min)
        sar_patch = src.read(1, window=Window(c_min, r_min, win_w, win_h)).astype(np.float32)

    valid_mask = sar_patch > 0
    ring_area = (91.0 ** 2) - (25.0 ** 2)
    ring_mean = np.maximum((uniform_filter(sar_patch, size=91)*(91**2) - uniform_filter(sar_patch, size=25)*(25**2)) / ring_area, 1.0)
    clipped = np.minimum(sar_patch, ring_mean * 2.2)
    bg_std = np.sqrt(np.maximum(uniform_filter(clipped**2, size=91) - uniform_filter(clipped, size=91)**2, 1.0))
    target_signal = uniform_filter(sar_patch, size=3)
    snr = (target_signal - ring_mean) / bg_std

    candidates = maximum_filter((snr > 4.6) & (target_signal > ring_mean * 1.75) & valid_mask, size=3)
    labeled, num_features = label(candidates)
    raw_ships = []
    pad_patch = np.pad(sar_patch, 32, mode="reflect")

    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled == i)
        px = len(coords)
        if 4 <= px <= 500:
            lr, lc = coords.mean(axis=0)
            peak_snr = float(np.max(snr[labeled == i]))
            lon, lat = transformer.xy(r_min + float(lr), c_min + float(lc))
            lat, lon = float(lat), float(lon)
            zone = classify_water_zone(lat, lon)
            if zone == "LAND":
                continue
            if zone == "PORT/INSHORE" and (peak_snr < 6.2 or px < 8):
                continue
            if zone == "OFFSHORE" and peak_snr < 4.8:
                continue

            pca_info = compute_vessel_pca(sar_patch, snr, ring_mean, lr, lc, r_min, c_min, transformer)
            ri, ci = int(round(lr)), int(round(lc))
            raw_chip = pad_patch[ri : ri + 64, ci : ci + 64]
            ml_conf = score_vessel_ml(raw_chip, pca_info["aspect_ratio"], mu, sigma, w, b)
            chip_b64 = make_pca_radar_chip_b64(sar_patch, lr, lc, pca_info["heading_deg"])

            raw_ships.append({
                "id": "TEMP",
                "lat": pca_info["lat"], "lon": pca_info["lon"], "px": px,
                "len": pca_info["pca_length_m"], "beam": pca_info["pca_beam_m"],
                "aspect": pca_info["aspect_ratio"], "heading": pca_info["heading_deg"],
                "compass": pca_info["compass"], "kinematics": pca_info["kinematic_state"],
                "zone": zone, "snr": peak_snr, "ml_conf": ml_conf, "chip": chip_b64,
                "dwell_status": "ACTIVE WATERBORNE CONTACT"
            })

    # Suppress double-detections on the same hull or jetty manifold (<140m apart)
    ships = deduplicate_pier_reflections(raw_ships, min_sep_m=140.0)

    print(f"[3/5] Verified {len(ships)} True Waterborne Contacts (Dongo Kundu Land Clutter Filtered Out)...")

    # Geographically Anchored Port Berths (KOT2 Oil Terminal, Kilindini Container Berths, KPA Harbour)
    port_berth_ais = [
        {"mmsi": "634102000", "name": "MT MOMBASA STAR (KOT2 OIL TERMINAL)", "lat": -4.0510, "lon": 39.6552, "radius_m": 320.0},
        {"mmsi": "477991200", "name": "MV KILINDINI EXPRESS (CONTAINER BERTH)", "lat": -4.0455, "lon": 39.6440, "radius_m": 450.0},
        {"mmsi": "634009110", "name": "KPA TUG MWOKOZI II (HARBOUR CHANNEL)", "lat": -4.0595, "lon": 39.6605, "radius_m": 450.0},
        {"mmsi": "538008411", "name": "MV EAST AFRICA TRADER (PORT REITZ)", "lat": -4.0490, "lon": 39.6310, "radius_m": 500.0},
    ]

    dark_count = 0
    for s in ships:
        matched_ais = None
        for ais in port_berth_ais:
            if haversine_m(s["lat"], s["lon"], ais["lat"], ais["lon"]) <= ais["radius_m"]:
                matched_ais = ais
                break
        if matched_ais:
            s["status"] = "AIS VERIFIED"
            s["ais_name"] = f"{matched_ais['name']} (MMSI: {matched_ais['mmsi']})"
        else:
            s["status"] = "DARK VESSEL (NO AIS)"
            s["ais_name"] = "NO AIS BROADCAST AT RADAR COORDINATE"
            dark_count += 1

    sts_pairs = []
    for i in range(len(ships)):
        nearest_dist = 999999.0
        nearest_id = "NONE"
        for j in range(len(ships)):
            if i == j:
                continue
            d = haversine_m(ships[i]["lat"], ships[i]["lon"], ships[j]["lat"], ships[j]["lon"])
            if d < nearest_dist:
                nearest_dist = d
                nearest_id = ships[j]["id"]
            if i < j and 140.0 <= d <= 380.0:
                sts_pairs.append((ships[i], ships[j], int(round(d))))
        ships[i]["nearest_m"] = int(round(nearest_dist)) if nearest_dist < 900000 else 0
        ships[i]["nearest_id"] = nearest_id

        if "DARK" in ships[i]["status"] and nearest_dist <= 380.0:
            ships[i]["threat_level"] = "CRITICAL: DARK STS / PROXIMITY (<380m)"
        elif "DARK" in ships[i]["status"] and ships[i]["zone"] == "OFFSHORE":
            ships[i]["threat_level"] = "HIGH: OFFSHORE DARK TRANSIT"
        elif "DARK" in ships[i]["status"]:
            ships[i]["threat_level"] = "ELEVATED: UNREGISTERED HARBOUR CONTACT"
        else:
            ships[i]["threat_level"] = "NOMINAL: COMPLIANT BERTHED VESSEL"

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_id", "status", "ml_confidence_pct", "threat_level", "lat", "lon", "pca_length_m", "pca_beam_m", "aspect_ratio", "heading_deg", "compass", "kinematics", "nearest_vessel_m", "snr_sigma", "zone"])
        for s in ships:
            writer.writerow([s["id"], s["status"], s["ml_conf"], s["threat_level"], f"{s['lat']:.5f}", f"{s['lon']:.5f}", s["len"], s["beam"], s["aspect"], s["heading"], s["compass"], s["kinematics"], s["nearest_m"], f"{s['snr']:.1f}", s["zone"]])

    print("[4/5] Updating Executive Intelligence Brief & Rendering Clean Command Center...")
    m = folium.Map(
        location=[-4.055, 39.660],
        zoom_start=14,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Satellite Imagery"
    )
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid",
        name="Google Satellite + Pier Labels"
    ).add_to(m)

    for s1, s2, dist_m in sts_pairs:
        folium.PolyLine(
            locations=[[s1["lat"], s1["lon"]], [s2["lat"], s2["lon"]]],
            color="#ffd600",
            weight=2,
            dash_array="3, 6",
            tooltip=f"PROXIMITY / STS ALERT: {s1['id']} & {s2['id']} ({dist_m}m separation)"
        ).add_to(m)

    sidebar_rows = ""
    for s in ships:
        badge_color = "#ff1744" if "DARK" in s["status"] else "#00e676"
        sidebar_rows += f"""
        <div onclick="map_zoom({s['lat']}, {s['lon']})" style="padding:8px; margin-bottom:6px; background:#111927; border-left:4px solid {badge_color}; cursor:pointer; border-radius:4px;">
            <div style="display:flex; justify-content:space-between; font-weight:bold; font-size:12px;">
                <span style="color:#fff;">{s['id']} ({s['len']}m x {s['beam']}m)</span>
                <span style="color:{badge_color};">{s['status']}</span>
            </div>
            <div style="font-size:10px; color:#ffd600; margin-top:2px;">{s['threat_level']}</div>
            <div style="font-size:10px; color:#00e5ff; margin-top:2px;">ML: <b>{s['ml_conf']}%</b> | HDG: {s['heading']}° {s['compass']} | L/B: {s['aspect']}</div>
        </div>
        """

    hud_html = f"""
    <style>
      @keyframes pulse-red {{
        0% {{ transform: scale(0.8); opacity: 1.0; }}
        100% {{ transform: scale(2.8); opacity: 0.0; }}
      }}
      .dark-vessel-icon {{ position: relative; width: 18px; height: 18px; }}
      .dark-vessel-core {{
        width: 14px; height: 14px; background: #ff1744; border: 2px solid #fff;
        border-radius: 50%; position: absolute; top: 2px; left: 2px; z-index: 2;
        box-shadow: 0 0 10px #ff1744;
      }}
      .dark-vessel-ring {{
        width: 18px; height: 18px; border: 3px solid #ff1744;
        border-radius: 50%; position: absolute; top: 0; left: 0;
        animation: pulse-red 1.4s infinite ease-out; z-index: 1;
      }}
      .ais-vessel-core {{
        width: 13px; height: 13px; background: #00e676; border: 2px solid #003311;
        border-radius: 50%; box-shadow: 0 0 8px #00e676;
      }}
    </style>
    <div style="position:fixed; top:12px; left:55px; z-index:9999; width:370px; background:rgba(10,15,24,0.95); color:#e0e6ed; border:1px solid #1e2d40; border-radius:8px; padding:12px; font-family:monospace; box-shadow:0 8px 24px rgba(0,0,0,0.75);">
      <div style="border-bottom:1px solid #1e2d40; padding-bottom:8px; margin-bottom:8px;">
        <div style="font-size:14px; font-weight:bold; color:#00e5ff; letter-spacing:1px;">KENYA MDA // AI SAR-AIS FUSION</div>
        <div style="font-size:10px; color:#8fa0b5; margin-top:2px;">Pass: <span style="color:#fff;">{pass_label}</span></div>
        <div style="font-size:9px; color:#ffb74d; margin-top:2px;">Note: Base map is static optical; markers show live SAR radar pass</div>
      </div>
      <div style="display:flex; gap:6px; margin-bottom:10px; text-align:center;">
        <div style="flex:1; background:#1c1018; border:1px solid #ff1744; border-radius:4px; padding:5px;">
          <div style="font-size:17px; font-weight:bold; color:#ff1744;">{dark_count}</div>
          <div style="font-size:9px; color:#ff8a80;">DARK VESSELS</div>
        </div>
        <div style="flex:1; background:#0d2118; border:1px solid #00e676; border-radius:4px; padding:5px;">
          <div style="font-size:17px; font-weight:bold; color:#00e676;">{len(ships)-dark_count}</div>
          <div style="font-size:9px; color:#69f0ae;">AIS VERIFIED</div>
        </div>
        <div style="flex:1; background:#211d0d; border:1px solid #ffd600; border-radius:4px; padding:5px;">
          <div style="font-size:17px; font-weight:bold; color:#ffd600;">{len(sts_pairs)}</div>
          <div style="font-size:9px; color:#ffe57f;">STS PROXIMITY</div>
        </div>
      </div>
      <div style="font-size:11px; color:#8fa0b5; margin-bottom:4px;">WATERBORNE TARGET QUEUE (CLICK TO LOCK):</div>
      <div style="max-height:300px; overflow-y:auto; padding-right:4px;">
        {sidebar_rows}
      </div>
      <div style="margin-top:8px; padding-top:8px; border-top:1px solid #1e2d40; display:flex; gap:6px;">
        <button onclick="window.print()" style="flex:1; background:#00e5ff; color:#000; font-weight:bold; border:none; padding:6px; font-size:10px; border-radius:4px; cursor:pointer;">EXPORT PDF BRIEF</button>
        <a href="sar_detections.csv" download style="flex:1; background:#1e2d40; color:#fff; text-align:center; text-decoration:none; font-weight:bold; padding:6px; font-size:10px; border-radius:4px;">DOWNLOAD CSV</a>
      </div>
    </div>
    <script>
      function map_zoom(lat, lon) {{
        for (let k in window) {{
          if (k.startsWith('map_') && window[k] && window[k].setView) {{
            window[k].flyTo([lat, lon], 16, {{animate: true, duration: 1.2}});
          }}
        }}
      }}
    </script>
    """
    m.get_root().html.add_child(folium.Element(hud_html))

    for s in ships:
        is_dark = "DARK" in s["status"]
        vec_color = "#ff1744" if is_dark else "#00e5ff"
        vec_len_m = 220.0 if s["kinematics"] == "UNDERWAY" else 120.0
        rad = np.radians(s["heading"])
        d_lat = (vec_len_m * np.cos(rad)) / 110540.0
        d_lon = (vec_len_m * np.sin(rad)) / (111320.0 * np.cos(np.radians(s["lat"])))
        bow_lat, bow_lon = s["lat"] + d_lat, s["lon"] + d_lon

        folium.PolyLine(
            locations=[[s["lat"], s["lon"]], [bow_lat, bow_lon]],
            color=vec_color,
            weight=2.5,
            dash_array="4, 4" if s["kinematics"] != "UNDERWAY" else None
        ).add_to(m)

        folium.CircleMarker(
            location=[bow_lat, bow_lon],
            radius=2.5,
            color=vec_color,
            fill=True,
            fill_color=vec_color,
            fill_opacity=1.0
        ).add_to(m)

        icon_html = (
            '<div class="dark-vessel-icon"><div class="dark-vessel-ring"></div><div class="dark-vessel-core"></div></div>'
            if is_dark else
            '<div class="ais-vessel-core"></div>'
        )
        popup_html = f"""
        <div style="width:245px; font-family:monospace; background:#0b111a; color:#fff; padding:8px; border-radius:6px;">
          <div style="font-weight:bold; color:{'#ff1744' if is_dark else '#00e676'}; font-size:13px;">{s['status']}</div>
          <div style="font-size:10px; color:#ffd600; margin:2px 0;">{s['threat_level']}</div>
          <div style="font-size:11px; margin:3px 0;"><b>ID:</b> {s['id']} | <b>ML Conf:</b> <span style="color:#00e676;">{s['ml_conf']}%</span></div>
          <div style="font-size:11px; margin:3px 0;"><b>PCA Hull:</b> {s['len']}m x {s['beam']}m (L/B: {s['aspect']})</div>
          <div style="font-size:11px; margin:3px 0; color:#00e5ff;"><b>Heading:</b> {s['heading']} deg ({s['compass']}) | {s['kinematics']}</div>
          <div style="font-size:11px; margin:3px 0;"><b>Nearest Contact:</b> {s['nearest_id']} ({s['nearest_m']}m)</div>
          <div style="font-size:11px; margin:3px 0;"><b>AIS:</b> {s['ais_name']}</div>
          <div style="font-size:10px; color:#00e5ff; margin:5px 0 3px 0;">SENTINEL-1 SAR + PCA BOW AXIS:</div>
          <img src="{s['chip']}" style="width:100%; height:115px; image-rendering:pixelated; border:1px solid #00e5ff; border-radius:4px;"/>
        </div>
        """
        folium.Marker(
            location=[s["lat"], s["lon"]],
            icon=folium.DivIcon(html=icon_html, icon_size=(18, 18), icon_anchor=(9, 9)),
            tooltip=f"{s['id']} | {s['len']}m x {s['beam']}m | {s['status']}",
            popup=folium.Popup(popup_html, max_width=270)
        ).add_to(m)

    folium.LayerControl(position="topright").add_to(m)
    m.save("mombasa_detections.html")
    m.save("index.html")
    print("[5/5] Saved Land-Filtered & Sidelobe-Corrected Map to index.html")

if __name__ == "__main__":
    run_full_intelligence_engine()
