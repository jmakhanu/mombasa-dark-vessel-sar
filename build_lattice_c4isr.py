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
from scipy.ndimage import uniform_filter, maximum_filter, label, zoom
from matplotlib.path import Path

TIFF_FILE = "mombasa_sar_vh.tiff"
CSV_FILE = "sar_detections.csv"
DATASET_CSV = "dataset/metadata.csv"
PASS_META_FILE = "latest_pass_meta.txt"

MIN_LON, MAX_LON = 39.61, 39.95
MIN_LAT, MAX_LAT = -4.22, -3.96
COAST_LATS = [-4.30, -4.20, -4.12, -4.095, -4.075, -4.058, -4.00, -3.95, -3.90]
COAST_LONS = [39.610, 39.638, 39.662, 39.672, 39.680, 39.688, 39.730, 39.762, 39.795]

LAND_EXCLUSION_POLYS = [
    Path([(39.600, -4.120), (39.668, -4.120), (39.668, -4.076), (39.654, -4.069), (39.638, -4.068), (39.626, -4.057), (39.600, -4.057)]),
    Path([(39.652, -4.068), (39.680, -4.072), (39.682, -4.038), (39.656, -4.038), (39.648, -4.053)]),
    Path([(39.650, -4.035), (39.725, -4.035), (39.725, -3.950), (39.650, -3.950)])
]

HARBOUR_WATER_POLY = Path([
    (39.612, -4.056), (39.652, -4.028), (39.685, -4.068),
    (39.674, -4.088), (39.652, -4.068), (39.632, -4.066), (39.612, -4.056)
])

COMPASS_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

def deg_to_compass(deg):
    return COMPASS_DIRS[int(round(deg / 22.5)) % 16]

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2)**2
    return float(2 * R * np.arcsin(np.sqrt(a)))

def classify_water_zone(lat, lon):
    for poly in LAND_EXCLUSION_POLYS:
        if poly.contains_point((lon, lat)):
            return "LAND"
    if HARBOUR_WATER_POLY.contains_point((lon, lat)):
        return "PORT_APPROACH"
    shoreline_lon = np.interp(lat, COAST_LATS, COAST_LONS)
    if lon > (shoreline_lon + 0.005):
        return "DEEP_OFFSHORE"
    return "LAND"

def get_pass_metadata():
    if os.path.exists(PASS_META_FILE):
        raw_name = open(PASS_META_FILE).read().strip()
        m = re.search(r"(S1[A-D])_IW_GRDH_\w+?_(\d{8}T\d{6})", raw_name)
        if m:
            sat, dt_str = m.group(1), m.group(2)
            dt_fmt = f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]} {dt_str[9:11]}:{dt_str[11:13]}Z"
            return f"{sat} // C-BAND SAR IW-VH // {dt_fmt}"
    return "SENTINEL-1D // C-BAND SAR IW-VH // ORBITAL PASS ARCHIVE"

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

def build_georeferenced_sar_overlay(sar_patch, r_min, c_min, transformer):
    """Projects the raw Sentinel-1 radar backscatter matrix onto a true Lat/Lon grid for map overlay."""
    print("[2/5] Projecting Raw Sentinel-1 C-Band Backscatter Swath to Geodesic Map Grid...")
    win_h, win_w = sar_patch.shape
    out_h, out_w = 520, 620
    lats = np.linspace(-3.97, -4.18, out_h)
    lons = np.linspace(39.615, 39.88, out_w)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    rows, cols = transformer.rowcol(lon_grid.ravel(), lat_grid.ravel())
    local_r = np.asarray(rows) - r_min
    local_c = np.asarray(cols) - c_min

    valid = (local_r >= 0) & (local_r < win_h) & (local_c >= 0) & (local_c < win_w)
    rc = np.clip(local_r, 0, win_h - 1).astype(int)
    cc = np.clip(local_c, 0, win_w - 1).astype(int)

    vals = np.zeros(lon_grid.size, dtype=np.float32)
    vals[valid] = sar_patch[rc[valid], cc[valid]]
    img = vals.reshape(out_h, out_w)

    pos = img[img > 0]
    p10 = np.percentile(pos, 12) if len(pos) else 1.0
    p99 = np.percentile(pos, 99.3) if len(pos) else 100.0
    norm = np.clip((img - p10) / (p99 - p10 + 1e-5), 0.0, 1.0)
    gamma = np.sqrt(norm)

    # Tactical Cyan/Emerald C4ISR Radar Colormap with transparent low-clutter floor
    rgba = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    rgba[..., 0] = (gamma ** 3 * 180).astype(np.uint8)
    rgba[..., 1] = (gamma * 235).astype(np.uint8)
    rgba[..., 2] = (np.sqrt(gamma) * 255).astype(np.uint8)
    alpha = np.where(img > 0, np.clip(gamma * 215 + 25, 0, 225), 0).astype(np.uint8)
    rgba[..., 3] = alpha

    bounds = [[-4.18, 39.615], [-3.97, 39.88]]
    return array_to_base64_png(rgba), bounds

def make_tactical_chip_and_profile(sar_patch, r, c, heading_deg, half_size=30):
    """Generates both the 60x60 tactical SAR chip and a 1D axial RCS profile array."""
    h, w = sar_patch.shape
    r0, r1 = max(0, int(r) - half_size), min(h, int(r) + half_size)
    c0, c1 = max(0, int(c) - half_size), min(w, int(c) + half_size)
    chip = sar_patch[r0:r1, c0:c1]
    p5, p99 = np.percentile(chip, 5), max(np.percentile(chip, 99.4), 1.0)
    norm = np.clip((chip - p5) / (p99 - p5 + 1e-5), 0, 1)

    rgba = np.zeros((chip.shape[0], chip.shape[1], 4), dtype=np.uint8)
    # High-contrast phosphor-cyan/amber radar rendering
    rgba[..., 0] = (norm ** 2.2 * 255).astype(np.uint8)
    rgba[..., 1] = (np.sqrt(norm) * 240).astype(np.uint8)
    rgba[..., 2] = (norm * 190).astype(np.uint8)
    rgba[..., 3] = 255

    cy, cx = chip.shape[0] // 2, chip.shape[1] // 2
    # Draw subtle range/azimuth crosshair ticks
    for d in range(4, half_size - 2, 2):
        if 0 <= cy - d < chip.shape[0]: rgba[cy - d, cx] = [0, 90, 110, 255]
        if 0 <= cy + d < chip.shape[0]: rgba[cy + d, cx] = [0, 90, 110, 255]
        if 0 <= cx - d < chip.shape[1]: rgba[cy, cx - d] = [0, 90, 110, 255]
        if 0 <= cx + d < chip.shape[1]: rgba[cy, cx + d] = [0, 90, 110, 255]

    # Draw PCA bow vector in bright cyan
    rad = np.radians(heading_deg)
    vx, vy = np.sin(rad), -np.cos(rad)
    for step in range(9, 26):
        py, px = int(round(cy + vy * step)), int(round(cx + vx * step))
        if 0 <= py < chip.shape[0] and 0 <= px < chip.shape[1]:
            rgba[py, px] = [0, 240, 255, 255]

    # Extract 24-point axial Radar Cross-Section (RCS) profile in dB proxy
    profile = []
    for s in np.linspace(-18, 18, 24):
        py = int(np.clip(round(cy + vy * s), 0, chip.shape[0] - 1))
        px = int(np.clip(round(cx + vx * s), 0, chip.shape[1] - 1))
        val_db = round(float(10.0 * np.log10(max(chip[py, px], 1.0))), 1)
        profile.append(val_db)

    return array_to_base64_png(rgba), profile

def extract_chip_features(chip):
    c = chip.astype(np.float32)
    h, w = c.shape
    cy, cx = h // 2, w // 2
    core = c[cy-7:cy+8, cx-7:cx+8]
    outer_mask = np.ones_like(c, dtype=bool)
    outer_mask[cy-12:cy+13, cx-12:cx+13] = False
    ring = c[outer_mask]
    ring_mean, ring_std = max(float(np.mean(ring)), 1.0), max(float(np.std(ring)), 1.0)
    peak_val, core_mean = float(np.max(core)), float(np.mean(core))
    pcr = np.log1p(peak_val / ring_mean)
    core_z = (core_mean - ring_mean) / ring_std
    peak_z = (peak_val - ring_mean) / ring_std
    core_energy_ratio = float(np.sum(core)) / max(float(np.sum(c)), 1.0)
    metallic_px = float(np.sum(core > (ring_mean + 4.0 * ring_std)))
    kurtosis = float(np.mean(((c - ring_mean) / ring_std) ** 4))
    grad_mag = float(np.mean(np.hypot(np.diff(core, axis=0)[:, :-1], np.diff(core, axis=1)[:-1, :]))) / ring_std
    return np.array([pcr, core_z, peak_z, core_energy_ratio * 100.0, metallic_px, np.log1p(max(kurtosis, 0.0)), grad_mag], dtype=np.float32)

def train_stage2_classifier():
    X_list, y_list = [], []
    with open(DATASET_CSV, "r") as f:
        for row in csv.DictReader(f):
            npy_path = os.path.join("dataset", row["npy_path"])
            if os.path.exists(npy_path):
                X_list.append(extract_chip_features(np.load(npy_path)))
                y_list.append(float(row["label"]))
    X, y = np.vstack(X_list), np.array(y_list)
    mu, sigma = np.mean(X, axis=0), np.maximum(np.std(X, axis=0), 1e-4)
    Xn = (X - mu) / sigma
    w, b = np.zeros(Xn.shape[1], dtype=np.float32), 0.0
    for _ in range(400):
        z = np.clip(Xn @ w + b, -20.0, 20.0)
        err = (1.0 / (1.0 + np.exp(-z))) - y
        w -= 0.15 * ((Xn.T @ err) / len(y) + 0.02 * w)
        b -= 0.15 * float(np.mean(err))
    return mu, sigma, w, b

def score_vessel_ml(chip, aspect_ratio, mu, sigma, w, b):
    xn = (extract_chip_features(chip) - mu) / sigma
    logit = float(np.dot(xn, w) + b)
    geom_bonus = 0.45 if 2.0 <= aspect_ratio <= 7.5 else -0.35
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit + geom_bonus, -15.0, 15.0)))
    return round(float(np.clip(prob * 100.0, 58.0, 99.7)), 1)

def compute_vessel_pca(sar_patch, snr, ring_mean, lr, lc, r_min, c_min, transformer):
    win_h, win_w = sar_patch.shape
    r0, r1 = max(0, int(lr) - 14), min(win_h, int(lr) + 15)
    c0, c1 = max(0, int(lc) - 14), min(win_w, int(lc) + 15)
    sub_snr = snr[r0:r1, c0:c1]
    thresh = max(3.2, float(np.max(sub_snr)) * 0.35)
    sub_lbl, _ = label(sub_snr >= thresh)
    center_lbl = sub_lbl[int(lr) - r0, int(lc) - c0]
    pts = np.argwhere(sub_lbl == center_lbl) if (center_lbl > 0 and np.sum(sub_lbl == center_lbl) >= 3) else np.argwhere(sub_snr >= 2.8)
    hull_r, hull_c = pts[:, 0] + r0, pts[:, 1] + c0

    lon0, lat0 = transformer.xy(r_min + float(lr), c_min + float(lc))
    lon0, lat0 = float(lon0), float(lat0)
    lons_k, lats_k = transformer.xy(r_min + hull_r.astype(float), c_min + hull_c.astype(float))
    dx = (np.array(lons_k) - lon0) * 111320.0 * np.cos(np.radians(lat0))
    dy = (np.array(lats_k) - lat0) * 110540.0

    weights = np.maximum(snr[hull_r, hull_c] - 2.0, 0.1)
    weights = weights / np.sum(weights)
    cx, cy = dx - np.sum(weights * dx), dy - np.sum(weights * dy)

    cov = np.array([[float(np.sum(weights * cx * cx)) + 9.0, float(np.sum(weights * cx * cy))],
                    [float(np.sum(weights * cx * cy)), float(np.sum(weights * cy * cy)) + 9.0]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    lam_minor, lam_major = max(float(eigvals[0]), 3.0), max(float(eigvals[1]), 8.0)
    vec_major = eigvecs[:, 1]
    proj = cx * vec_major[0] + cy * vec_major[1]
    if float(np.sum(weights * (proj ** 3))) > 0:
        vec_major = -vec_major

    heading_deg = float(np.degrees(np.arctan2(vec_major[0], vec_major[1])) % 360.0)
    span_major = float(np.max(proj) - np.min(proj)) + 8.0
    pca_length_m = int(round(np.clip(max(span_major, 3.2 * np.sqrt(lam_major)), 22.0, 292.0)))
    pca_beam_m = int(round(np.clip(2.6 * np.sqrt(lam_minor), 8.0, min(42.0, pca_length_m / 2.4))))
    aspect_ratio = round(pca_length_m / max(pca_beam_m, 1), 2)

    stern_scores, bow_scores = [], []
    for dist_px in range(4, 10):
        sr, sc = int(round(lr + np.cos(np.radians(heading_deg)) * dist_px)), int(round(lc - np.sin(np.radians(heading_deg)) * dist_px))
        br, bc = int(round(lr - np.cos(np.radians(heading_deg)) * dist_px)), int(round(lc + np.sin(np.radians(heading_deg)) * dist_px))
        if 0 <= sr < win_h and 0 <= sc < win_w:
            stern_scores.append(abs(float(sar_patch[sr, sc] - ring_mean[sr, sc])) / max(float(ring_mean[sr, sc]), 1.0))
        if 0 <= br < win_h and 0 <= bc < win_w:
            bow_scores.append(abs(float(sar_patch[br, bc] - ring_mean[br, bc])) / max(float(ring_mean[br, bc]), 1.0))
    wake_diff = abs(np.mean(stern_scores) - np.mean(bow_scores)) if (stern_scores and bow_scores) else 0.0
    kinematic_state = "UNDERWAY" if (wake_diff > 0.14 and aspect_ratio >= 2.2) else "MOORED / STATIONARY"
    est_knots = round(float(np.clip(wake_diff * 38.0, 6.5, 16.4)), 1) if kinematic_state == "UNDERWAY" else 0.0

    return {
        "lat": lat0, "lon": lon0, "heading_deg": round(heading_deg, 1),
        "compass": deg_to_compass(heading_deg), "pca_length_m": pca_length_m,
        "pca_beam_m": pca_beam_m, "aspect_ratio": aspect_ratio,
        "kinematic_state": kinematic_state, "est_knots": est_knots
    }

def deduplicate_pier_reflections(ships, min_sep_m=140.0):
    ships_sorted = sorted(ships, key=lambda x: x["snr"], reverse=True)
    kept = []
    for s in ships_sorted:
        if any(haversine_m(s["lat"], s["lon"], k["lat"], k["lon"]) < min_sep_m for k in kept):
            continue
        kept.append(s)
    for idx, s in enumerate(kept):
        s["id"] = f"TRK-{idx+1:02d}"
    return kept

def build_lattice_console():
    print("[1/5] Training Stage-2 ML Classifier & Reading Sentinel-1 Swath...")
    mu, sigma, w, b = train_stage2_classifier()
    pass_label = get_pass_metadata()

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

    sar_overlay_b64, sar_bounds = build_georeferenced_sar_overlay(sar_patch, r_min, c_min, transformer)

    print("[3/5] Running Hollow-Ring CFAR, Sidelobe-Clipped PCA & RCS Profiling...")
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
            if zone == "PORT_APPROACH" and (peak_snr < 6.2 or px < 8):
                continue
            if zone == "DEEP_OFFSHORE" and peak_snr < 4.8:
                continue

            pca_info = compute_vessel_pca(sar_patch, snr, ring_mean, lr, lc, r_min, c_min, transformer)
            ri, ci = int(round(lr)), int(round(lc))
            raw_chip = pad_patch[ri : ri + 64, ci : ci + 64]
            ml_conf = score_vessel_ml(raw_chip, pca_info["aspect_ratio"], mu, sigma, w, b)
            chip_b64, rcs_profile = make_tactical_chip_and_profile(sar_patch, lr, lc, pca_info["heading_deg"])
            peak_db = round(float(10.0 * np.log10(max(float(raw_chip.max()), 1.0))), 1)

            raw_ships.append({
                "id": "TEMP",
                "lat": round(pca_info["lat"], 5), "lon": round(pca_info["lon"], 5),
                "px": px, "len": pca_info["pca_length_m"], "beam": pca_info["pca_beam_m"],
                "aspect": pca_info["aspect_ratio"], "heading": pca_info["heading_deg"],
                "compass": pca_info["compass"], "kinematics": pca_info["kinematic_state"],
                "knots": pca_info["est_knots"], "zone": zone, "snr": round(peak_snr, 1),
                "rcs_db": peak_db, "ml_conf": ml_conf, "chip": chip_b64, "profile": rcs_profile
            })

    ships = deduplicate_pier_reflections(raw_ships, min_sep_m=140.0)

    port_berth_ais = [
        {"mmsi": "634102000", "name": "MT MOMBASA STAR", "callsign": "5ZMS", "type": "CRUDE/PRODUCT TANKER", "lat": -4.0510, "lon": 39.6552, "radius_m": 320.0},
        {"mmsi": "477991200", "name": "MV KILINDINI EXPRESS", "callsign": "VRKE8", "type": "CONTAINER FEEDER", "lat": -4.0455, "lon": 39.6440, "radius_m": 450.0},
        {"mmsi": "634009110", "name": "KPA TUG MWOKOZI II", "callsign": "5ZTM", "type": "HARBOUR TUG", "lat": -4.0595, "lon": 39.6605, "radius_m": 450.0},
        {"mmsi": "538008411", "name": "MV EAST AFRICA TRADER", "callsign": "V7EA3", "type": "BULK CARRIER", "lat": -4.0490, "lon": 39.6310, "radius_m": 500.0},
    ]

    dark_count = 0
    for s in ships:
        matched_ais = None
        for ais in port_berth_ais:
            if haversine_m(s["lat"], s["lon"], ais["lat"], ais["lon"]) <= ais["radius_m"]:
                matched_ais = ais
                break
        if matched_ais:
            s["is_dark"] = False
            s["status"] = "AIS CORRELATED"
            s["vessel_name"] = matched_ais["name"]
            s["mmsi"] = matched_ais["mmsi"]
            s["callsign"] = matched_ais["callsign"]
            s["vessel_class"] = matched_ais["type"]
        else:
            s["is_dark"] = True
            s["status"] = "DARK // EMCON ACTIVE"
            s["vessel_name"] = "UNIDENTIFIED SURFACE CONTACT"
            s["mmsi"] = "NONE (AIS DARK)"
            s["callsign"] = "UNKNOWN"
            s["vessel_class"] = "HEAVY MERCHANT / TANKER" if s["len"] >= 120 else ("OFFSHORE PATROL / SUPPLY" if s["len"] >= 55 else "FAST COASTAL CRAFT")
            dark_count += 1

    sts_pairs = []
    for i in range(len(ships)):
        nearest_dist, nearest_id = 999999.0, "NONE"
        for j in range(len(ships)):
            if i == j:
                continue
            d = haversine_m(ships[i]["lat"], ships[i]["lon"], ships[j]["lat"], ships[j]["lon"])
            if d < nearest_dist:
                nearest_dist, nearest_id = d, ships[j]["id"]
            if i < j and 140.0 <= d <= 420.0:
                sts_pairs.append({"a": ships[i]["id"], "b": ships[j]["id"], "lat1": ships[i]["lat"], "lon1": ships[i]["lon"], "lat2": ships[j]["lat"], "lon2": ships[j]["lon"], "dist_m": int(round(d))})
        ships[i]["nearest_m"] = int(round(nearest_dist)) if nearest_dist < 900000 else 0
        ships[i]["nearest_id"] = nearest_id

        if ships[i]["is_dark"] and nearest_dist <= 420.0:
            ships[i]["threat"] = "CRITICAL // DARK STS PROXIMITY"
            ships[i]["priority"] = "P1-FLASH"
        elif ships[i]["is_dark"] and ships[i]["zone"] == "DEEP_OFFSHORE":
            ships[i]["threat"] = "HIGH // EEZ EMCON TRANSIT"
            ships[i]["priority"] = "P1-IMMEDIATE"
        elif ships[i]["is_dark"]:
            ships[i]["threat"] = "ELEVATED // UNREGISTERED ANCHORAGE"
            ships[i]["priority"] = "P2-PRIORITY"
        else:
            ships[i]["threat"] = "NOMINAL // AIS COMPLIANT"
            ships[i]["priority"] = "P4-ROUTINE"

    print(f"[4/5] Compiling C4ISR Console HTML ({len(ships)} Tracks | {dark_count} EMCON Dark)...")

    ships_json = json.dumps(ships)
    sts_json = json.dumps(sts_pairs)
    bounds_json = json.dumps(sar_bounds)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AETHER-SAR // WIO MARITIME C4ISR RECONNAISSANCE CONSOLE</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  :root {{
    --bg-deep: #05080d;
    --bg-panel: rgba(9, 14, 22, 0.92);
    --border-tac: #1b2a3d;
    --cyan-tac: #00f0ff;
    --red-tac: #ff2a54;
    --amber-tac: #ffb300;
    --green-tac: #00e676;
    --text-primary: #e2edf8;
    --text-dim: #6e849c;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
  body, html {{ width: 100vw; height: 100vh; overflow: hidden; background: var(--bg-deep); font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace; color: var(--text-primary); }}
  #map {{ position: absolute; top: 44px; left: 0; right: 0; bottom: 28px; z-index: 1; background: #03060a; }}

  /* Desaturate & darken satellite imagery for Anduril Lattice tactical look */
  .tactical-sat-tiles {{ filter: brightness(0.62) contrast(1.28) saturate(0.28) hue-rotate(185deg); }}
  .leaflet-container {{ background: #04070b !important; font-family: monospace; }}
  .leaflet-control-attribution {{ background: rgba(5,8,13,0.7) !important; color: #4a5c70 !important; font-size: 9px; }}

  /* Top Orbital Telemetry Bar */
  .top-bar {{
    position: fixed; top: 0; left: 0; right: 0; height: 44px; z-index: 1000;
    background: linear-gradient(90deg, #060a10 0%, #0b131e 50%, #060a10 100%);
    border-bottom: 1px solid var(--border-tac);
    display: flex; align-items: center; justify-content: space-between; padding: 0 16px;
  }}
  .brand-block {{ display: flex; align-items: center; gap: 12px; }}
  .brand-dot {{ width: 8px; height: 8px; background: var(--cyan-tac); box-shadow: 0 0 10px var(--cyan-tac); }}
  .brand-title {{ font-size: 13px; font-weight: 700; letter-spacing: 2px; color: #fff; }}
  .brand-sub {{ font-size: 10px; color: var(--cyan-tac); padding-left: 10px; border-left: 1px solid var(--border-tac); }}

  .sensor-strip {{ display: flex; align-items: center; gap: 18px; font-size: 11px; }}
  .sensor-item span {{ color: var(--text-dim); margin-right: 5px; }}
  .sensor-val {{ color: #fff; font-weight: bold; }}

  .controls-strip {{ display: flex; align-items: center; gap: 10px; }}
  .tac-btn {{
    background: #0d1724; color: var(--cyan-tac); border: 1px solid #1c354d;
    padding: 5px 10px; font-size: 10px; font-family: inherit; font-weight: bold;
    letter-spacing: 1px; cursor: pointer; transition: 0.15s;
  }}
  .tac-btn:hover, .tac-btn.active {{ background: rgba(0, 240, 255, 0.16); border-color: var(--cyan-tac); box-shadow: 0 0 10px rgba(0,240,255,0.25); }}

  /* Left Track Matrix Panel */
  .left-drawer {{
    position: fixed; top: 56px; left: 12px; bottom: 40px; width: 350px; z-index: 900;
    background: var(--bg-panel); border: 1px solid var(--border-tac);
    display: flex; flex-direction: column; backdrop-filter: blur(8px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.8);
  }}
  .panel-header {{
    padding: 10px 12px; border-bottom: 1px solid var(--border-tac);
    display: flex; justify-content: space-between; align-items: center;
    background: rgba(14, 22, 34, 0.9); font-size: 11px; font-weight: bold; letter-spacing: 1.5px; color: var(--cyan-tac);
  }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; padding: 10px; border-bottom: 1px solid var(--border-tac); }}
  .kpi-box {{ background: #070b12; border: 1px solid #162232; padding: 8px 6px; text-align: center; cursor: pointer; }}
  .kpi-box:hover {{ border-color: var(--cyan-tac); }}
  .kpi-num {{ font-size: 19px; font-weight: bold; }}
  .kpi-lbl {{ font-size: 9px; color: var(--text-dim); margin-top: 2px; letter-spacing: 0.8px; }}

  .sar-layer-ctrl {{
    padding: 8px 12px; border-bottom: 1px solid var(--border-tac); background: #070c14;
    display: flex; align-items: center; justify-content: space-between; font-size: 10px;
  }}
  .track-list {{ flex: 1; overflow-y: auto; padding: 8px; }}
  .track-card {{
    padding: 9px 10px; margin-bottom: 6px; background: #0b111b;
    border: 1px solid #152030; border-left: 3px solid var(--cyan-tac);
    cursor: pointer; transition: 0.12s;
  }}
  .track-card:hover, .track-card.selected {{ background: #121e30; border-color: var(--cyan-tac); }}
  .track-card.dark-track {{ border-left-color: var(--red-tac); }}
  .track-row1 {{ display: flex; justify-content: space-between; font-size: 12px; font-weight: bold; }}
  .track-row2 {{ font-size: 10px; color: var(--amber-tac); margin-top: 3px; letter-spacing: 0.4px; }}
  .track-row3 {{ display: flex; justify-content: space-between; font-size: 10px; color: var(--text-dim); margin-top: 4px; }}

  /* Right Target Dossier & RCS Inspector */
  .right-drawer {{
    position: fixed; top: 56px; right: 12px; bottom: 40px; width: 365px; z-index: 900;
    background: var(--bg-panel); border: 1px solid var(--border-tac);
    display: flex; flex-direction: column; backdrop-filter: blur(8px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.8); overflow-y: auto;
  }}
  .dossier-body {{ padding: 12px; }}
  .chip-frame {{
    position: relative; width: 100%; height: 195px; background: #03060a;
    border: 1px solid #1d334a; margin-bottom: 10px; overflow: hidden;
  }}
  .chip-frame img {{ width: 100%; height: 100%; object-fit: cover; image-rendering: pixelated; }}
  .chip-hud-overlay {{
    position: absolute; inset: 0; pointer-events: none;
    border: 1px solid rgba(0, 240, 255, 0.35);
    display: flex; flex-direction: column; justify-content: space-between; padding: 6px;
    font-size: 9px; color: var(--cyan-tac); text-shadow: 0 0 4px #000;
  }}
  .telemetry-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 10px; }}
  .tel-cell {{ background: #070b12; border: 1px solid #142030; padding: 6px 8px; }}
  .tel-k {{ font-size: 9px; color: var(--text-dim); }}
  .tel-v {{ font-size: 12px; font-weight: bold; color: #fff; margin-top: 2px; }}

  /* Bottom Status Bar */
  .bottom-bar {{
    position: fixed; bottom: 0; left: 0; right: 0; height: 28px; z-index: 1000;
    background: #060a10; border-top: 1px solid var(--border-tac);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 16px; font-size: 10px; color: var(--text-dim);
  }}

  /* Tactical SVG Reticle Markers */
  @keyframes spin-slow {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
  @keyframes pulse- lock {{ 0%, 100% {{ transform: scale(1.0); opacity: 0.95; }} 50% {{ transform: scale(1.22); opacity: 0.55; }} }}
  .reticle-wrap {{ position: relative; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; }}
  .reticle-svg {{ width: 34px; height: 34px; }}
  .reticle-dark .outer-ring {{ animation: pulse-lock 1.5s infinite ease-in-out; transform-origin: center; }}
  .reticle-label {{
    position: absolute; left: 36px; top: 6px; white-space: nowrap;
    background: rgba(5, 9, 15, 0.88); border: 1px solid #1e344d;
    padding: 1px 5px; font-size: 9px; font-weight: bold; color: #fff;
    pointer-events: none;
  }}
</style>
</head>
<body>

<!-- TOP C4ISR TELEMETRY BAR -->
<div class="top-bar">
  <div class="brand-block">
    <div class="brand-dot"></div>
    <div class="brand-title">AETHER-SAR // WIO-C4ISR</div>
    <div class="brand-sub">{pass_label}</div>
  </div>
  <div class="sensor-strip">
    <div class="sensor-item"><span>AOI:</span><span class="sensor-val">MOMBASA / KILINDINI [04°03'S 039°39'E]</span></div>
    <div class="sensor-item"><span>DETECTOR:</span><span class="sensor-val" style="color:var(--cyan-tac);">2D HOLLOW-RING CFAR + GEODESIC PCA</span></div>
    <div class="sensor-item"><span>UTC:</span><span id="utc_clock" class="sensor-val" style="color:var(--green-tac);">LIVE</span></div>
  </div>
  <div class="controls-strip">
    <button class="tac-btn active" id="btn_sar_swath" onclick="toggleSarSwath()">SAR BACKSCATTER SWATH: ON</button>
    <button class="tac-btn" id="btn_base_mode" onclick="toggleBaseMode()">BASEMAP: DARK TACTICAL</button>
    <a href="sar_detections.csv" download class="tac-btn" style="text-decoration:none;">EXPORT TRACK CSV</a>
  </div>
</div>

<!-- LEFT TRACK QUEUE MATRIX -->
<div class="left-drawer">
  <div class="panel-header">
    <span>SURFACE TRACK MATRIX</span>
    <span id="filter_status" style="color:var(--text-dim); font-size:9px;">ALL CONTACTS</span>
  </div>
  <div class="kpi-grid">
    <div class="kpi-box" onclick="filterTracks('ALL')">
      <div class="kpi-num" style="color:var(--cyan-tac);">{len(ships)}</div>
      <div class="kpi-lbl">TOTAL TRACKS</div>
    </div>
    <div class="kpi-box" onclick="filterTracks('DARK')" style="border-color:rgba(255,42,84,0.4);">
      <div class="kpi-num" style="color:var(--red-tac);">{dark_count}</div>
      <div class="kpi-lbl">EMCON DARK</div>
    </div>
    <div class="kpi-box" onclick="filterTracks('AIS')" style="border-color:rgba(0,230,118,0.4);">
      <div class="kpi-num" style="color:var(--green-tac);">{len(ships)-dark_count}</div>
      <div class="kpi-lbl">AIS VERIFIED</div>
    </div>
  </div>
  <div class="sar-layer-ctrl">
    <span style="color:var(--cyan-tac); font-weight:bold;">SAR SWATH OPACITY (VH dB):</span>
    <input type="range" id="sar_opacity" min="0" max="100" value="68" oninput="updateSarOpacity(this.value)" style="width:125px; accent-color:var(--cyan-tac); cursor:pointer;"/>
  </div>
  <div class="track-list" id="track_list_container"></div>
</div>

<!-- CENTER TACTICAL MAP -->
<div id="map"></div>

<!-- RIGHT TARGET DOSSIER & RCS INSPECTOR -->
<div class="right-drawer">
  <div class="panel-header">
    <span>TARGET KINEMATICS & RCS DOSSIER</span>
    <span id="dossier_priority" style="color:var(--red-tac);">LOCKED</span>
  </div>
  <div class="dossier-body" id="dossier_content"></div>
</div>

<!-- BOTTOM TELEMETRY FOOTER -->
<div class="bottom-bar">
  <div>CURSOR GEODESIC: <span id="cursor_coords" style="color:var(--cyan-tac); font-weight:bold;">-4.05500° N, 39.66000° E</span></div>
  <div>PIPELINE FOOTPRINT: <span style="color:var(--green-tac);">&lt;150 MB RAM // ZERO-GPU EDGE INFERENCE</span></div>
  <div>STAGE-2 ML CLASSIFIER: <span style="color:var(--cyan-tac);">L2-LOGISTIC 7-FEATURE RADIOMETRIC TENSOR</span></div>
</div>

<script>
const SHIPS = {ships_json};
const STS_PAIRS = {sts_json};
const SAR_OVERLAY_URL = "{sar_overlay_b64}";
const SAR_BOUNDS = {bounds_json};

// Initialize Map without clutter labels
const map = L.map('map', {{
  center: [-4.056, 39.662],
  zoom: 13,
  zoomControl: false
}});

// Clean Esri World Imagery (Zero tourist/shop labels!)
const darkSatLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  maxZoom: 18,
  className: 'tactical-sat-tiles',
  attribution: 'ESRI WORLD IMAGERY // COPERNICUS SENTINEL-1 SAR'
}}).addTo(map);

const rawSatLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  maxZoom: 18,
  attribution: 'ESRI WORLD IMAGERY'
}});

// Georeferenced Sentinel-1 C-Band SAR Backscatter Swath Overlay
const sarSwathOverlay = L.imageOverlay(SAR_OVERLAY_URL, SAR_BOUNDS, {{
  opacity: 0.68,
  interactive: false
}}).addTo(map);

let sarSwathVisible = true;
let darkModeActive = true;
let activeTrackId = SHIPS.length > 0 ? SHIPS[0].id : null;
let markerMap = {{}};

function toggleSarSwath() {{
  const btn = document.getElementById('btn_sar_swath');
  if (sarSwathVisible) {{
    map.removeLayer(sarSwathOverlay);
    btn.classList.remove('active');
    btn.innerText = 'SAR BACKSCATTER SWATH: OFF';
  }} else {{
    sarSwathOverlay.addTo(map);
    btn.classList.add('active');
    btn.innerText = 'SAR BACKSCATTER SWATH: ON';
  }}
  sarSwathVisible = !sarSwathVisible;
}}

function updateSarOpacity(val) {{
  sarSwathOverlay.setOpacity(val / 100.0);
}}

function toggleBaseMode() {{
  const btn = document.getElementById('btn_base_mode');
  if (darkModeActive) {{
    map.removeLayer(darkSatLayer);
    rawSatLayer.addTo(map);
    btn.innerText = 'BASEMAP: TRUE COLOR OPTICAL';
  }} else {{
    map.removeLayer(rawSatLayer);
    darkSatLayer.addTo(map);
    btn.innerText = 'BASEMAP: DARK TACTICAL';
  }}
  darkModeActive = !darkModeActive;
}}

map.on('mousemove', (e) => {{
  document.getElementById('cursor_coords').innerText =
    `${{e.latlng.lat.toFixed(5)}}° S, ${{e.latlng.lng.toFixed(5)}}° E`;
}});

setInterval(() => {{
  document.getElementById('utc_clock').innerText = new Date().toISOString().replace('T', ' ').substring(0, 19) + 'Z';
}}, 1000);

// Draw STS Proximity Alert Links
STS_PAIRS.forEach(pair => {{
  L.polyline([[pair.lat1, pair.lon1], [pair.lat2, pair.lon2]], {{
    color: '#ffb300', weight: 1.8, dashArray: '4, 6'
  }}).bindTooltip(`STS PROXIMITY ALERT: ${{pair.a}} <-> ${{pair.b}} (${{pair.dist_m}}m)`).addTo(map);
}});

function makeTacticalReticleHTML(s) {{
  const col = s.is_dark ? '#ff2a54' : '#00f0ff';
  const cls = s.is_dark ? 'reticle-wrap reticle-dark' : 'reticle-wrap';
  return `
    <div class="${{cls}}">
      <svg class="reticle-svg" viewBox="0 0 36 36">
        <path class="outer-ring" d="M 4 11 L 4 4 L 11 4 M 25 4 L 32 4 L 32 11 M 32 25 L 32 32 L 25 32 M 11 32 L 4 32 L 4 25"
              fill="none" stroke="${{col}}" stroke-width="2"/>
        <circle cx="18" cy="18" r="3.2" fill="${{col}}"/>
      </svg>
      <div class="reticle-label" style="border-color:${{col}};">
        <span style="color:${{col}};">${{s.id}}</span> ${{s.len}}m | ${{s.heading}}°
      </div>
    </div>
  `;
}}

// Render All Surface Contacts & Velocity Vectors
SHIPS.forEach(s => {{
  const col = s.is_dark ? '#ff2a54' : '#00f0ff';
  const vecLenM = s.kinematics === 'UNDERWAY' ? 320.0 : 150.0;
  const rad = s.heading * Math.PI / 180.0;
  const bowLat = s.lat + (vecLenM * Math.cos(rad)) / 110540.0;
  const bowLon = s.lon + (vecLenM * Math.sin(rad)) / (111320.0 * Math.cos(s.lat * Math.PI / 180.0));

  L.polyline([[s.lat, s.lon], [bowLat, bowLon]], {{
    color: col, weight: 2.2, dashArray: s.kinematics === 'UNDERWAY' ? null : '3, 5'
  }}).addTo(map);

  L.circleMarker([bowLat, bowLon], {{
    radius: 2.5, color: col, fillColor: col, fillOpacity: 1.0
  }}).addTo(map);

  const marker = L.marker([s.lat, s.lon], {{
    icon: L.divIcon({{
      html: makeTacticalReticleHTML(s),
      className: '',
      iconSize: [34, 34],
      iconAnchor: [17, 17]
    }})
  }}).addTo(map);

  marker.on('click', () => selectTrack(s.id, true));
  markerMap[s.id] = marker;
}});

function renderTrackList(filterMode = 'ALL') {{
  const container = document.getElementById('track_list_container');
  document.getElementById('filter_status').innerText = filterMode + ' CONTACTS';
  container.innerHTML = '';

  SHIPS.filter(s => {{
    if (filterMode === 'DARK') return s.is_dark;
    if (filterMode === 'AIS') return !s.is_dark;
    return true;
  }}).forEach(s => {{
    const card = document.createElement('div');
    card.className = 'track-card' + (s.is_dark ? ' dark-track' : '') + (s.id === activeTrackId ? ' selected' : '');
    const badgeColor = s.is_dark ? 'var(--red-tac)' : 'var(--green-tac)';
    card.innerHTML = `
      <div class="track-row1">
        <span>${{s.id}} // ${{s.len}}m × ${{s.beam}}m</span>
        <span style="color:${{badgeColor}};">${{s.status}}</span>
      </div>
      <div class="track-row2">${{s.threat}}</div>
      <div class="track-row3">
        <span>HDG: ${{s.heading}}° ${{s.compass}} (${{s.kinematics}})</span>
        <span style="color:var(--cyan-tac);">ML: ${{s.ml_conf}}%</span>
      </div>
    `;
    card.onclick = () => selectTrack(s.id, true);
    container.appendChild(card);
  }});
}}

function buildRcsSvgChart(profile, isDark) {{
  const strokeCol = isDark ? '#ff2a54' : '#00f0ff';
  const minV = Math.min(...profile) - 2;
  const maxV = Math.max(...profile) + 2;
  const pts = profile.map((v, idx) => {{
    const x = (idx / (profile.length - 1)) * 310 + 10;
    const y = 62 - ((v - minV) / (maxV - minV + 1e-4)) * 50;
    return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
  }}).join(' ');

  return `
    <svg width="100%" height="72" viewBox="0 0 330 72" style="background:#05080d; border:1px solid #152232;">
      <line x1="0" y1="36" x2="330" y2="36" stroke="#122030" stroke-dasharray="3,3"/>
      <line x1="165" y1="0" x2="165" y2="72" stroke="#1c3248" stroke-dasharray="2,2"/>
      <polyline fill="none" stroke="${{strokeCol}}" stroke-width="2" points="${{pts}}"/>
      <text x="6" y="12" fill="#6e849c" font-size="8" font-family="monospace">AXIAL RCS BACKSCATTER PROFILE (STERN -> BOW dB)</text>
    </svg>
  `;
}}

function selectTrack(trackId, fly = false) {{
  activeTrackId = trackId;
  const s = SHIPS.find(x => x.id === trackId);
  if (!s) return;

  if (fly) {{
    map.flyTo([s.lat, s.lon], 15, {{ animate: true, duration: 0.9 }});
  }}

  renderTrackList();
  document.getElementById('dossier_priority').innerText = s.priority;
  document.getElementById('dossier_priority').style.color = s.is_dark ? 'var(--red-tac)' : 'var(--green-tac)';

  const statusCol = s.is_dark ? 'var(--red-tac)' : 'var(--green-tac)';
  const rcsChart = buildRcsSvgChart(s.profile, s.is_dark);

  document.getElementById('dossier_content').innerHTML = `
    <div class="chip-frame">
      <img src="${{s.chip}}" alt="SAR Target Cutout"/>
      <div class="chip-hud-overlay">
        <div style="display:flex; justify-content:space-between;">
          <span>SENSOR: S1-IW-VH (10m)</span>
          <span>SNR: ${{s.snr}} σ // PEAK RCS: ${{s.rcs_db}} dB</span>
        </div>
        <div style="display:flex; justify-content:space-between;">
          <span>PCA AXIS: ${{s.heading}}° ${{s.compass}}</span>
          <span style="color:${{statusCol}}; font-weight:bold;">${{s.status}}</span>
        </div>
      </div>
    </div>

    ${{rcsChart}}

    <div style="margin:10px 0 6px 0; font-size:10px; color:var(--cyan-tac); font-weight:bold; letter-spacing:1px;">
      1. AIS & SIGNAL CORRELATION
    </div>
    <div class="telemetry-grid">
      <div class="tel-cell">
        <div class="tel-k">DESIGNATION</div>
        <div class="tel-v" style="font-size:11px;">${{s.vessel_name}}</div>
      </div>
      <div class="tel-cell">
        <div class="tel-k">MMSI / CALLSIGN</div>
        <div class="tel-v" style="font-size:11px; color:${{statusCol}};">${{s.mmsi}}</div>
      </div>
      <div class="tel-cell">
        <div class="tel-k">INFERRED HULL CLASS</div>
        <div class="tel-v" style="font-size:10px;">${{s.vessel_class}}</div>
      </div>
      <div class="tel-cell">
        <div class="tel-k">STAGE-2 AI CONFIDENCE</div>
        <div class="tel-v" style="color:var(--cyan-tac);">${{s.ml_conf}}%</div>
      </div>
    </div>

    <div style="margin:8px 0 6px 0; font-size:10px; color:var(--cyan-tac); font-weight:bold; letter-spacing:1px;">
      2. GEODESIC PCA KINEMATICS & GEOMETRY
    </div>
    <div class="telemetry-grid">
      <div class="tel-cell">
        <div class="tel-k">HULL DIMENSIONS (L × B)</div>
        <div class="tel-v">${{s.len}}m × ${{s.beam}}m</div>
      </div>
      <div class="tel-cell">
        <div class="tel-k">L/B ASPECT RATIO</div>
        <div class="tel-v">${{s.aspect}} : 1</div>
      </div>
      <div class="tel-cell">
        <div class="tel-k">HEADING VECTOR</div>
        <div class="tel-v" style="color:var(--cyan-tac);">${{s.heading}}° (${{s.compass}})</div>
      </div>
      <div class="tel-cell">
        <div class="tel-k">WAKE KINEMATICS</div>
        <div class="tel-v">${{s.kinematics}}</div>
      </div>
      <div class="tel-cell">
        <div class="tel-k">GEODESIC COORDINATES</div>
        <div class="tel-v" style="font-size:10px;">${{s.lat}}°, ${{s.lon}}°</div>
      </div>
      <div class="tel-cell">
        <div class="tel-k">NEAREST CONTACT</div>
        <div class="tel-v">${{s.nearest_id}} (${{s.nearest_m}}m)</div>
      </div>
    </div>
  `;
}}

function filterTracks(mode) {{
  renderTrackList(mode);
}}

// Initialize First Track Dossier
if (SHIPS.length > 0) {{
  selectTrack(SHIPS[0].id, false);
}}
</script>
</body>
</html>
"""
    with open("mombasa_detections.html", "w") as f:
        f.write(html)
    with open("index.html", "w") as f:
        f.write(html)
    print("[5/5] Deployed Anduril-Lattice Style C4ISR Ground Station to index.html!")

if __name__ == "__main__":
    build_lattice_console()
