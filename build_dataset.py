import os
import csv
import json
import zlib
import struct
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import GCPTransformer
from scipy.ndimage import uniform_filter, maximum_filter, label

TIFF_FILE = "mombasa_sar_vh.tiff"
DATASET_DIR = "dataset"
CHIP_SIZE = 64
HALF = CHIP_SIZE // 2

MIN_LON, MAX_LON = 39.61, 39.95
MIN_LAT, MAX_LAT = -4.22, -3.96
COAST_LATS = [-4.30, -4.20, -4.12, -4.08, -4.05, -4.00, -3.95, -3.90]
COAST_LONS = [39.610, 39.638, 39.662, 39.675, 39.688, 39.730, 39.762, 39.795]

def save_grayscale_png(filepath, chip):
    p2, p99 = np.percentile(chip, 2), max(np.percentile(chip, 99.2), 1.0)
    norm = np.clip((chip - p2) / (p99 - p2 + 1e-5), 0, 1)
    gray = (np.sqrt(norm) * 255).astype(np.uint8)
    h, w = gray.shape
    raw_data = b"".join(b"\x00" + gray[y].tobytes() for y in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw_data, 6))
    png += chunk(b"IEND", b"")
    with open(filepath, "wb") as f:
        f.write(png)

def classify_water_zone(lat, lon):
    if (-4.083 <= lat <= -4.050) and (39.615 <= lon <= 39.663):
        return "PORT_INSHORE"
    shoreline_lon = np.interp(lat, COAST_LATS, COAST_LONS)
    if lon > (shoreline_lon + 0.004):
        return "OFFSHORE"
    return "LAND"

def extract_chip(sar_patch, r, c):
    r, c = int(round(r)), int(round(c))
    pad_patch = np.pad(sar_patch, HALF, mode="reflect")
    return pad_patch[r : r + CHIP_SIZE, c : c + CHIP_SIZE].copy()

def build_kaggle_style_dataset():
    os.makedirs(f"{DATASET_DIR}/chips_npy", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/chips_png", exist_ok=True)

    print("[1/4] Slicing Mombasa Maritime ROI from Sentinel-1 TIFF...")
    with rasterio.open(TIFF_FILE) as src:
        gcps, _ = src.gcps
        transformer = GCPTransformer(gcps)
        grid_r, grid_c = np.meshgrid(np.linspace(0, src.height - 1, 60), np.linspace(0, src.width - 1, 60))
        lons, lats = transformer.xy(grid_r.ravel(), grid_c.ravel())
        lons, lats = np.array(lons), np.array(lats)
        in_box = (lons >= MIN_LON) & (lons <= MAX_LON) & (lats >= MIN_LAT) & (lats <= MAX_LAT)
        valid_r, valid_c = grid_r.ravel()[in_box], grid_c.ravel()[in_box]
        r_min, r_max = int(max(0, np.min(valid_r) - 100)), int(min(src.height, np.max(valid_r) + 100))
        c_min, c_max = int(max(0, np.min(valid_c) - 100)), int(min(src.width, np.max(valid_c) + 100))

        win_h, win_w = int(r_max - r_min), int(c_max - c_min)
        sar_patch = src.read(1, window=Window(c_min, r_min, win_w, win_h)).astype(np.float32)

    print("[2/4] Saving compressed Port ROI (.npz) for fast reloading...")
    port_sub = sar_patch[:min(1400, win_h), :min(1400, win_w)].astype(np.float16)
    np.savez_compressed(f"{DATASET_DIR}/mombasa_harbour_roi.npz", sar_vh=port_sub)

    print("[3/4] Extracting Positive Ship Chips (label=1) & Negative Clutter Chips (label=0)...")
    valid_mask = sar_patch > 0
    outer_sz, guard_sz = 91.0, 25.0
    ring_area = (outer_sz ** 2) - (guard_sz ** 2)
    ring_mean = np.maximum((uniform_filter(sar_patch, size=91)*(91**2) - uniform_filter(sar_patch, size=25)*(25**2)) / ring_area, 1.0)
    clipped = np.minimum(sar_patch, ring_mean * 2.2)
    bg_std = np.sqrt(np.maximum(uniform_filter(clipped**2, size=91) - uniform_filter(clipped, size=91)**2, 1.0))
    target_signal = uniform_filter(sar_patch, size=3)
    snr = (target_signal - ring_mean) / bg_std

    candidates = maximum_filter((snr > 4.6) & (target_signal > ring_mean * 1.75) & valid_mask, size=3)
    labeled, num_features = label(candidates)
    
    records = []
    ship_centers = []

    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled == i)
        px = len(coords)
        if 4 <= px <= 600:
            lr, lc = coords.mean(axis=0)
            peak_snr = float(np.max(snr[labeled == i]))
            lon, lat = transformer.xy(r_min + float(lr), c_min + float(lc))
            lat, lon = float(lat), float(lon)
            zone = classify_water_zone(lat, lon)
            if zone == "LAND":
                continue
            if zone == "PORT_INSHORE" and (peak_snr < 6.0 or px < 8):
                continue
            if zone == "OFFSHORE" and peak_snr < 4.8:
                continue

            sample_id = f"SHIP_{len(ship_centers)+1:03d}"
            chip = extract_chip(sar_patch, lr, lc)
            np.save(f"{DATASET_DIR}/chips_npy/{sample_id}.npy", chip.astype(np.float16))
            save_grayscale_png(f"{DATASET_DIR}/chips_png/{sample_id}.png", chip)

            ship_centers.append((lr, lc))
            records.append({
                "sample_id": sample_id,
                "label": 1,
                "class_name": "vessel",
                "zone": zone,
                "lat": f"{lat:.5f}",
                "lon": f"{lon:.5f}",
                "pixel_area": px,
                "est_length_m": max(15, int(np.sqrt(px) * 12)),
                "snr_sigma": f"{peak_snr:.2f}",
                "max_backscatter": f"{float(chip.max()):.1f}",
                "mean_clutter": f"{float(ring_mean[int(lr), int(lc)]):.1f}",
                "npy_path": f"chips_npy/{sample_id}.npy",
                "png_path": f"chips_png/{sample_id}.png"
            })

    np.random.seed(42)
    neg_count = 0
    attempts = 0
    while neg_count < len(ship_centers) and attempts < 2000:
        attempts += 1
        rr = np.random.randint(HALF, win_h - HALF)
        cc = np.random.randint(HALF, win_w - HALF)
        if sar_patch[rr, cc] <= 0 or snr[rr, cc] > 2.5:
            continue
        if any(np.hypot(rr - sr, cc - sc) < 80 for sr, sc in ship_centers):
            continue
        lon, lat = transformer.xy(r_min + float(rr), c_min + float(cc))
        lat, lon = float(lat), float(lon)
        zone = classify_water_zone(lat, lon)
        if zone == "LAND":
            continue

        neg_count += 1
        sample_id = f"CLUTTER_{neg_count:03d}"
        chip = extract_chip(sar_patch, rr, cc)
        np.save(f"{DATASET_DIR}/chips_npy/{sample_id}.npy", chip.astype(np.float16))
        save_grayscale_png(f"{DATASET_DIR}/chips_png/{sample_id}.png", chip)

        records.append({
            "sample_id": sample_id,
            "label": 0,
            "class_name": "ocean_clutter",
            "zone": zone,
            "lat": f"{lat:.5f}",
            "lon": f"{lon:.5f}",
            "pixel_area": 0,
            "est_length_m": 0,
            "snr_sigma": f"{float(snr[rr, cc]):.2f}",
            "max_backscatter": f"{float(chip.max()):.1f}",
            "mean_clutter": f"{float(ring_mean[rr, cc]):.1f}",
            "npy_path": f"chips_npy/{sample_id}.npy",
            "png_path": f"chips_png/{sample_id}.png"
        })

    csv_path = f"{DATASET_DIR}/metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    kaggle_meta = {
        "title": "Mombasa Sentinel-1 SAR Ship and Dark Vessel Dataset",
        "id": "idrissmakhanu/mombasa-sentinel1-sar-ships",
        "licenses": [{"name": "CC-BY-4.0"}]
    }
    with open(f"{DATASET_DIR}/dataset-metadata.json", "w") as f:
        json.dump(kaggle_meta, f, indent=2)

    print(f"[4/4] Dataset Built in '{DATASET_DIR}/' ({len(ship_centers)} vessels, {neg_count} clutter samples)")

if __name__ == "__main__":
    build_kaggle_style_dataset()
