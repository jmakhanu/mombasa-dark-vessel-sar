import os
import csv
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import GCPTransformer
from scipy.ndimage import uniform_filter, maximum_filter, label, center_of_mass
import folium

TIFF_FILE = "mombasa_sar_vh.tiff"
CSV_FILE = "sar_detections.csv"

# Mombasa Port + Offshore Indian Ocean Scan Box
MIN_LON, MAX_LON = 39.61, 40.00
MIN_LAT, MAX_LAT = -4.25, -3.95

# Exact Mombasa Outer Coastline Profile (South to North: Tiwi -> Likoni -> Nyali -> Bamburi -> Mtwapa)
COAST_LATS = [-4.30, -4.20, -4.12, -4.08, -4.05, -4.00, -3.95, -3.90]
COAST_LONS = [39.610, 39.638, 39.662, 39.675, 39.688, 39.730, 39.762, 39.795]

def classify_water_zone(lat, lon):
    # 1. Check if inside Kilindini Harbour / Port Reitz water channel (excludes inland Changamwe/Island)
    in_kilindini_water = (-4.083 <= lat <= -4.050) and (39.615 <= lon <= 39.663)
    if in_kilindini_water:
        return "PORT/INSHORE"
    
    # 2. Check if East of the Mombasa outer coastline (Open Indian Ocean + Outer Anchorage)
    shoreline_lon = np.interp(lat, COAST_LATS, COAST_LONS)
    if lon > (shoreline_lon + 0.004):  # 400m buffer past breaking beach surf
        return "OFFSHORE"
        
    return "LAND"

def run_low_ram_cfar():
    if not os.path.exists(TIFF_FILE):
        print(f"Error: {TIFF_FILE} not found!")
        return

    print(f"[1/4] Mapping satellite swath over Mombasa & Indian Ocean...")
    with rasterio.open(TIFF_FILE) as src:
        gcps, _ = src.gcps
        transformer = GCPTransformer(gcps)
        
        grid_r, grid_c = np.meshgrid(
            np.linspace(0, src.height - 1, 60),
            np.linspace(0, src.width - 1, 60)
        )
        lons, lats = transformer.xy(grid_r.ravel(), grid_c.ravel())
        lons, lats = np.array(lons), np.array(lats)
        
        print(f"      Full Satellite Swath Covers Lon: {lons.min():.2f} to {lons.max():.2f}, Lat: {lats.min():.2f} to {lats.max():.2f}")
        
        in_box = (lons >= MIN_LON) & (lons <= MAX_LON) & (lats >= MIN_LAT) & (lats <= MAX_LAT)
        valid_r = grid_r.ravel()[in_box]
        valid_c = grid_c.ravel()[in_box]
        r_min = int(max(0, np.min(valid_r) - 200))
        r_max = int(min(src.height, np.max(valid_r) + 200))
        c_min = int(max(0, np.min(valid_c) - 200))
        c_max = int(min(src.width, np.max(valid_c) + 200))

        win_h, win_w = int(r_max - r_min), int(c_max - c_min)
        print(f"[2/4] Slicing Ocean Window ({win_w}x{win_h} px) into RAM...")
        window = Window(c_min, r_min, win_w, win_h)
        sar_patch = src.read(1, window=window).astype(np.float32)

    print("[3/4] Running Dual-Zone Hollow-Ring CFAR Detector...")
    valid_mask = sar_patch > 0
    
    # Hollow Ring Mean (91x91 outer background minus 25x25 inner guard band around ship)
    outer_sz, guard_sz = 91.0, 25.0
    ring_area = (outer_sz ** 2) - (guard_sz ** 2)
    sum_outer = uniform_filter(sar_patch, size=int(outer_sz)) * (outer_sz ** 2)
    sum_guard = uniform_filter(sar_patch, size=int(guard_sz)) * (guard_sz ** 2)
    ring_mean = np.maximum((sum_outer - sum_guard) / ring_area, 1.0)
    
    # Estimate local wave clutter standard deviation without ship spikes
    clipped = np.minimum(sar_patch, ring_mean * 2.2)
    bg_std = np.sqrt(np.maximum(uniform_filter(clipped ** 2, size=91) - uniform_filter(clipped, size=91) ** 2, 1.0))
    
    # 3x3 smoothed target signal
    target_signal = uniform_filter(sar_patch, size=3)
    snr = (target_signal - ring_mean) / bg_std
    
    # Candidate peaks (4.8 sigma catches smaller offshore vessels; we filter strictly by zone next)
    candidate_pixels = (snr > 4.8) & (target_signal > ring_mean * 1.8) & valid_mask
    # Merge closely spaced reflections from the same ship hull (bow + superstructure)
    candidate_pixels = maximum_filter(candidate_pixels, size=3)

    labeled_ships, num_features = label(candidate_pixels)
    ships = []
    
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled_ships == i)
        pixel_count = len(coords)
        
        if 4 <= pixel_count <= 600:
            local_r, local_c = coords.mean(axis=0)
            peak_snr = float(np.max(snr[labeled_ships == i]))
            
            global_r = r_min + float(local_r)
            global_c = c_min + float(local_c)
            lon, lat = transformer.xy(global_r, global_c)
            lat, lon = float(lat), float(lon)
            
            zone = classify_water_zone(lat, lon)
            if zone == "LAND":
                continue
            
            # Port water is noisy (require 6.2 sigma & >=9 px); Open Ocean is clean (require 5.0 sigma)
            if zone == "PORT/INSHORE" and (peak_snr < 6.2 or pixel_count < 9):
                continue
            if zone == "OFFSHORE" and peak_snr < 5.0:
                continue
                
            est_length = max(15, int(np.sqrt(pixel_count) * 12))
            ships.append((lat, lon, pixel_count, est_length, zone, peak_snr))

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_id", "lat", "lon", "pixel_cluster", "est_length_m", "zone", "snr_sigma"])
        for idx, (lat, lon, px, length, zone, peak_snr) in enumerate(ships, 1):
            writer.writerow([f"SAR-{idx:02d}", f"{lat:.5f}", f"{lon:.5f}", px, length, zone, f"{peak_snr:.1f}"])

    offshore_count = sum(1 for s in ships if s[4] == "OFFSHORE")
    port_count = len(ships) - offshore_count
    print(f"[4/4] Success! Found {offshore_count} OFFSHORE vessels and {port_count} PORT vessels.")

    # Use Esri World Satellite Imagery as Primary Default (Never blocks local files!)
    m = folium.Map(
        location=[-4.07, 39.72],
        zoom_start=12,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Satellite Imagery"
    )

    # Secondary Layer: Google Hybrid (Satellite + Shipping Labels, also never blocks!)
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid Satellite",
        name="Google Satellite + Labels",
        overlay=False,
        control=True
    ).add_to(m)

    # Visualize the Coastline Land-Mask Boundary (Yellow Line)
    coast_points = [[lat, lon + 0.004] for lat, lon in zip(COAST_LATS, COAST_LONS) if -4.26 <= lat <= -3.94]
    folium.PolyLine(
        coast_points,
        color="#ffcc00",
        weight=2,
        dash_array="6, 6",
        tooltip="Offshore Radar Mask Boundary"
    ).add_to(m)

    for idx, (lat, lon, px_size, est_length, zone, peak_snr) in enumerate(ships, 1):
        color = "#ff2222" if zone == "OFFSHORE" else "#ff9900"
        folium.CircleMarker(
            location=[lat, lon],
            radius=8 if zone == "OFFSHORE" else 6,
            color="#ffffff",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            tooltip=f"SAR-{idx:02d} [{zone}] (~{est_length}m | {peak_snr:.1f}σ)",
            popup=(
                f"<b>Target ID: SAR-{idx:02d}</b><br>"
                f"Zone: <b>{zone}</b><br>"
                f"Coords: {lat:.5f}, {lon:.5f}<br>"
                f"Radar SNR: {peak_snr:.1f}σ<br>"
                f"Est. Hull Size: ~{est_length}m"
            ),
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save("mombasa_detections.html")
    print("\nDONE! Refresh Firefox (F5) — Satellite map will load immediately with zero 403 errors!")

if __name__ == "__main__":
    run_low_ram_cfar()
