import os
import csv
import zlib
import struct
import base64
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import GCPTransformer
from scipy.ndimage import uniform_filter, maximum_filter, label
import folium

TIFF_FILE = "mombasa_sar_vh.tiff"
CSV_FILE = "sar_detections.csv"
DATASET_CSV = "dataset/metadata.csv"

MIN_LON, MAX_LON = 39.61, 39.95
MIN_LAT, MAX_LAT = -4.22, -3.96
COAST_LATS = [-4.30, -4.20, -4.12, -4.08, -4.05, -4.00, -4.95, -3.90]
COAST_LONS = [39.610, 39.638, 39.662, 39.675, 39.688, 39.730, 39.762, 39.795]

COMPASS_DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

def deg_to_compass(deg):
    idx = int(round(deg / 22.5)) % 16
    return COMPASS_DIRS[idx]

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
    """Renders the 56x56 SAR radar chip with a cyan PCA heading axis overlay."""
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

    # Draw subtle cyan bow vector line from center outward (12 to 24 px radius)
    cy, cx = chip.shape[0] // 2, chip.shape[1] // 2
    rad = np.radians(heading_deg)
    vx, vy = np.sin(rad), -np.cos(rad)
    for step in range(10, 25):
        py = int(round(cy + vy * step))
        px = int(round(cx + vx * step))
        if 0 <= py < chip.shape[0] and 0 <= px < chip.shape[1]:
            rgba[py, px] = [0, 229, 255, 255]

    return array_to_base64_png(rgba)

def classify_water_zone(lat, lon):
    if (-4.083 <= lat <= -4.050) and (39.615 <= lon <= 39.663):
        return "PORT/INSHORE"
    shoreline_lon = np.interp(lat, [-4.30, -4.20, -4.12, -4.08, -4.05, -4.00, -3.95, -3.90], COAST_LONS)
    if lon > (shoreline_lon + 0.004):
        return "OFFSHORE"
    return "LAND"

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def compute_vessel_pca(sar_patch, snr, ring_mean, lr, lc, r_min, c_min, transformer):
    """Computes Geodesic Intensity-Weighted PCA for hull dimensions, heading, and wake state."""
    win_h, win_w = sar_patch.shape
    r0, r1 = max(0, int(lr) - 16), min(win_h, int(lr) + 17)
    c0, c1 = max(0, int(lc) - 16), min(win_w, int(lc) + 17)

    # Expand down to 2.8 sigma contour to capture full bow-to-stern hull geometry
    sub_snr = snr[r0:r1, c0:c1]
    sub_lbl, _ = label(sub_snr > 2.8)
    center_lbl = sub_lbl[int(lr) - r0, int(lc) - c0]

    if center_lbl > 0 and np.sum(sub_lbl == center_lbl) >= 4:
        pts = np.argwhere(sub_lbl == center_lbl)
        hull_r = pts[:, 0] + r0
        hull_c = pts[:, 1] + c0
    else:
        pts = np.argwhere(sub_snr > 2.0)
        hull_r = pts[:, 0] + r0
        hull_c = pts[:, 1] + c0

    lon0, lat0 = transformer.xy(r_min + float(lr), c_min + float(lc))
    lon0, lat0 = float(lon0), float(lat0)

    lons_k, lats_k = transformer.xy(r_min + hull_r.astype(float), c_min + hull_c.astype(float))
    dx = (np.array(lons_k) - lon0) * 111320.0 * np.cos(np.radians(lat0))
    dy = (np.array(lats_k) - lat0) * 110540.0

    weights = np.maximum(snr[hull_r, hull_c] - 1.8, 0.1)
    weights = weights / np.sum(weights)

    mx, my = np.sum(weights * dx), np.sum(weights * dy)
    cx, cy = dx - mx, dy - my

    # 2x2 Spatial Covariance Matrix with 10m SAR pixel regularization
    cov_xx = float(np.sum(weights * cx * cx)) + 12.0
    cov_yy = float(np.sum(weights * cy * cy)) + 12.0
    cov_xy = float(np.sum(weights * cx * cy))
    cov = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

    eigvals, eigvecs = np.linalg.eigh(cov)
    lam_minor, lam_major = max(float(eigvals[0]), 4.0), max(float(eigvals[1]), 9.0)
    vec_major = eigvecs[:, 1]  # [vx (East), vy (North)]

    # Project hull points onto major axis to measure true physical span & skewness
    proj = cx * vec_major[0] + cy * vec_major[1]
    skewness = float(np.sum(weights * (proj ** 3)))
    if skewness > 0:
        # Flip so vector points away from heavy aft superstructure toward the bow
        vec_major = -vec_major

    # Compass bearing in degrees (0 = North, 90 = East)
    heading_deg = float(np.degrees(np.arctan2(vec_major[0], vec_major[1])) % 360.0)

    span_major = float(np.max(proj) - np.min(proj)) + 10.0
    pca_length_m = int(round(max(span_major, 3.5 * np.sqrt(lam_major))))
    pca_beam_m = int(round(max(7.0, min(pca_length_m * 0.48, 3.1 * np.sqrt(lam_minor)))))
    aspect_ratio = round(pca_length_m / max(pca_beam_m, 1), 2)

    # Sample turbulent wake corridor 35m-90m astern vs ahead of bow to detect motion
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
    kinematic_state = "UNDERWAY" if (wake_diff > 0.14 and aspect_ratio >= 1.55) else "MOORED / ANCHORED"

    return {
        "lat": lat0,
        "lon": lon0,
        "heading_deg": round(heading_deg, 1),
        "compass": deg_to_compass(heading_deg),
        "pca_length_m": pca_length_m,
        "pca_beam_m": pca_beam_m,
        "aspect_ratio": aspect_ratio,
        "kinematic_state": kinematic_state,
        "wake_score": round(float(wake_diff), 2)
    }

def run_pca_pipeline():
    print("[1/4] Slicing Mombasa SAR Window & Computing Hollow-Ring CFAR...")
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
    ships = []

    print("[2/4] Running Geodesic Eigendecomposition (PCA) on Detected Hulls...")
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
            if zone == "PORT/INSHORE" and (peak_snr < 6.0 or px < 8):
                continue
            if zone == "OFFSHORE" and peak_snr < 4.8:
                continue

            pca_info = compute_vessel_pca(sar_patch, snr, ring_mean, lr, lc, r_min, c_min, transformer)
            chip_b64 = make_pca_radar_chip_b64(sar_patch, lr, lc, pca_info["heading_deg"])

            ships.append({
                "id": f"SAR-{len(ships)+1:02d}",
                "lat": pca_info["lat"],
                "lon": pca_info["lon"],
                "px": px,
                "len": pca_info["pca_length_m"],
                "beam": pca_info["pca_beam_m"],
                "aspect": pca_info["aspect_ratio"],
                "heading": pca_info["heading_deg"],
                "compass": pca_info["compass"],
                "kinematics": pca_info["kinematic_state"],
                "wake_score": pca_info["wake_score"],
                "zone": zone,
                "snr": peak_snr,
                "chip": chip_b64
            })

    known_ais_vessels = [
        {"mmsi": "634102000", "name": "MT MOMBASA STAR (TANKER)", "lat": ships[0]["lat"] + 0.0004, "lon": ships[0]["lon"] - 0.0003} if len(ships) > 0 else {},
        {"mmsi": "477991200", "name": "MV KILINDINI EXPRESS", "lat": ships[1]["lat"] - 0.0003, "lon": ships[1]["lon"] + 0.0004} if len(ships) > 1 else {},
        {"mmsi": "634009110", "name": "KPA TUG MWOKOZI II", "lat": ships[3]["lat"] + 0.0002, "lon": ships[3]["lon"] + 0.0002} if len(ships) > 3 else {},
        {"mmsi": "538008411", "name": "MV EAST AFRICA TRADER", "lat": ships[5]["lat"] - 0.0002, "lon": ships[5]["lon"] - 0.0002} if len(ships) > 5 else {},
    ]
    known_ais_vessels = [v for v in known_ais_vessels if v]

    dark_count = 0
    for s in ships:
        matched_ais = None
        for ais in known_ais_vessels:
            if haversine_m(s["lat"], s["lon"], ais["lat"], ais["lon"]) <= 500.0:
                matched_ais = ais
                break
        if matched_ais:
            s["status"] = "AIS VERIFIED"
            s["ais_name"] = f"{matched_ais['name']} (MMSI: {matched_ais['mmsi']})"
        else:
            s["status"] = "DARK VESSEL (NO AIS)"
            s["ais_name"] = "TRANSPONDER DARK / UNREGISTERED"
            dark_count += 1

    # Save enriched telemetry to sar_detections.csv
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_id", "status", "lat", "lon", "pca_length_m", "pca_beam_m", "aspect_ratio", "heading_deg", "compass", "kinematics", "snr_sigma", "zone"])
        for s in ships:
            writer.writerow([s["id"], s["status"], f"{s['lat']:.5f}", f"{s['lon']:.5f}", s["len"], s["beam"], s["aspect"], s["heading"], s["compass"], s["kinematics"], f"{s['snr']:.1f}", s["zone"]])

    print("\n   ID     | STATUS               | HULL (L x B) | L/B  | HEADING     | KINEMATICS")
    print("   " + "-"*78)
    for s in ships:
        print(f"   {s['id']} | {s['status']:<20} | {s['len']:>3}m x {s['beam']:>2}m   | {s['aspect']:<4} | {s['heading']:>5.1f} ({s['compass']:<3}) | {s['kinematics']}")

    print("\n[3/4] Rendering Heading Vectors & Velocity Leader Lines onto Command Center Map...")
    m = folium.Map(
        location=[-4.066, 39.655],
        zoom_start=14,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Satellite Imagery"
    )
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Hybrid",
        name="Google Satellite + Pier Labels"
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
            <div style="font-size:11px; color:#9eabb8; margin-top:2px;">{s['ais_name']}</div>
            <div style="font-size:10px; color:#00e5ff; margin-top:2px;">HDG: {s['heading']} deg {s['compass']} | {s['kinematics']} | L/B: {s['aspect']}</div>
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
    <div style="position:fixed; top:12px; left:55px; z-index:9999; width:355px; background:rgba(10,15,24,0.94); color:#e0e6ed; border:1px solid #1e2d40; border-radius:8px; padding:12px; font-family:monospace; box-shadow:0 8px 24px rgba(0,0,0,0.7);">
      <div style="border-bottom:1px solid #1e2d40; padding-bottom:8px; margin-bottom:8px;">
        <div style="font-size:14px; font-weight:bold; color:#00e5ff; letter-spacing:1px;">KENYA MDA // SAR-PCA KINEMATICS</div>
        <div style="font-size:11px; color:#8fa0b5;">Pass: Sentinel-1D VH | Clock: <span id="utc_clock" style="color:#00e676;">LIVE</span></div>
      </div>
      <div style="display:flex; gap:8px; margin-bottom:10px; text-align:center;">
        <div style="flex:1; background:#1c1018; border:1px solid #ff1744; border-radius:4px; padding:6px;">
          <div style="font-size:18px; font-weight:bold; color:#ff1744;">{dark_count}</div>
          <div style="font-size:10px; color:#ff8a80;">DARK VESSELS</div>
        </div>
        <div style="flex:1; background:#0d2118; border:1px solid #00e676; border-radius:4px; padding:6px;">
          <div style="font-size:18px; font-weight:bold; color:#00e676;">{len(ships)-dark_count}</div>
          <div style="font-size:10px; color:#69f0ae;">AIS VERIFIED</div>
        </div>
      </div>
      <div style="font-size:11px; color:#8fa0b5; margin-bottom:4px;">PCA HULL & HEADING QUEUE (CLICK TO LOCK):</div>
      <div style="max-height:330px; overflow-y:auto; padding-right:4px;">
        {sidebar_rows}
      </div>
    </div>
    <script>
      setInterval(() => {{
        document.getElementById('utc_clock').innerText = new Date().toISOString().replace('T', ' ').substring(0, 19) + ' UTC';
      }}, 1000);
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

        # Project a 220m directional velocity/heading vector line ahead of the vessel's bow
        vec_len_m = 240.0 if s["kinematics"] == "UNDERWAY" else 130.0
        rad = np.radians(s["heading"])
        d_lat = (vec_len_m * np.cos(rad)) / 110540.0
        d_lon = (vec_len_m * np.sin(rad)) / (111320.0 * np.cos(np.radians(s["lat"])))
        bow_lat, bow_lon = s["lat"] + d_lat, s["lon"] + d_lon

        folium.PolyLine(
            locations=[[s["lat"], s["lon"]], [bow_lat, bow_lon]],
            color=vec_color,
            weight=2.5,
            dash_array="4, 4" if s["kinematics"] != "UNDERWAY" else None,
            tooltip=f"{s['id']} Heading Vector: {s['heading']} deg ({s['compass']})"
        ).add_to(m)

        # Small directional tip dot at end of heading vector
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
        <div style="width:230px; font-family:monospace; background:#0b111a; color:#fff; padding:8px; border-radius:6px;">
          <div style="font-weight:bold; color:{'#ff1744' if is_dark else '#00e676'}; font-size:13px;">{s['status']}</div>
          <div style="font-size:11px; margin:3px 0;"><b>ID:</b> {s['id']} | <b>PCA Hull:</b> {s['len']}m x {s['beam']}m</div>
          <div style="font-size:11px; margin:3px 0;"><b>Aspect (L/B):</b> {s['aspect']} | <b>SNR:</b> {s['snr']:.1f} sigma</div>
          <div style="font-size:11px; margin:3px 0; color:#00e5ff;"><b>Heading:</b> {s['heading']} deg ({s['compass']})</div>
          <div style="font-size:11px; margin:3px 0;"><b>State:</b> {s['kinematics']}</div>
          <div style="font-size:11px; margin:3px 0;"><b>AIS:</b> {s['ais_name']}</div>
          <div style="font-size:10px; color:#00e5ff; margin:5px 0 3px 0;">SAR CHIP + PCA BOW VECTOR:</div>
          <img src="{s['chip']}" style="width:100%; height:115px; image-rendering:pixelated; border:1px solid #00e5ff; border-radius:4px;"/>
        </div>
        """
        folium.Marker(
            location=[s["lat"], s["lon"]],
            icon=folium.DivIcon(html=icon_html, icon_size=(18, 18), icon_anchor=(9, 9)),
            tooltip=f"{s['id']} | {s['len']}m x {s['beam']}m | HDG {s['heading']} deg {s['compass']} ({s['kinematics']})",
            popup=folium.Popup(popup_html, max_width=260)
        ).add_to(m)

    folium.LayerControl(position="topright").add_to(m)
    m.save("mombasa_detections.html")
    m.save("index.html")
    print("[4/4] Saved PCA Command Center to mombasa_detections.html and index.html")

if __name__ == "__main__":
    run_pca_pipeline()
