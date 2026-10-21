# Western Indian Ocean — Sentinel-1 SAR Dark Vessel & Maritime Intelligence Engine

A low-RAM (<150 MB) C4ISR and commercial maritime intelligence pipeline that ingests **Sentinel-1 C-Band SAR (`VH`)** imagery, runs a **Dual-Zone Hollow-Ring CFAR** ship detector, extracts a **Kaggle-formatted 64x64 SAR Chip Dataset**, and cross-references radar reflections against AIS broadcasts off Mombasa, Kenya.

## Repository Contents
* `download_sar.py` — Streams single-band `VH` GeoTIFF nodes directly from Copernicus CDSE without `.SAFE` zip bottlenecks.
* `detect_ships.py` — Windowed GCP coordinate translation, shoreline land-masking, and Hollow-Ring CFAR detection.
* `build_dataset.py` — Extracts 64x64 `.npy` float16 backscatter tensors, `.png` target chips, and `metadata.csv` into `dataset/`.
* `live_command_center.py` — Generates the interactive SAR-AIS Fusion Command Center (`index.html` / `mombasa_detections.html`).
* `dataset/` — 24 balanced Sentinel-1 SAR ship and ocean clutter samples + compressed Kilindini Harbour ROI (`.npz`).
