# MOMBASA MARITIME DOMAIN AWARENESS — AUTOMATED SAR INTELLIGENCE BRIEF
**Generated:** 2026-10-25 08:37 UTC | **Sensor:** Copernicus Sentinel-1D C-Band SAR (VH) | **AI Pipeline:** Hollow-Ring CFAR + Geodesic PCA + Stage-2 ML Classifier

## 1. Executive Summary
* **Total Radar Contacts Verified:** 12 vessels (7 AIS-Compliant, **5 Dark / Non-Broadcasting Vessels**)
* **Close-Proximity / Potential Ship-to-Ship (STS) Pairs (<380m):** 6 pairs flagged
* **Coverage Zone:** Kilindini Harbour, Kipevu Oil Terminal (KOT2), Mombasa Shipyard, and Western Indian Ocean Approaches

## 2. Target Telemetry & Stage-2 AI Confidence Matrix
| ID | AIS Status | ML Conf | Hull (L x B) | Heading | Kinematics | Nearest Contact | Intelligence Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SAR-01** | AIS VERIFIED | **99.6%** | 68m x 33m | 103.1 deg (ESE) | MOORED / ANCHORED | SAR-02 (1799m) | NOMINAL: COMPLIANT COMMERCIAL VESSEL |
| **SAR-02** | AIS VERIFIED | **99.6%** | 270m x 97m | 196.4 deg (SSW) | UNDERWAY | SAR-03 (1362m) | NOMINAL: COMPLIANT COMMERCIAL VESSEL |
| **SAR-03** | AIS VERIFIED | **91.2%** | 271m x 40m | 132.2 deg (SE) | MOORED / ANCHORED | SAR-04 (150m) | NOMINAL: COMPLIANT COMMERCIAL VESSEL |
| **SAR-04** | AIS VERIFIED | **96.9%** | 315m x 40m | 311.5 deg (NW) | MOORED / ANCHORED | SAR-03 (150m) | NOMINAL: COMPLIANT COMMERCIAL VESSEL |
| **SAR-05** | AIS VERIFIED | **93.0%** | 143m x 36m | 151.7 deg (SSE) | MOORED / ANCHORED | SAR-04 (196m) | NOMINAL: COMPLIANT COMMERCIAL VESSEL |
| **SAR-06** | AIS VERIFIED | **96.1%** | 67m x 32m | 97.5 deg (E) | UNDERWAY | SAR-07 (149m) | NOMINAL: COMPLIANT COMMERCIAL VESSEL |
| **SAR-07** | AIS VERIFIED | **80.8%** | 108m x 26m | 46.3 deg (NE) | MOORED / ANCHORED | SAR-06 (149m) | NOMINAL: COMPLIANT COMMERCIAL VESSEL |
| **SAR-08** | DARK VESSEL (NO AIS) | **99.8%** | 274m x 66m | 253.6 deg (WSW) | UNDERWAY | SAR-09 (215m) | CRITICAL: DARK STS / PROXIMITY (<380m) |
| **SAR-09** | DARK VESSEL (NO AIS) | **94.7%** | 95m x 42m | 56.6 deg (ENE) | UNDERWAY | SAR-08 (215m) | CRITICAL: DARK STS / PROXIMITY (<380m) |
| **SAR-10** | DARK VESSEL (NO AIS) | **99.8%** | 145m x 40m | 17.6 deg (NNE) | MOORED / ANCHORED | SAR-06 (480m) | ELEVATED: UNREGISTERED HARBOUR CONTACT |
| **SAR-11** | DARK VESSEL (NO AIS) | **99.8%** | 130m x 54m | 145.3 deg (SE) | UNDERWAY | SAR-12 (300m) | CRITICAL: DARK STS / PROXIMITY (<380m) |
| **SAR-12** | DARK VESSEL (NO AIS) | **93.5%** | 233m x 54m | 59.7 deg (ENE) | UNDERWAY | SAR-11 (300m) | CRITICAL: DARK STS / PROXIMITY (<380m) |
