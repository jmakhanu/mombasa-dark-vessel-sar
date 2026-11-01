import os
import datetime
import requests
from getpass import getpass
from tqdm import tqdm

AOI_POLYGON = "POLYGON((39.6 -4.2, 39.9 -4.2, 39.9 -3.9, 39.6 -3.9, 39.6 -4.2))"
# Automatically search the trailing 14-day window up to today
now_utc = datetime.datetime.now(datetime.timezone.utc)
START_DATE = (now_utc - datetime.timedelta(days=14)).strftime("%Y-%m-%dT00:00:00.000Z")
END_DATE = now_utc.strftime("%Y-%m-%dT23:59:59.000Z")

def get_access_token(user, password):
    print("\n[1/4] Authenticating with Copernicus CDSE...")
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "username": user.strip(),
        "password": password.strip(),
        "grant_type": "password",
    }
    response = requests.post(url, data=data, timeout=30)
    if response.status_code != 200:
        raise ValueError(f"Login failed ({response.status_code}). Check your CDSE credentials.")
    return response.json()["access_token"]

def search_sentinel1_grd():
    print(f"[2/4] Searching Sentinel-1 GRD over Mombasa ({START_DATE[:10]} to {END_DATE[:10]})...")
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    query = (
        f"?$filter=Collection/Name eq 'SENTINEL-1' and contains(Name,'GRD') "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{AOI_POLYGON}') "
        f"and ContentDate/Start gt {START_DATE} and ContentDate/Start lt {END_DATE}"
        f"&$orderby=ContentDate/Start desc&$top=1"
    )
    response = requests.get(url + query, timeout=30)
    response.raise_for_status()
    results = response.json().get("value", [])
    if not results:
        raise RuntimeError("No SAR images found in the last 14 days.")
    latest = results[0]
    print(f"      Found Latest Pass: {latest['Name']}")
    with open("latest_pass_meta.txt", "w") as f:
        f.write(latest["Name"])
    return latest["Id"], latest["Name"]

def find_vh_tiff_node(image_id, image_name, token):
    print("[3/4] Locating raw VH radar band inside .SAFE node...")
    headers = {"Authorization": f"Bearer {token}"}
    nodes_url = (
        f"https://download.dataspace.copernicus.eu/odata/v1/Products({image_id})"
        f"/Nodes({image_name})/Nodes(measurement)/Nodes"
    )
    response = requests.get(nodes_url, headers=headers, timeout=30)
    response.raise_for_status()
    files = response.json().get("result", [])
    for f in files:
        if "-vh-" in f["Id"].lower() and f["Id"].endswith(".tiff"):
            return f["Id"]
    return files[0]["Id"]

def download_single_tiff(image_id, image_name, tiff_id, token):
    print(f"[4/4] Streaming single radar band: {tiff_id}")
    url = (
        f"https://download.dataspace.copernicus.eu/odata/v1/Products({image_id})"
        f"/Nodes({image_name})/Nodes(measurement)/Nodes({tiff_id})/$value"
    )
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, allow_redirects=False, stream=True, timeout=60)
    while response.status_code in (301, 302, 303, 307, 308):
        url = response.headers["Location"]
        response.close()
        response = requests.get(url, headers=headers, allow_redirects=False, stream=True, timeout=60)

    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    output_file = "mombasa_sar_vh.tiff"

    with open(output_file, "wb") as f, tqdm(
        desc="Downloading", total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                size = f.write(chunk)
                bar.update(size)

    print(f"\nSuccess: Saved directly as {output_file}")

if __name__ == "__main__":
    print("=== DARK VESSEL SAR INGESTION (LOCAL + CLOUD AUTONOMOUS) ===")
    user_email = os.environ.get("CDSE_EMAIL") or input("Enter your Copernicus Email: ")
    user_pass = os.environ.get("CDSE_PASSWORD") or getpass("Enter your Copernicus Password: ")

    token = get_access_token(user_email, user_pass)
    img_id, img_name = search_sentinel1_grd()
    tiff_name = find_vh_tiff_node(img_id, img_name, token)
    download_single_tiff(img_id, img_name, tiff_name, token)
