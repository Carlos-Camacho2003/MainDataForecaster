import sys
from pathlib import Path
import requests
import os
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from common.sharepoint_uploader import SharePointUploader
from urllib.parse import quote

def main():
    uploader = SharePointUploader()
    if not uploader.connect():
        print("Failed to connect")
        return

    # Target path
    base_path = "Mantenimiento Predictivo Castilla/Fuentes de Datos/Forecast/Fore_Planta"
    print(f"Checking folders in: {base_path}")

    # Use graph client internals to find the folder ID first, or keys
    # We already know verify connect() gets the Drive ID.
    
    headers = {"Authorization": f"Bearer {uploader.graph_client.access_token}"}
    drive_id = uploader.graph_client.drive_id
    
    # URL to list children of the specific path
    # Endpoint: /drives/{drive-id}/root:/{path}:/children
    encoded_path = quote(base_path)
    url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/root:/{encoded_path}:/children"
    
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 404:
            print(f"[ERROR] Path not found: {base_path}")
            return
            
        r.raise_for_status()
        items = r.json().get("value", [])
        
        print("\nFound items:")
        for item in items:
            name = item.get("name", "Unknown").encode("ascii", "ignore").decode("ascii")
            is_folder = "folder" in item
            print(f" - {name} [{'Folder' if is_folder else 'File'}]")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
