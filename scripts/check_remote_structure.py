import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

from common.sharepoint_uploader import SharePointUploader

def main():
    uploader = SharePointUploader()
    if not uploader.connect():
        print("Failed to connect")
        return

    import requests
    
    # List all drives manually using the token
    print(f"Listing drives for site ID: {uploader.graph_client.site_id}")
    
    headers = {"Authorization": f"Bearer {uploader.graph_client.access_token}"}
    drives_url = f"https://graph.microsoft.com/v1.0/sites/{uploader.graph_client.site_id}/drives"
    
    try:
        r = requests.get(drives_url, headers=headers)
        r.raise_for_status()
        drives = r.json().get("value", [])
        
        for drive in drives:
            d_name = drive.get('name', 'Unknown').encode('ascii', 'ignore').decode('ascii')
            d_id = drive.get('id', 'Unknown')
            d_url = drive.get('webUrl', 'Unknown').encode('ascii', 'ignore').decode('ascii')
            print(f" - Drive: {d_name} (ID: {d_id}) (URL: {d_url})")
            
            # List root children of this drive (Top level only)
            print(f"   Root contents:")
            root_url = f"https://graph.microsoft.com/v1.0/drives/{drive['id']}/root/children"
            try:
                rr = requests.get(root_url, headers=headers)
                children = rr.json().get("value", [])
                if not children:
                    print("     (Empty)")
                for child in children:
                    c_name = child.get('name').encode('ascii', 'ignore').decode('ascii')
                    c_type = 'Folder' if 'folder' in child else 'File'
                    print(f"     - {c_name} ({c_type})")
            except Exception as e:
                print(f"     [Error listing children: {e}]")
                
    except Exception as e:
        print(f"Error getting drives: {e}")

if __name__ == "__main__":
    main()
