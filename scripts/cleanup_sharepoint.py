"""
Script to cleanup accidentally uploaded files in SharePoint.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.sharepoint_uploader import SharePointUploader

def main():
    parser = argparse.ArgumentParser(description="Cleanup SharePoint folders")
    parser.add_argument("--dry-run", action="store_true", help="Only list files, do not delete")
    parser.add_argument("--yes", action="store_true", help="Confirm deletion without prompt")
    args = parser.parse_args()

    print("=" * 60)
    print("🧹 SharePoint Cleanup Utility")
    print("=" * 60)
    
    uploader = SharePointUploader()
    if not uploader.connect():
        print("❌ Could not connect to SharePoint")
        return

    # Folders to check/clean
    folders_to_clean = [
        "Mantenimiento Predictivo Castilla/Fuentes de Datos/Forecast/Fore_Desfibradora",
        "Mantenimiento Predictivo Castilla/Fuentes de Datos/Forecast/Fore_Picadora",
        "Mantenimiento Predictivo Castilla/Fuentes de Datos/Forecast/Fore_Planta"
    ]
    
    for folder in folders_to_clean:
        print(f"\nScanning {folder}...")
        try:
            files = uploader.graph_client.list_files(folder)
            if not files:
                print("   (Empty)")
                continue
                
            print(f"   Found {len(files)} files:")
            for f in files:
                print(f"    - {f.name}")
            
            if not args.dry_run:
                if not args.yes:
                    confirm = input(f"   ⚠️ Delete all {len(files)} files in {folder}? [y/N]: ")
                    if confirm.lower() != 'y':
                        print("   Skipped.")
                        continue
                
                print(f"   Deleting contents of {folder}...")
                if uploader.delete_folder_contents(folder):
                    print("   ✅ Deleted.")
                else:
                    print("   ❌ Error deleting.")
            else:
                print("   [Dry Run] Would delete these files.")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    main()
