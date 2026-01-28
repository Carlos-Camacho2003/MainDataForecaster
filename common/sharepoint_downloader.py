"""
SharePoint Downloader Utility

Handles downloading raw data and performance indicators from SharePoint.
Uses Microsoft Graph API as the primary transport (recommended for app-only auth).
"""

from __future__ import annotations
import os
from pathlib import Path
import shutil
from typing import Optional, Dict
import logging

# Load .env if present (so running the GUI picks up credentials)
try:
    from dotenv import load_dotenv
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

# Import the Graph API transport
from common.sharepoint_graph import SharePointGraphClient

logger = logging.getLogger("SharePointDownloader")


class SharePointDownloader:
    """
    SharePoint downloader for raw data and performance metrics.
    
    Uses Microsoft Graph API for reliable app-only authentication.
    """
    
    def __init__(
        self,
        site_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
        document_library: str = "Documentos Compartidos"
    ):
        """
        Initialize SharePoint downloader.
        
        Args:
            site_url: SharePoint site URL
            client_id: Azure AD app client ID
            client_secret: Azure AD app client secret
            tenant_id: Azure AD tenant ID
            document_library: SharePoint document library name (kept for compatibility)
        """
        # Initialize Graph API client
        self.graph_client = SharePointGraphClient(
            site_url=site_url,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            base_folder=os.getenv("SHAREPOINT_BASE_FOLDER", ""),
        )
        
        # Keep references for compatibility
        self.site_url = self.graph_client.site_url
        self.document_library = document_library
        self.base_folder = self.graph_client.base_folder
        self.last_error_details: Optional[str] = None
        self.debug = self.graph_client.debug

    def _set_last_error(self, message: str) -> None:
        self.last_error_details = message.strip() if message else None
        
    def connect(self) -> bool:
        """
        Establish connection to SharePoint using Graph API.
        
        Returns:
            True if connection successful, False otherwise
        """
        success = self.graph_client.connect()
        if not success:
            self._set_last_error(self.graph_client.last_error or "Connection failed")
            print(f"❌ Cannot connect: {self.last_error_details}")
        return success

    def download_folder(self, remote_folder: str, local_folder: str | Path) -> int:
        """
        Download all files from a remote SharePoint folder to a local folder.
        
        Args:
            remote_folder: Path relative to document library (e.g. "Fuentes de Datos/Performance")
            local_folder: Local path to save files
            
        Returns:
            Number of files downloaded
        """
        if not self.graph_client.access_token:
            if not self.connect():
                return 0
                
        local_folder = Path(local_folder)
        local_folder.mkdir(parents=True, exist_ok=True)
        
        try:
            # List files in the remote folder
            files = self.graph_client.list_files(remote_folder)
            
            if not files:
                print(f"⚠ No files found in {remote_folder}")
                return 0
            
            count = 0
            for file_info in files:
                # Construct full remote path
                remote_path = f"{remote_folder}/{file_info.name}"
                local_path = local_folder / file_info.name
                
                # Download using the Graph client
                # Need to handle base_folder correctly - list_files doesn't include it in results
                if self.graph_client.download_file(remote_path, local_path):
                    count += 1
            
            return count
            
        except Exception as e:
            self._set_last_error(f"Error downloading from {remote_folder}: {str(e)}")
            print(f"❌ {self.last_error_details}")
            return 0

    def download_file(self, remote_path: str, local_path: str | Path) -> bool:
        """
        Download a single file from SharePoint.
        
        Args:
            remote_path: Path to file relative to document library
            local_path: Local path to save the file
            
        Returns:
            True if download successful
        """
        if not self.graph_client.access_token:
            if not self.connect():
                return False
        
        return self.graph_client.download_file(remote_path, local_path)

    def list_files(self, remote_folder: str) -> list:
        """
        List files in a remote SharePoint folder.
        
        Args:
            remote_folder: Path relative to document library
            
        Returns:
            List of FileInfo objects
        """
        if not self.graph_client.access_token:
            if not self.connect():
                return []
        
        return self.graph_client.list_files(remote_folder)

    @staticmethod
    def _clean_local_folder(local_folder: Path) -> None:
        if local_folder.exists():
            shutil.rmtree(local_folder)
        local_folder.mkdir(parents=True, exist_ok=True)

    def sync_data(self, project_root: str | Path) -> Dict[str, int]:
        """
        Synchronize all required data folders from SharePoint.
        
                Structure:
                Fuentes de Datos/
                    Performance/
                        Máquinas/
                            Perf_Desfibradora -> data/epi/
                            Perf_Picadora -> data/epi/
                        Planta -> data/epi/
                    Dimensiones -> data/dimensions/
                    Mantenimiento Predictivo Castilla/Fuentes de Datos/Desfibradora -> data/raw/DESF/
                    Mantenimiento Predictivo Castilla/Fuentes de Datos/Picadora -> data/raw/PICADORA/
        """
        project_root = Path(project_root)
        results = {}
        
        # Define mappings: Remote Path -> Local Path (relative to project root)
        mappings = [
            ("Mantenimiento Predictivo Castilla/Fuentes de Datos/Perfomance/Máquinas/Perf_Desfibradora", "data/epi/DESF"),
            ("Mantenimiento Predictivo Castilla/Fuentes de Datos/Perfomance/Máquinas/Perf_Picadora", "data/epi/PICADORA"),
            ("Mantenimiento Predictivo Castilla/Fuentes de Datos/Perfomance/Planta", "data/epi/PLANT"),
            ("Mantenimiento Predictivo Castilla/Fuentes de Datos/Dimensiones", "data/dimensions"),
            ("Mantenimiento Predictivo Castilla/Fuentes de Datos/Desfibradora", "data/raw/DESF"),
            ("Mantenimiento Predictivo Castilla/Fuentes de Datos/Picadora", "data/raw/PICADORA")
        ]
        
        for remote, local_rel in mappings:
            local_path = project_root / local_rel

            # Clean local raw folders before downloading consolidated raw data
            if local_path.parts[-2:] in [("raw", "DESF"), ("raw", "PICADORA")]:
                print(f"🧹 Limpiando carpeta local: {local_path}")
                self._clean_local_folder(local_path)

            print(f"⬇ Syncing {remote} -> {local_path}")
            count = self.download_folder(remote, local_path)
            results[remote] = count
            
        return results


# Standalone usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download files from SharePoint")
    parser.add_argument("--folder", help="Remote folder to download")
    parser.add_argument("--output", help="Local output folder")
    parser.add_argument("--sync", action="store_true", help="Sync all data folders")
    
    args = parser.parse_args()
    
    downloader = SharePointDownloader()
    
    if args.sync:
        project_root = Path(__file__).resolve().parent.parent
        results = downloader.sync_data(project_root)
        print("\nSync results:")
        for folder, count in results.items():
            print(f"  {folder}: {count} files")
    elif args.folder and args.output:
        if downloader.connect():
            count = downloader.download_folder(args.folder, args.output)
            print(f"Downloaded {count} files")
    else:
        print("Use --sync or provide --folder and --output")
