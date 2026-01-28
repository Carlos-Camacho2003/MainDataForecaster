"""
SharePoint Uploader Utility

Handles uploading forecast outputs and metadata to SharePoint for Power BI integration.
Uses Microsoft Graph API as the primary transport (recommended for app-only auth).

Usage:
    from common.sharepoint_uploader import SharePointUploader
    
    uploader = SharePointUploader()
    uploader.upload_forecast("forecasts/corriente_motor_a_forecast.csv", "SARIMAX")
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import hashlib
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

logger = logging.getLogger("SharePointUploader")


class SharePointUploader:
    """
    SharePoint uploader for forecast outputs.
    
    Uses Microsoft Graph API for reliable app-only authentication.
    
    Supports:
    - CSV forecast files
    - JSON metadata files
    - PNG/HTML visualization files
    - Model artifacts (.pkl, .pth)
    """
    
    def __init__(
        self,
        site_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
        document_library: str = "Documentos Compartidos",
        base_folder: str = "",
    ):
        """
        Initialize SharePoint uploader.
        
        Args:
            site_url: SharePoint site URL (e.g., https://company.sharepoint.com/sites/yoursite)
            client_id: Azure AD app client ID
            client_secret: Azure AD app client secret
            tenant_id: Azure AD tenant ID or domain (e.g., contoso.onmicrosoft.com)
            document_library: SharePoint document library name (not used with Graph API, kept for compatibility)
            base_folder: Base folder path within library
        """
        # Initialize Graph API client
        self.graph_client = SharePointGraphClient(
            site_url=site_url,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            base_folder=base_folder or os.getenv("SHAREPOINT_BASE_FOLDER", ""),
        )
        
        # Keep reference for compatibility
        self.site_url = self.graph_client.site_url
        self.base_folder = self.graph_client.base_folder
        self.document_library = document_library
        
        self.upload_log: List[Dict] = []
        self.last_error_details: Optional[str] = None
        
        # Validate configuration
        if not all([self.graph_client.site_url, self.graph_client.client_id, 
                    self.graph_client.client_secret, self.graph_client.tenant_id]):
            print("⚠ SharePoint credentials not configured. Set environment variables:")
            print("   SHAREPOINT_SITE_URL")
            print("   SHAREPOINT_CLIENT_ID")
            print("   SHAREPOINT_CLIENT_SECRET")
            print("   SHAREPOINT_TENANT_ID")
            print("\n   Or create a .env file in the project root.")

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
        return success
    
    def ensure_folder(self, folder_path: str) -> bool:
        """
        Ensure folder exists in SharePoint, create if needed.
        
        Args:
            folder_path: Relative folder path (e.g., "IDC_RIOP_Forecasts/SARIMAX/2025-11")
        
        Returns:
            True if folder exists or was created successfully
        """
        if not self.graph_client.access_token:
            if not self.connect():
                return False
        
        success = self.graph_client.create_folder(folder_path)
        if not success:
            self._set_last_error(self.graph_client.last_error or "Folder creation failed")
        return success
    
    def delete_folder_contents(self, remote_folder: str) -> bool:
        """
        Delete all files in a SharePoint folder.
        
        Args:
            remote_folder: Folder path relative to document library
            
        Returns:
            True if successful
        """
        if not self.graph_client.access_token:
            if not self.connect():
                return False
        
        try:
            files = self.graph_client.list_files(remote_folder)
            if not files:
                print(f"ℹ Folder already empty or not found: {remote_folder}")
                return True
            
            print(f"🗑 Deleting {len(files)} files from {remote_folder}...")
            
            for file_info in files:
                remote_path = f"{remote_folder}/{file_info.name}"
                self.graph_client.delete_file(remote_path)
            
            print(f"✅ Cleared folder: {remote_folder}")
            return True
            
        except Exception as e:
            self._set_last_error(f"Failed to clear folder: {str(e)}")
            print(f"❌ Failed to clear folder {remote_folder}: {str(e)}")
            return False

    def upload_file(
        self,
        local_path: str | Path,
        remote_folder: str,
        remote_filename: Optional[str] = None,
        overwrite: bool = True
    ) -> bool:
        """
        Upload a file to SharePoint.
        
        Args:
            local_path: Path to local file
            remote_folder: Target folder in SharePoint (relative to document library)
            remote_filename: Target filename (default: same as local)
            overwrite: Whether to overwrite existing files
        
        Returns:
            True if upload successful
        """
        if not self.graph_client.access_token:
            if not self.connect():
                return False
        
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"❌ File not found: {local_path}")
            self._set_last_error(f"File not found: {local_path}")
            return False
        
        remote_filename = remote_filename or local_path.name
        
        success = self.graph_client.upload_file(
            local_path=local_path,
            remote_folder=remote_folder,
            remote_filename=remote_filename,
            overwrite=overwrite
        )
        
        if success:
            # Log upload
            upload_info = {
                "timestamp": datetime.now().isoformat(),
                "local_path": str(local_path),
                "remote_folder": remote_folder,
                "remote_filename": remote_filename,
                "file_size": local_path.stat().st_size,
                "checksum": self._calculate_checksum(local_path)
            }
            self.upload_log.append(upload_info)
        else:
            self._set_last_error(self.graph_client.last_error or "Upload failed")
        
        return success
    
    def upload_forecast(
        self,
        forecast_csv: str | Path,
        model_type: str = "SARIMAX",
        include_metadata: bool = True
    ) -> bool:
        """
        Upload forecast CSV and associated metadata.
        
        Args:
            forecast_csv: Path to forecast CSV file
            model_type: Model type (SARIMAX or NBEATS)
            include_metadata: Whether to upload summary JSON
        
        Returns:
            True if all uploads successful
        """
        forecast_csv = Path(forecast_csv)
        if not forecast_csv.exists():
            print(f"❌ Forecast file not found: {forecast_csv}")
            return False
        
        # Extract variable name and timestamp
        variable = forecast_csv.stem.replace("_forecast", "")
        timestamp = datetime.now().strftime("%Y-%m")
        
        # Organize by model and month
        if self.base_folder:
            remote_folder = f"{model_type}/{timestamp}"
        else:
            remote_folder = f"{model_type}/{timestamp}"
        
        # Upload forecast CSV
        success = self.upload_file(forecast_csv, remote_folder)
        
        # Upload metadata if available
        if include_metadata and success:
            summary_json = forecast_csv.parent / f"{variable}_summary.json"
            if summary_json.exists():
                self.upload_file(summary_json, remote_folder)
        
        return success

    def upload_anomaly_results(
        self,
        csv_dir: str | Path,
        machine: str
    ) -> bool:
        """
        Upload anomaly detection results (summary and details).
        
        Args:
            csv_dir: Directory containing anomaly CSVs
            machine: Machine name (DESF or PICADORA)
        """
        csv_dir = Path(csv_dir)
        remote_folder = "Fuentes de Datos/Anomalias"

        # Collect CSVs using several naming conventions
        files_to_upload: list[Path] = []
        preferred_csv_names = [
            f"ANOMALY_SUMMARY_{machine}.csv",
            f"ANOMALY_DETAILS_{machine}.csv",
            f"ANOMALIES_{machine}.csv",
        ]
        for name in preferred_csv_names:
            p = csv_dir / name
            if p.exists():
                files_to_upload.append(p)

        if not files_to_upload:
            print(f"ℹ No anomaly CSV files found in {csv_dir}")
            return False

        success = True
        for f in files_to_upload:
            print(f"Uploading anomaly file: {f.name}")
            if not self.upload_file(f, remote_folder):
                success = False

        return success
    
    def upload_batch(
        self,
        files: List[str | Path],
        remote_folder: str
    ) -> Dict[str, bool]:
        """
        Upload multiple files in batch.
        
        Args:
            files: List of local file paths
            remote_folder: Target folder in SharePoint
        
        Returns:
            Dictionary mapping filenames to success status
        """
        results = {}
        
        for file_path in files:
            file_path = Path(file_path)
            success = self.upload_file(file_path, remote_folder)
            results[file_path.name] = success
        
        return results
    
    def save_upload_log(self, output_path: str | Path = "upload_log.json"):
        """
        Save upload log to JSON file.
        
        Args:
            output_path: Path to save log file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.upload_log, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Upload log saved to: {output_path}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum for file integrity verification."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def create_config_template(output_path: str = ".env.template"):
        """
        Create template .env file with SharePoint configuration.
        
        Args:
            output_path: Path to save template file
        """
        template = """# SharePoint Configuration for IDC_RIOP
# Copy this file to .env and fill in your credentials

# SharePoint Site URL (e.g., https://company.sharepoint.com/sites/yoursite)
SHAREPOINT_SITE_URL=

# App-only credentials (Client ID / Client Secret)
# Requires Microsoft Graph API permissions: Sites.ReadWrite.All
SHAREPOINT_CLIENT_ID=
SHAREPOINT_CLIENT_SECRET=

# Azure AD Tenant ID (GUID or domain like contoso.onmicrosoft.com)
SHAREPOINT_TENANT_ID=

# Optional: Base folder within Documents library
# If your SharePoint has a top-level folder that contains "Fuentes de Datos", etc.
SHAREPOINT_BASE_FOLDER=
"""
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(template)
        
        print(f"✅ Created configuration template: {output_path}")
        print("   Copy to .env and add your SharePoint credentials")


# Standalone usage example
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload files to SharePoint")
    parser.add_argument("--file", required=True, help="File to upload")
    parser.add_argument("--folder", required=True, help="Target SharePoint folder")
    parser.add_argument("--model", choices=["SARIMAX", "NBEATS"], default="SARIMAX")
    parser.add_argument("--create-config", action="store_true", help="Create .env template")
    
    args = parser.parse_args()
    
    if args.create_config:
        SharePointUploader.create_config_template()
    else:
        uploader = SharePointUploader()
        
        if uploader.connect():
            uploader.upload_forecast(args.file, model_type=args.model)
            uploader.save_upload_log()
