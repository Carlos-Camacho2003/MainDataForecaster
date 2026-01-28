"""
SharePoint Graph API Transport Layer

Provides Microsoft Graph API integration for SharePoint operations.
This is the primary transport for upload/download operations.

Based on team's sharepoint_uploader_graph.py but refactored to use
environment-based configuration and improved error handling.
"""

from __future__ import annotations
import os
import requests
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, NamedTuple
from urllib.parse import urlparse, quote

# Load .env if present
try:
    from dotenv import load_dotenv
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

logger = logging.getLogger("SharePointGraph")


class FileInfo(NamedTuple):
    """Information about a file in SharePoint."""
    name: str
    size: int
    id: str
    web_url: str
    last_modified: str


class SharePointGraphClient:
    """
    Microsoft Graph API client for SharePoint operations.
    
    This client uses the Graph API (graph.microsoft.com) which is the
    recommended approach for app-only authentication scenarios.
    
    Usage:
        client = SharePointGraphClient()
        if client.connect():
            client.upload_file("local/file.csv", "Remote/Folder")
            client.download_file("Remote/Folder/file.csv", "local/download.csv")
    """
    
    # Simple upload limit (files larger than this need upload sessions)
    SIMPLE_UPLOAD_MAX_SIZE = 4 * 1024 * 1024  # 4 MB
    
    def __init__(
        self,
        site_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
        base_folder: Optional[str] = None,
    ):
        """
        Initialize the Graph API client.
        
        Args:
            site_url: SharePoint site URL (e.g., https://company.sharepoint.com/sites/mysite)
            client_id: Azure AD application client ID
            client_secret: Azure AD application client secret
            tenant_id: Azure AD tenant ID
            base_folder: Base folder path within Documents library
        """
        def _clean(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            value = value.strip()
            try:
                from urllib.parse import unquote
                value = unquote(value)
            except Exception:
                pass
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1].strip()
            value = value.replace('"', '').replace("'", "")
            return value
        
        self.site_url = _clean(site_url or os.getenv("SHAREPOINT_SITE_URL"))
        self.client_id = _clean(client_id or os.getenv("SHAREPOINT_CLIENT_ID"))
        self.client_secret = _clean(client_secret or os.getenv("SHAREPOINT_CLIENT_SECRET"))
        self.tenant_id = _clean(tenant_id or os.getenv("SHAREPOINT_TENANT_ID"))
        self.base_folder = _clean(base_folder or os.getenv("SHAREPOINT_BASE_FOLDER", "")) or ""
        
        # Parse site URL to extract hostname and path
        self.site_hostname: Optional[str] = None
        self.site_path: Optional[str] = None
        if self.site_url:
            parsed = urlparse(self.site_url)
            self.site_hostname = parsed.netloc
            self.site_path = parsed.path
        
        # Connection state
        self.access_token: Optional[str] = None
        self.site_id: Optional[str] = None
        self.drive_id: Optional[str] = None
        self.last_error: Optional[str] = None
        
        self.debug = str(os.getenv("SHAREPOINT_DEBUG", "")).strip().lower() in {"1", "true", "yes", "on"}
    
    def _set_error(self, message: str) -> None:
        self.last_error = message.strip() if message else None
        if message:
            logger.error(message)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers for Graph API requests."""
        return {"Authorization": f"Bearer {self.access_token}"}
    
    def _apply_base_folder(self, remote_path: str) -> str:
        """Apply base folder prefix to remote path."""
        remote_path = remote_path.strip("/")
        if not self.base_folder:
            return remote_path
        base = self.base_folder.strip("/")
        if not remote_path:
            return base
        return f"{base}/{remote_path}"
    
    def connect(self) -> bool:
        """
        Establish connection to SharePoint via Microsoft Graph API.
        
        1. Acquires OAuth2 token with graph.microsoft.com scope
        2. Resolves site_id from site URL
        3. Gets drive_id for the Documents library
        
        Returns:
            True if connection successful, False otherwise
        """
        self.last_error = None
        
        if not all([self.site_url, self.client_id, self.client_secret, self.tenant_id]):
            self._set_error(
                "Missing SharePoint credentials. Required: "
                "SHAREPOINT_SITE_URL, SHAREPOINT_CLIENT_ID, SHAREPOINT_CLIENT_SECRET, SHAREPOINT_TENANT_ID"
            )
            return False
        
        try:
            # Step 1: Get OAuth2 token with Graph API scope
            token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"
            token_data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "scope": "https://graph.microsoft.com/.default"
            }
            print(f"[OK] Authenticated as {self.client_id[:8]}... (App-only)")
            
            logger.info("Requesting access token from Azure AD...")
            response = requests.post(token_url, data=token_data, timeout=30)
            response.raise_for_status()
            
            self.access_token = response.json()["access_token"]
            logger.info("Access token acquired successfully")
            
            if self.debug:
                logger.debug(f"Token: {self.access_token[:20]}...{self.access_token[-10:]}")
            
            # Step 2: Get site ID
            headers = self._get_headers()
            site_query_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_hostname}:{self.site_path}"
            
            logger.info(f"Resolving site: {self.site_hostname}{self.site_path}")
            r = requests.get(site_query_url, headers=headers, timeout=30)
            r.raise_for_status()
            
            site_data = r.json()
            self.site_id = site_data["id"]
            site_name = site_data.get("name", "Unknown")
            logger.info(f"Site resolved: {site_name} (ID: {self.site_id[:30]}...)")
            
            # Step 3: Get drive ID (Documents library)
            drives_url = f"https://graph.microsoft.com/v1.0/sites/{self.site_id}/drives"
            r = requests.get(drives_url, headers=headers, timeout=30)
            r.raise_for_status()
            
            drives = r.json().get("value", [])
            if not drives:
                self._set_error("No document libraries found in the site")
                return False
            
            # Use the first drive (typically "Documents" / "Documentos Compartidos")
            self.drive_id = drives[0]["id"]
            drive_name = drives[0].get("name", "Documents")
            logger.info(f"Drive: {drive_name} (ID: {self.drive_id[:30]}...)")
            
            print(f"[OK] Connected to SharePoint: {site_name}")
            return True
            
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "Unknown"
            body = e.response.text[:500] if e.response is not None else ""
            self._set_error(f"HTTP {status}: {body}")
            print(f"[ERROR] Connection failed: HTTP {status}")
            return False
        except Exception as e:
            self.last_error = str(e)
            try:
                print(f"[ERROR] Connection failed: {str(e)}")
            except:
                print(f"[ERROR] Connection failed: {str(e).encode('utf-8', errors='ignore')}")
            return False
    
    def create_folder(self, folder_path: str) -> bool:
        """
        Create a folder in SharePoint (creates parent folders as needed).
        
        Args:
            folder_path: Folder path relative to Documents library
            
        Returns:
            True if folder exists or was created successfully
        """
        if not self.access_token or not self.drive_id:
            self._set_error("Not connected. Call connect() first.")
            return False
        
        folder_path = self._apply_base_folder(folder_path)
        headers = self._get_headers()
        
        try:
            # Build folder path incrementally
            parts = [p for p in folder_path.split("/") if p]
            current_path = ""
            
            for part in parts:
                current_path = f"{current_path}/{part}" if current_path else part
                
                # Check if folder exists
                check_url = f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/root:/{quote(current_path, safe='/')}"
                r = requests.get(check_url, headers=headers, timeout=30)
                
                if r.status_code == 404:
                    # Folder doesn't exist, create it
                    parent_path = "/".join(current_path.split("/")[:-1])
                    if parent_path:
                        create_url = f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/root:/{quote(parent_path, safe='/')}:/children"
                    else:
                        create_url = f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/root/children"
                    
                    folder_data = {
                        "name": part,
                        "folder": {},
                        "@microsoft.graph.conflictBehavior": "fail"
                    }
                    
                    r = requests.post(
                        create_url,
                        headers={**headers, "Content-Type": "application/json"},
                        json=folder_data,
                        timeout=30
                    )
                    
                    if r.status_code not in (200, 201):
                        # Check if it's a conflict (folder already exists)
                        if r.status_code == 409:
                            continue
                        self._set_error(f"Failed to create folder {part}: {r.status_code} {r.text[:200]}")
                        return False
                    
                    logger.info(f"Created folder: {current_path}")
            
            return True
            
        except Exception as e:
            self._set_error(f"Error creating folder {folder_path}: {str(e)}")
            return False
    
    def upload_file(
        self,
        local_path: str | Path,
        remote_folder: str,
        remote_filename: Optional[str] = None,
        overwrite: bool = True
    ) -> bool:
        """
        Upload a file to SharePoint using Graph API.
        
        Args:
            local_path: Path to local file
            remote_folder: Target folder in SharePoint (relative to Documents/base_folder)
            remote_filename: Target filename (default: same as local)
            overwrite: Whether to overwrite existing files
            
        Returns:
            True if upload successful
        """
        if not self.access_token or not self.drive_id:
            self._set_error("Not connected. Call connect() first.")
            return False
        
        local_path = Path(local_path)
        if not local_path.exists():
            self._set_error(f"File not found: {local_path}")
            return False
        
        remote_filename = remote_filename or local_path.name
        remote_folder = self._apply_base_folder(remote_folder)
        
        # Ensure target folder exists
        if remote_folder:
            self.create_folder(remote_folder)
        
        try:
            file_size = local_path.stat().st_size
            headers = self._get_headers()
            
            # Construct upload URL
            if remote_folder:
                upload_url = f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/root:/{quote(remote_folder, safe='/')}/{quote(remote_filename)}:/content"
            else:
                upload_url = f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/root:/{quote(remote_filename)}:/content"
            
            if file_size <= self.SIMPLE_UPLOAD_MAX_SIZE:
                # Simple upload for small files
                with open(local_path, "rb") as f:
                    r = requests.put(upload_url, headers=headers, data=f, timeout=60)
                
                if r.status_code in (200, 201):
                    logger.info(f"Uploaded: {local_path.name} -> {remote_folder}/{remote_filename}")
                    print(f"[OK] Uploaded: {local_path.name}")
                    return True
                else:
                    self._set_error(f"Upload failed: {r.status_code} {r.text[:200]}")
                    print(f"[ERROR] Upload failed: {local_path.name}")
                    return False
            else:
                # Large file upload using upload session
                return self._upload_large_file(local_path, remote_folder, remote_filename)
                
        except Exception as e:
            self._set_error(f"Upload error: {str(e)}")
            print(f"[ERROR] Upload failed: {local_path.name} - {str(e)}")
            return False
    
    def _upload_large_file(
        self,
        local_path: Path,
        remote_folder: str,
        remote_filename: str
    ) -> bool:
        """Upload large files using resumable upload session."""
        headers = self._get_headers()
        
        try:
            # Create upload session
            if remote_folder:
                session_url = f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/root:/{quote(remote_folder, safe='/')}/{quote(remote_filename)}:/createUploadSession"
            else:
                session_url = f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/root:/{quote(remote_filename)}:/createUploadSession"
            
            session_data = {
                "item": {
                    "@microsoft.graph.conflictBehavior": "replace",
                    "name": remote_filename
                }
            }
            
            r = requests.post(
                session_url,
                headers={**headers, "Content-Type": "application/json"},
                json=session_data,
                timeout=30
            )
            r.raise_for_status()
            
            upload_url = r.json()["uploadUrl"]
            file_size = local_path.stat().st_size
            chunk_size = 10 * 1024 * 1024  # 10 MB chunks
            
            with open(local_path, "rb") as f:
                chunk_start = 0
                while chunk_start < file_size:
                    chunk_end = min(chunk_start + chunk_size, file_size) - 1
                    chunk_data = f.read(chunk_size)
                    
                    chunk_headers = {
                        "Content-Length": str(len(chunk_data)),
                        "Content-Range": f"bytes {chunk_start}-{chunk_end}/{file_size}"
                    }
                    
                    r = requests.put(upload_url, headers=chunk_headers, data=chunk_data, timeout=120)
                    
                    if r.status_code not in (200, 201, 202):
                        self._set_error(f"Chunk upload failed: {r.status_code}")
                        return False
                    
                    chunk_start = chunk_end + 1
                    progress = (chunk_start / file_size) * 100
                    logger.info(f"Upload progress: {progress:.1f}%")
            
            logger.info(f"Uploaded (large): {local_path.name}")
            print(f"[OK] Uploaded: {local_path.name}")
            return True
            
        except Exception as e:
            self._set_error(f"Large file upload error: {str(e)}")
            return False
    
    def download_file(self, remote_path: str, local_path: str | Path) -> bool:
        """
        Download a file from SharePoint.
        
        Args:
            remote_path: Path to file in SharePoint (relative to Documents/base_folder)
            local_path: Local path to save the file
            
        Returns:
            True if download successful
        """
        if not self.access_token or not self.drive_id:
            self._set_error("Not connected. Call connect() first.")
            return False
        
        remote_path = self._apply_base_folder(remote_path)
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            headers = self._get_headers()
            download_url = f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/root:/{quote(remote_path, safe='/')}:/content"
            
            r = requests.get(download_url, headers=headers, stream=True, timeout=120)
            r.raise_for_status()
            
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded: {remote_path} -> {local_path}")
            print(f"[OK] Downloaded: {local_path.name}")
            return True
            
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "Unknown"
            self._set_error(f"Download failed: HTTP {status}")
            print(f"[ERROR] Download failed: {remote_path}")
            return False
        except Exception as e:
            self._set_error(f"Download error: {str(e)}")
            print(f"[ERROR] Download failed: {str(e)}")
            return False
    
    def list_files(self, remote_folder: str) -> List[FileInfo]:
        """
        List files in a SharePoint folder.
        
        Args:
            remote_folder: Folder path relative to Documents/base_folder
            
        Returns:
            List of FileInfo objects
        """
        if not self.access_token or not self.drive_id:
            self._set_error("Not connected. Call connect() first.")
            return []
        
        remote_folder = self._apply_base_folder(remote_folder)
        headers = self._get_headers()
        
        try:
            if remote_folder:
                list_url = f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/root:/{quote(remote_folder, safe='/')}:/children"
            else:
                list_url = f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/root/children"
            
            r = requests.get(list_url, headers=headers, timeout=30)
            r.raise_for_status()
            
            items = r.json().get("value", [])
            files = []
            
            for item in items:
                # Skip folders
                if "folder" in item:
                    continue
                
                files.append(FileInfo(
                    name=item["name"],
                    size=item.get("size", 0),
                    id=item["id"],
                    web_url=item.get("webUrl", ""),
                    last_modified=item.get("lastModifiedDateTime", "")
                ))
            
            return files
            
        except Exception as e:
            self._set_error(f"Error listing files: {str(e)}")
            return []
    
    def delete_file(self, remote_path: str) -> bool:
        """
        Delete a file from SharePoint.
        
        Args:
            remote_path: Path to file in SharePoint
            
        Returns:
            True if deletion successful
        """
        if not self.access_token or not self.drive_id:
            self._set_error("Not connected. Call connect() first.")
            return False
        
        remote_path = self._apply_base_folder(remote_path)
        headers = self._get_headers()
        
        try:
            delete_url = f"https://graph.microsoft.com/v1.0/drives/{self.drive_id}/root:/{quote(remote_path, safe='/')}"
            r = requests.delete(delete_url, headers=headers, timeout=30)
            
            if r.status_code in (200, 204):
                logger.info(f"Deleted: {remote_path}")
                return True
            elif r.status_code == 404:
                logger.warning(f"File not found: {remote_path}")
                return True  # Consider already deleted as success
            else:
                self._set_error(f"Delete failed: {r.status_code}")
                return False
                
        except Exception as e:
            self._set_error(f"Delete error: {str(e)}")
            return False
    
    def upload_folder(
        self,
        local_folder: str | Path,
        remote_folder: str,
        file_extensions: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Upload all files from a local folder to SharePoint.
        
        Args:
            local_folder: Local folder path
            remote_folder: Target folder in SharePoint
            file_extensions: List of extensions to include (e.g., [".csv", ".xlsx"])
            
        Returns:
            Dictionary mapping filenames to upload success status
        """
        results = {}
        local_folder = Path(local_folder)
        
        if not local_folder.exists() or not local_folder.is_dir():
            self._set_error(f"Local folder not found: {local_folder}")
            return results
        
        files = []
        for file_path in local_folder.iterdir():
            if file_path.is_file():
                if file_extensions is None or file_path.suffix.lower() in file_extensions:
                    files.append(file_path)
        
        logger.info(f"Uploading {len(files)} file(s) from {local_folder}")
        
        for file_path in files:
            success = self.upload_file(file_path, remote_folder)
            results[file_path.name] = success
        
        return results
    
    def download_folder(self, remote_folder: str, local_folder: str | Path) -> int:
        """
        Download all files from a SharePoint folder.
        
        Args:
            remote_folder: Folder path in SharePoint
            local_folder: Local folder to save files
            
        Returns:
            Number of files downloaded
        """
        local_folder = Path(local_folder)
        local_folder.mkdir(parents=True, exist_ok=True)
        
        files = self.list_files(remote_folder)
        if not files:
            logger.warning(f"No files found in {remote_folder}")
            return 0
        
        count = 0
        for file_info in files:
            remote_path = f"{remote_folder}/{file_info.name}" if remote_folder else file_info.name
            # Don't apply base folder again - list_files already returned paths relative to base
            local_path = local_folder / file_info.name
            
            # Temporarily remove base folder prefix to avoid double-application
            original_base = self.base_folder
            self.base_folder = ""
            
            if self.download_file(self._apply_base_folder(remote_path.lstrip("/")), local_path):
                count += 1
            
            self.base_folder = original_base
        
        return count
