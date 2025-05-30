import requests
import json
import os
import sys
import subprocess
import tempfile
from packaging import version
from PyQt5.QtWidgets import QMessageBox, QProgressDialog
from PyQt5.QtCore import QThread, pyqtSignal

class UpdateChecker(QThread):
    update_available = pyqtSignal(dict)
    no_update = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, current_version="1.0.0"):
        super().__init__()
        self.current_version = current_version
        GITHUB_USERNAME = os.environ.get('GITHUB_USERNAME')
        self.github_repo = "markm39/document-search-tool"
    
    def run(self):
        try:
            # Check GitHub releases API
            url = f"https://api.github.com/repos/{self.github_repo}/releases/latest"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                release_data = response.json()
                latest_version = release_data['tag_name'].lstrip('v')
                
                if version.parse(latest_version) > version.parse(self.current_version):
                    self.update_available.emit(release_data)
                else:
                    self.no_update.emit()
            else:
                self.error_occurred.emit("Could not check for updates")
                
        except Exception as e:
            self.error_occurred.emit(str(e))

class AutoUpdater:
    def __init__(self, parent_window, current_version="1.0.0"):
        self.parent = parent_window
        self.current_version = current_version
        self.github_repo = "markm39/document-search-tool"
    
    def check_for_updates(self, silent=False):
        """Check for updates. If silent=True, only notify if update available"""
        self.silent = silent
        self.update_checker = UpdateChecker(self.current_version)
        self.update_checker.update_available.connect(self.prompt_update)
        self.update_checker.no_update.connect(self.no_update_available)
        self.update_checker.error_occurred.connect(self.update_error)
        self.update_checker.start()
    
    def prompt_update(self, release_data):
        """Show update dialog"""
        reply = QMessageBox.question(
            self.parent,
            "Update Available",
            f"A new version is available!\n\n"
            f"Current: v{self.current_version}\n"
            f"Latest: {release_data['tag_name']}\n\n"
            f"Changes:\n{release_data['body'][:200]}...\n\n"
            f"Would you like to download and install it?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.download_update(release_data)
    
    def download_update(self, release_data):
        """Download and install update"""
        # Find the appropriate download URL for the current platform
        platform = sys.platform
        download_url = None
        
        for asset in release_data['assets']:
            if platform == 'darwin' and 'mac' in asset['name'].lower():
                download_url = asset['browser_download_url']
                break
            elif platform == 'win32' and 'win' in asset['name'].lower():
                download_url = asset['browser_download_url']
                break
            elif platform == 'linux' and 'linux' in asset['name'].lower():
                download_url = asset['browser_download_url']
                break
        
        if download_url:
            self.perform_download(download_url)
        else:
            QMessageBox.information(
                self.parent,
                "Manual Download Required",
                f"Please download the update manually from:\n{release_data['html_url']}"
            )
    
    def perform_download(self, download_url):
        """Download the update file"""
        try:
            # Show progress dialog
            progress = QProgressDialog("Downloading update...", "Cancel", 0, 100, self.parent)
            progress.setWindowModality(2)  # Application modal
            progress.show()
            
            # Download file
            response = requests.get(download_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            # Save to temp directory
            temp_dir = tempfile.gettempdir()
            filename = download_url.split('/')[-1]
            temp_file = os.path.join(temp_dir, filename)
            
            downloaded = 0
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if progress.wasCanceled():
                        return
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress.setValue(int(downloaded * 100 / total_size))
            
            progress.close()
            
            # Show completion and instructions
            QMessageBox.information(
                self.parent,
                "Update Downloaded",
                f"Update downloaded to:\n{temp_file}\n\n"
                f"Please close this application and run the new version."
            )
            
            # Optionally open the download location
            if sys.platform == 'darwin':
                subprocess.run(['open', temp_dir])
            elif sys.platform == 'win32':
                subprocess.run(['explorer', temp_dir])
            
        except Exception as e:
            QMessageBox.critical(self.parent, "Download Error", f"Failed to download update: {str(e)}")
    
    def no_update_available(self):
        if not self.silent:
            QMessageBox.information(self.parent, "No Updates", "You have the latest version!")
    
    def update_error(self, error_msg):
        if not self.silent:
            QMessageBox.warning(self.parent, "Update Check Failed", f"Could not check for updates: {error_msg}")