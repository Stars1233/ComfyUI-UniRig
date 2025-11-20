"""
Installation script for ComfyUI-UniRig dependencies.

This script is automatically run by ComfyUI Manager to install
dependencies that require special handling (torch-cluster, torch-scatter, spconv).
Also handles automatic Blender installation for mesh preprocessing.
"""

import subprocess
import sys
import os
import platform
import urllib.request
import tarfile
import zipfile
import shutil
from pathlib import Path

def install_requirements():
    """Install basic requirements from requirements.txt."""
    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print(f"[UniRig Install] Warning: requirements.txt not found at {requirements_file}")
        return True

    print(f"[UniRig Install] Installing basic dependencies from requirements.txt...")

    cmd = [
        sys.executable, "-m", "pip", "install",
        "-r", str(requirements_file)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[UniRig Install] ✓ Basic dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print("[UniRig Install] ✗ Failed to install basic dependencies")
        print(f"[UniRig Install] Error: {e.stderr}")
        return False

def get_torch_info():
    """Get PyTorch and CUDA versions."""
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # e.g., "2.3.1"

        if torch.cuda.is_available():
            # Get CUDA version from torch
            cuda_version = torch.version.cuda  # e.g., "12.1"
            if cuda_version:
                # Convert to format like cu121
                cuda_suffix = 'cu' + cuda_version.replace('.', '')
            else:
                cuda_suffix = 'cpu'
        else:
            cuda_suffix = 'cpu'

        return torch_version, cuda_suffix
    except ImportError:
        print("[UniRig Install] ERROR: PyTorch not found. Please install PyTorch first.")
        sys.exit(1)

def install_torch_geometric_deps(torch_version, cuda_suffix):
    """Install torch-cluster and torch-scatter."""
    print(f"[UniRig Install] Detected PyTorch {torch_version} with {cuda_suffix}")

    # Check if already installed
    try:
        import torch_cluster
        import torch_scatter
        print("[UniRig Install] torch-cluster and torch-scatter already installed")
        return True
    except ImportError:
        pass

    # Construct the PyTorch Geometric wheel URL
    # Format: https://data.pyg.org/whl/torch-{version}+{cuda}/torch_cluster-{version}.html
    base_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_suffix}.html"

    print(f"[UniRig Install] Installing torch-cluster and torch-scatter from PyTorch Geometric...")
    print(f"[UniRig Install] Wheel URL: {base_url}")

    # Install both packages
    packages = ["torch-scatter", "torch-cluster"]

    for package in packages:
        print(f"[UniRig Install] Installing {package}...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            package,
            "-f", base_url,
            "--no-cache-dir"
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[UniRig Install] ✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[UniRig Install] ✗ Failed to install {package}")
            print(f"[UniRig Install] Error: {e.stderr}")
            return False

    return True

def install_spconv(cuda_suffix):
    """Install spconv if CUDA is available."""
    if cuda_suffix == 'cpu':
        print("[UniRig Install] Skipping spconv (CPU-only environment)")
        return True

    # Check if already installed
    try:
        import spconv
        print("[UniRig Install] spconv already installed")
        return True
    except ImportError:
        pass

    print(f"[UniRig Install] Installing spconv for {cuda_suffix}...")

    # Try different spconv versions/names
    # spconv package naming can vary: spconv-cu120, spconv-cu118, etc.
    # Newer CUDA versions may need to fall back to older spconv versions

    # Map CUDA versions to compatible spconv versions
    cuda_to_spconv = {
        'cu128': ['cu121', 'cu120'],  # CUDA 12.8 -> try cu121 or cu120
        'cu127': ['cu121', 'cu120'],
        'cu126': ['cu121', 'cu120'],
        'cu125': ['cu121', 'cu120'],
        'cu124': ['cu121', 'cu120'],
        'cu123': ['cu121', 'cu120'],
        'cu122': ['cu121', 'cu120'],
        'cu121': ['cu121', 'cu120'],
        'cu120': ['cu120'],
        'cu118': ['cu118'],
        'cu117': ['cu117'],
    }

    # Get list of versions to try
    versions_to_try = cuda_to_spconv.get(cuda_suffix, [cuda_suffix])

    for spconv_cuda in versions_to_try:
        spconv_package = f"spconv-{spconv_cuda}"
        print(f"[UniRig Install] Trying {spconv_package}...")

        cmd = [
            sys.executable, "-m", "pip", "install",
            spconv_package
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[UniRig Install] ✓ {spconv_package} installed successfully")
            return True
        except subprocess.CalledProcessError:
            continue

    print(f"[UniRig Install] ✗ Failed to install spconv for {cuda_suffix}")
    print(f"[UniRig Install] Note: spconv is optional. UniRig may work without it.")
    print(f"[UniRig Install] For manual installation, see https://github.com/traveller59/spconv")
    # Don't fail the entire installation if spconv fails - it might not be critical
    return True

def install_flash_attn():
    """Install flash-attn from official prebuilt wheels."""
    # Check if already installed
    try:
        import flash_attn
        print("[UniRig Install] flash-attn already installed")
        return True
    except ImportError:
        pass

    print("[UniRig Install] Installing flash-attn from official prebuilt wheel...")

    # Get PyTorch and CUDA info
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # e.g., "2.8.0"
        torch_major_minor = '.'.join(torch_version.split('.')[:2])  # e.g., "2.8"

        if not torch.cuda.is_available():
            print("[UniRig Install] CUDA not available, skipping flash-attn (GPU-only)")
            return True

        cuda_version = torch.version.cuda  # e.g., "12.8"
        cuda_major = cuda_version.split('.')[0] if cuda_version else None

        if not cuda_major:
            print("[UniRig Install] Could not detect CUDA version, skipping flash-attn")
            return True

    except ImportError:
        print("[UniRig Install] PyTorch not found, skipping flash-attn")
        return True

    # Construct the official wheel URL
    # Format: flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
    flash_attn_version = "2.8.3"  # Latest version as of installation
    python_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

    wheel_url = (
        f"https://github.com/Dao-AILab/flash-attention/releases/download/"
        f"v{flash_attn_version}/flash_attn-{flash_attn_version}%2Bcu{cuda_major}"
        f"torch{torch_major_minor}cxx11abiTRUE-{python_version}-{python_version}-linux_x86_64.whl"
    )

    print(f"[UniRig Install] Detected PyTorch {torch_version}, CUDA {cuda_version}, Python {python_version}")
    print(f"[UniRig Install] Downloading from: {wheel_url}")

    cmd = [
        sys.executable, "-m", "pip", "install",
        wheel_url
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        print("[UniRig Install] flash-attn installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print("[UniRig Install] flash-attn installation failed")
        print("[UniRig Install] Note: flash-attn is optional but recommended for performance.")
        print("[UniRig Install] You may need to install manually from:")
        print(f"[UniRig Install]   https://github.com/Dao-AILab/flash-attention/releases")
        return True  # Don't fail - it's optional
    except subprocess.TimeoutExpired:
        print("[UniRig Install] flash-attn installation timed out")
        return True  # Don't fail - it's optional


# Blender Installation Functions

def get_platform_info():
    """Detect current platform and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        plat = "macos"
        arch = "arm64" if machine == "arm64" else "x64"
    elif system == "linux":
        plat = "linux"
        arch = "x64"
    elif system == "windows":
        plat = "windows"
        arch = "x64"
    else:
        plat = None
        arch = None

    return plat, arch


def get_blender_download_url(platform_name, architecture):
    """
    Get Blender 4.2 LTS download URL for the platform.

    Args:
        platform_name: "linux", "macos", or "windows"
        architecture: "x64" or "arm64"

    Returns:
        tuple: (download_url, version, filename) or (None, None, None) if not found
    """
    version = "4.2.3"
    base_url = "https://download.blender.org/release/Blender4.2"

    urls = {
        ("linux", "x64"): (
            f"{base_url}/blender-{version}-linux-x64.tar.xz",
            version,
            f"blender-{version}-linux-x64.tar.xz"
        ),
        ("macos", "x64"): (
            f"{base_url}/blender-{version}-macos-x64.dmg",
            version,
            f"blender-{version}-macos-x64.dmg"
        ),
        ("macos", "arm64"): (
            f"{base_url}/blender-{version}-macos-arm64.dmg",
            version,
            f"blender-{version}-macos-arm64.dmg"
        ),
        ("windows", "x64"): (
            f"{base_url}/blender-{version}-windows-x64.zip",
            version,
            f"blender-{version}-windows-x64.zip"
        ),
    }

    key = (platform_name, architecture)
    if key in urls:
        url, ver, filename = urls[key]
        print(f"[UniRig Install] Using Blender {ver} for {platform_name}-{architecture}")
        return url, ver, filename

    return None, None, None


def download_file(url, dest_path):
    """Download file with progress."""
    print(f"[UniRig Install] Downloading: {url}")
    print(f"[UniRig Install] Destination: {dest_path}")

    last_printed_percent = [-1]  # Use list to allow modification in nested function

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)

        # Only print every 10% to reduce verbosity
        if percent >= last_printed_percent[0] + 10 or percent >= 100:
            sys.stdout.write(f"\r[UniRig Install] Progress: {percent}%")
            sys.stdout.flush()
            last_printed_percent[0] = percent

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        sys.stdout.write("\n")
        sys.stdout.flush()
        print("[UniRig Install] Download complete!")
        return True
    except Exception as e:
        print(f"\n[UniRig Install] Error downloading: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract tar.gz, tar.xz, zip, or handle DMG (macOS)."""
    print(f"[UniRig Install] Extracting: {archive_path}")

    try:
        if archive_path.endswith(('.tar.gz', '.tar.xz', '.tar.bz2')):
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(extract_to)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.dmg'):
            print("[UniRig Install] DMG detected - mounting disk image...")

            mount_result = subprocess.run(
                ['hdiutil', 'attach', '-nobrowse', archive_path],
                capture_output=True,
                text=True
            )

            if mount_result.returncode != 0:
                print(f"[UniRig Install] Error mounting DMG: {mount_result.stderr}")
                return False

            mount_point = None
            for line in mount_result.stdout.split('\n'):
                if '/Volumes/' in line:
                    mount_point = line.split('\t')[-1].strip()
                    break

            if not mount_point:
                print("[UniRig Install] Error: Could not find mount point")
                return False

            try:
                blender_app = Path(mount_point) / "Blender.app"
                if blender_app.exists():
                    dest_app = Path(extract_to) / "Blender.app"
                    shutil.copytree(blender_app, dest_app)
                    print(f"[UniRig Install] Copied Blender.app to: {dest_app}")
                else:
                    print(f"[UniRig Install] Error: Blender.app not found in {mount_point}")
                    return False

            finally:
                subprocess.run(['hdiutil', 'detach', mount_point], check=False)

        else:
            print(f"[UniRig Install] Error: Unknown archive format: {archive_path}")
            return False

        print(f"[UniRig Install] Extraction complete!")
        return True

    except Exception as e:
        print(f"[UniRig Install] Error extracting: {e}")
        return False


def find_blender_executable(blender_dir):
    """Find the blender executable in the extracted directory."""
    plat, _ = get_platform_info()

    if plat == "windows":
        exe_pattern = "**/blender.exe"
    elif plat == "macos":
        exe_pattern = "**/MacOS/blender"
    else:  # linux
        exe_pattern = "**/blender"

    executables = list(Path(blender_dir).glob(exe_pattern))

    if executables:
        return executables[0]
    return None


def install_blender(target_dir=None):
    """
    Install Blender for mesh preprocessing.

    Args:
        target_dir: Optional target directory. If None, uses lib/blender under script directory.

    Returns:
        str: Path to Blender executable, or None if installation failed.
    """
    print("\n" + "="*60)
    print("ComfyUI-UniRig: Blender Installation")
    print("="*60 + "\n")

    if target_dir is None:
        script_dir = Path(__file__).parent.absolute()
        target_dir = script_dir / "lib" / "blender"
    else:
        target_dir = Path(target_dir)

    # Check if Blender already installed
    blender_exe = find_blender_executable(target_dir)
    if blender_exe and blender_exe.exists():
        print("[UniRig Install] Blender already installed at:")
        print(f"[UniRig Install]   {blender_exe}")
        print("[UniRig Install] Skipping download.")
        return str(blender_exe)

    # Detect platform
    plat, arch = get_platform_info()
    if not plat or not arch:
        print("[UniRig Install] Error: Could not detect platform")
        print("[UniRig Install] Please install Blender manually from: https://www.blender.org/download/")
        return None

    print(f"[UniRig Install] Detected platform: {plat}-{arch}")

    # Get download URL
    url, version, filename = get_blender_download_url(plat, arch)
    if not url:
        print("[UniRig Install] Error: Could not find Blender download for your platform")
        print("[UniRig Install] Please install Blender manually from: https://www.blender.org/download/")
        return None

    # Create temporary download directory
    temp_dir = target_dir.parent / "_temp_blender_download"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download
        download_path = temp_dir / filename
        if not download_file(url, str(download_path)):
            return None

        # Extract
        target_dir.mkdir(parents=True, exist_ok=True)
        if not extract_archive(str(download_path), str(target_dir)):
            return None

        print("\n[UniRig Install] Blender installation complete!")
        print(f"[UniRig Install] Location: {target_dir}")

        # Find blender executable
        blender_exe = find_blender_executable(target_dir)

        if blender_exe:
            print(f"[UniRig Install] Blender executable: {blender_exe}")
            return str(blender_exe)
        else:
            print("[UniRig Install] Warning: Could not find blender executable")
            return None

    except Exception as e:
        print(f"\n[UniRig Install] Error during installation: {e}")
        return None

    finally:
        # Cleanup temp files
        if temp_dir.exists():
            print("[UniRig Install] Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main installation routine."""
    print("=" * 60)
    print("ComfyUI-UniRig: Installing dependencies")
    print("=" * 60)

    # Install basic requirements first (trimesh, numpy, etc.)
    if not install_requirements():
        print("[UniRig Install] Failed to install basic requirements")
        print("[UniRig Install] You may need to install manually:")
        print("[UniRig Install]   pip install -r requirements.txt")
        sys.exit(1)

    # Get PyTorch info
    torch_version, cuda_suffix = get_torch_info()

    # Install torch-cluster and torch-scatter
    if not install_torch_geometric_deps(torch_version, cuda_suffix):
        print("[UniRig Install] Failed to install PyTorch Geometric dependencies")
        print("[UniRig Install] You may need to install manually:")
        print(f"[UniRig Install]   pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-{torch_version}+{cuda_suffix}.html")
        sys.exit(1)

    # Install spconv
    install_spconv(cuda_suffix)

    # Try to install flash-attn (optional)
    install_flash_attn()

    print("=" * 60)
    print("[UniRig Install] Installation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
