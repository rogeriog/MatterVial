"""
A module to handle the encoding and decoding of data between original feature
manifolds (OFM, MatMiner) and their latent-space representations using
pre-trained autoencoders.
"""

import sys
import os
import pandas as pd
import requests
import tarfile
import tempfile

# --- Path Setup ---
# This block ensures that the 'autoencoder_tools' package can be found and imported.
# It calculates the absolute path to the 'packages' directory by navigating
# up two levels from this script's location.
try:
    packages_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'packages'))
    if packages_path not in sys.path:
        sys.path.insert(0, packages_path)
    from autoencoder_tools.data_processing import encode_dataset, decode_dataset
    _AUTOENCODER_TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import 'autoencoder_tools'. Decoder functions will not work until dependencies are available.")
    print(f"Package path: {packages_path}")
    print(f"Error: {e}")
    _AUTOENCODER_TOOLS_AVAILABLE = False

    # Create dummy functions for testing
    def encode_dataset(*args, **kwargs):
        raise ImportError("autoencoder_tools not available")

    def decode_dataset(*args, **kwargs):
        raise ImportError("autoencoder_tools not available")


# --- Model Configuration ---
# Get the directory where the model files are located.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DECODERS_DIR = os.path.join(_CURRENT_DIR, 'decoders')

# --- Figshare Download Configuration ---
_FIGSHARE_DECODERS_URL = "https://figshare.com/ndownloader/files/57758209"
_DECODERS_ARCHIVE_NAME = "decoders_archive.tar.gz"

# --- OFM Model Files ---
_OFM_SCALER_FILE = os.path.join(_DECODERS_DIR, 'OFM_scaler_autoencoder.pkl')
_OFM_COLUMNS_FILE = os.path.join(_DECODERS_DIR, 'OFM_encoded_columns.txt')
_OFM_AUTOENCODER_FILE = os.path.join(_DECODERS_DIR, 'OFM__AutoEncoder.h5')

# --- MatMiner (MM) Model Files ---
_MM_SCALER_FILE = os.path.join(_DECODERS_DIR, 'MM_scaler_autoencoder.pkl')
_MM_COLUMNS_FILE = os.path.join(_DECODERS_DIR, 'MM_encoded_columns.txt')
_MM_AUTOENCODER_FILE = os.path.join(_DECODERS_DIR, 'MM__AutoEncoder.h5')


# ----------------------------------------------------------------------------
# Decoder Files Download Functions
# ----------------------------------------------------------------------------

def _ensure_decoders_available():
    """
    Ensure decoder files are available locally. Download and extract if needed.

    This function checks if the required decoder files exist locally. If not,
    it automatically downloads the decoder archive from Figshare and extracts
    the files to the correct location.

    The decoder archive contains:
    - MM__AutoEncoder.h5 (130MB) - MatMiner autoencoder model
    - OFM__AutoEncoder.h5 (62MB) - OFM autoencoder model
    - Associated scaler and column files

    Raises:
        RuntimeError: If download or extraction fails
        FileNotFoundError: If critical files are missing after extraction
    """
    # Check if critical decoder files exist
    critical_files = [
        _MM_AUTOENCODER_FILE,
        _OFM_AUTOENCODER_FILE,
        _MM_SCALER_FILE,
        _OFM_SCALER_FILE,
        _MM_COLUMNS_FILE,
        _OFM_COLUMNS_FILE
    ]

    missing_files = [f for f in critical_files if not os.path.exists(f)]

    if not missing_files:
        return  # All files are available

    # Create decoders directory if it doesn't exist
    os.makedirs(_DECODERS_DIR, exist_ok=True)

    print("Decoder files not found locally. Downloading from Figshare...")
    print(f"Missing files: {len(missing_files)}/{len(critical_files)}")
    _download_and_extract_decoders()

    # Verify files were extracted successfully
    still_missing = [f for f in critical_files if not os.path.exists(f)]
    if still_missing:
        raise FileNotFoundError(
            f"Critical decoder files still missing after download: {still_missing}"
        )


def _download_and_extract_decoders():
    """
    Download decoder archive from Figshare and extract it.

    Downloads the compressed decoder archive (~50MB compressed, ~191MB extracted)
    containing all autoencoder models and associated files.

    Raises:
        RuntimeError: If download or extraction fails
    """
    try:
        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = os.path.join(temp_dir, _DECODERS_ARCHIVE_NAME)

            print(f"Downloading decoder archive (~50MB compressed)...")

            # Download with progress indication
            response = requests.get(_FIGSHARE_DECODERS_URL, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rDownloading: {progress:.1f}%", end='', flush=True)

            print(f"\nDownload complete. Extracting to {_DECODERS_DIR}...")

            # Extract the archive
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path=_CURRENT_DIR)

            print("Decoder files extracted successfully!")

    except Exception as e:
        raise RuntimeError(f"Failed to download decoder files: {e}")


# ----------------------------------------------------------------------------
# OFM Encoder/Decoder Functions
# ----------------------------------------------------------------------------

def encode_ofm(ofm_dataframe: pd.DataFrame, save_path: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Encodes a DataFrame from the Original Feature Manifold (OFM) to the latent space (l-OFM).

    This function automatically downloads the required OFM decoder files from Figshare
    if they are not available locally (~50MB compressed, ~191MB extracted).

    Args:
        ofm_dataframe (pd.DataFrame): The input data with original OFM features.
        save_path (str, optional): Path to save the encoded .pkl file. Defaults to None.
        verbose (bool, optional): If True, prints status messages. Defaults to True.

    Returns:
        pd.DataFrame: The encoded data in the latent space.

    Note:
        First-time usage will trigger a download of decoder files from Figshare.
        Subsequent calls will use the cached local files.

    Raises:
        ImportError: If autoencoder_tools dependencies are not available.
    """
    if not _AUTOENCODER_TOOLS_AVAILABLE:
        raise ImportError(
            "autoencoder_tools package is not available. "
            "Please ensure all dependencies are installed."
        )

    # Ensure decoder files are available
    _ensure_decoders_available()

    if verbose:
        print("\n--- Encoding OFM data to l-OFM ---")

    lofm_dataframe = encode_dataset(
        dataset=ofm_dataframe,
        scaler=_OFM_SCALER_FILE,
        columns_to_read=_OFM_COLUMNS_FILE,
        autoencoder=_OFM_AUTOENCODER_FILE,
        save_name=save_path,
        feat_prefix='l-OFM'
    )

    if verbose:
        print("OFM Encoding complete.")
    return lofm_dataframe


def decode_ofm(lofm_dataframe: pd.DataFrame, save_path: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Decodes a DataFrame from the latent space (l-OFM) back to the Original Feature Manifold (OFM).

    This function automatically downloads the required OFM decoder files from Figshare
    if they are not available locally (~50MB compressed, ~191MB extracted).

    Args:
        lofm_dataframe (pd.DataFrame): The input data in the l-OFM latent space.
        save_path (str, optional): Path to save the decoded .pkl file. Defaults to None.
        verbose (bool, optional): If True, prints status messages. Defaults to True.

    Returns:
        pd.DataFrame: The decoded data in the original OFM space.

    Note:
        First-time usage will trigger a download of decoder files from Figshare.
        Subsequent calls will use the cached local files.

    Raises:
        ImportError: If autoencoder_tools dependencies are not available.
    """
    if not _AUTOENCODER_TOOLS_AVAILABLE:
        raise ImportError(
            "autoencoder_tools package is not available. "
            "Please ensure all dependencies are installed."
        )

    # Ensure decoder files are available
    _ensure_decoders_available()

    if verbose:
        print("\n--- Decoding l-OFM data to OFM ---")

    ofm_dataframe = decode_dataset(
        dataset=lofm_dataframe,
        scaler=_OFM_SCALER_FILE,
        columns_to_read=_OFM_COLUMNS_FILE,
        autoencoder=_OFM_AUTOENCODER_FILE,
        save_name=save_path,
        encoder_type="regular"
    )

    if verbose:
        print("OFM Decoding complete.")
    return ofm_dataframe


# ----------------------------------------------------------------------------
# MatMiner (MM) Encoder/Decoder Functions
# ----------------------------------------------------------------------------

def encode_mm(mm_dataframe: pd.DataFrame, save_path: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Encodes a DataFrame from the MatMiner feature space (MM) to the latent space (l-MM).

    This function automatically downloads the required MatMiner decoder files from Figshare
    if they are not available locally (~50MB compressed, ~191MB extracted).

    Args:
        mm_dataframe (pd.DataFrame): The input data with original MatMiner features.
        save_path (str, optional): Path to save the encoded .pkl file. Defaults to None.
        verbose (bool, optional): If True, prints status messages. Defaults to True.

    Returns:
        pd.DataFrame: The encoded data in the latent space.

    Note:
        First-time usage will trigger a download of decoder files from Figshare.
        Subsequent calls will use the cached local files.

    Raises:
        ImportError: If autoencoder_tools dependencies are not available.
    """
    if not _AUTOENCODER_TOOLS_AVAILABLE:
        raise ImportError(
            "autoencoder_tools package is not available. "
            "Please ensure all dependencies are installed."
        )

    # Ensure decoder files are available
    _ensure_decoders_available()

    if verbose:
        print("\n--- Encoding MatMiner data to l-MM ---")

    lmm_dataframe = encode_dataset(
        dataset=mm_dataframe,
        scaler=_MM_SCALER_FILE,
        columns_to_read=_MM_COLUMNS_FILE,
        autoencoder=_MM_AUTOENCODER_FILE,
        save_name=save_path,
        feat_prefix='l-MM'
    )

    if verbose:
        print("MatMiner Encoding complete.")
    return lmm_dataframe


def decode_mm(lmm_dataframe: pd.DataFrame, save_path: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Decodes a DataFrame from the latent space (l-MM) back to the MatMiner feature space (MM).

    This function automatically downloads the required MatMiner decoder files from Figshare
    if they are not available locally (~50MB compressed, ~191MB extracted).

    Args:
        lmm_dataframe (pd.DataFrame): The input data in the l-MM latent space.
        save_path (str, optional): Path to save the decoded .pkl file. Defaults to None.
        verbose (bool, optional): If True, prints status messages. Defaults to True.

    Returns:
        pd.DataFrame: The decoded data in the original MatMiner feature space.

    Note:
        First-time usage will trigger a download of decoder files from Figshare.
        Subsequent calls will use the cached local files.

    Raises:
        ImportError: If autoencoder_tools dependencies are not available.
    """
    if not _AUTOENCODER_TOOLS_AVAILABLE:
        raise ImportError(
            "autoencoder_tools package is not available. "
            "Please ensure all dependencies are installed."
        )

    # Ensure decoder files are available
    _ensure_decoders_available()

    if verbose:
        print("\n--- Decoding l-MM data to MatMiner ---")

    mm_dataframe = decode_dataset(
        dataset=lmm_dataframe,
        scaler=_MM_SCALER_FILE,
        columns_to_read=_MM_COLUMNS_FILE,
        autoencoder=_MM_AUTOENCODER_FILE,
        save_name=save_path,
        encoder_type="regular"
    )

    if verbose:
        print("MatMiner Decoding complete.")
    return mm_dataframe


# ----------------------------------------------------------------------------
# Utility Functions for Cache Management
# ----------------------------------------------------------------------------

def clear_decoder_cache():
    """
    Remove locally cached decoder files to free up disk space.

    This function removes all decoder files (~191MB) from the local cache.
    The files will be automatically re-downloaded when needed.

    Returns:
        bool: True if cache was cleared successfully, False if no cache existed
    """
    import shutil

    if os.path.exists(_DECODERS_DIR):
        try:
            shutil.rmtree(_DECODERS_DIR)
            print("Decoder cache cleared successfully.")
            return True
        except Exception as e:
            print(f"Error clearing decoder cache: {e}")
            return False
    else:
        print("No decoder cache found to clear.")
        return False


def get_decoder_cache_info():
    """
    Get information about cached decoder files.

    Returns:
        dict: Dictionary containing cache status, size, and file count information
    """
    if not os.path.exists(_DECODERS_DIR):
        return {
            "status": "not_cached",
            "size_bytes": 0,
            "size_mb": 0,
            "files": 0,
            "missing_files": []
        }

    # Check which critical files exist
    critical_files = [
        _MM_AUTOENCODER_FILE,
        _OFM_AUTOENCODER_FILE,
        _MM_SCALER_FILE,
        _OFM_SCALER_FILE,
        _MM_COLUMNS_FILE,
        _OFM_COLUMNS_FILE
    ]

    existing_files = [f for f in critical_files if os.path.exists(f)]
    missing_files = [os.path.basename(f) for f in critical_files if not os.path.exists(f)]

    # Calculate total size
    total_size = 0
    total_files = 0

    if os.path.exists(_DECODERS_DIR):
        for root, dirs, files in os.walk(_DECODERS_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    total_files += 1
                    total_size += os.path.getsize(file_path)

    status = "complete" if not missing_files else "partial" if existing_files else "not_cached"

    return {
        "status": status,
        "size_bytes": total_size,
        "size_mb": round(total_size / (1024 * 1024), 2),
        "files": total_files,
        "critical_files_present": len(existing_files),
        "critical_files_total": len(critical_files),
        "missing_files": missing_files
    }


def configure_figshare_url(new_url: str):
    """
    Configure the Figshare URL for decoder downloads.

    This function allows updating the Figshare download URL without modifying
    the source code. Useful for testing or when the Figshare URL changes.

    Args:
        new_url (str): The new Figshare download URL

    Example:
        >>> configure_figshare_url("https://figshare.com/ndownloader/files/12345678")
    """
    global _FIGSHARE_DECODERS_URL
    _FIGSHARE_DECODERS_URL = new_url
    print(f"Figshare URL updated to: {new_url}")


# ----------------------------------------------------------------------------
# Module Information
# ----------------------------------------------------------------------------

def get_decoder_info():
    """
    Get comprehensive information about the decoder module and cached files.

    Returns:
        dict: Complete information about decoder configuration and cache status
    """
    cache_info = get_decoder_cache_info()

    return {
        "module_version": "1.0.0",
        "figshare_url": _FIGSHARE_DECODERS_URL,
        "decoders_directory": _DECODERS_DIR,
        "archive_name": _DECODERS_ARCHIVE_NAME,
        "cache_info": cache_info,
        "supported_operations": [
            "encode_ofm", "decode_ofm",
            "encode_mm", "decode_mm"
        ],
        "file_types": {
            "autoencoders": ["MM__AutoEncoder.h5", "OFM__AutoEncoder.h5"],
            "scalers": ["MM_scaler_autoencoder.pkl", "OFM_scaler_autoencoder.pkl"],
            "columns": ["MM_encoded_columns.txt", "OFM_encoded_columns.txt"]
        }
    }