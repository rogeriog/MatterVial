"""
A module to handle the encoding and decoding of data between original feature
manifolds (OFM, MatMiner) and their latent-space representations using
pre-trained autoencoders.
"""

import sys
import os
import pandas as pd

# --- Path Setup ---
# This block ensures that the 'autoencoder_tools' package can be found and imported.
# It calculates the absolute path to the 'packages' directory by navigating
# up two levels from this script's location.
try:
    packages_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'packages'))
    if packages_path not in sys.path:
        sys.path.insert(0, packages_path)
    from autoencoder_tools.data_processing import encode_dataset, decode_dataset
except ImportError as e:
    print(f"Error: Could not import 'autoencoder_tools'. Please ensure the package exists at {packages_path}")
    sys.exit(1)


# --- Model Configuration ---
# Get the directory where the model files are located.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DECODERS_DIR = os.path.join(_CURRENT_DIR, 'decoders')

# --- OFM Model Files ---
_OFM_SCALER_FILE = os.path.join(_DECODERS_DIR, 'OFM_scaler_autoencoder.pkl')
_OFM_COLUMNS_FILE = os.path.join(_DECODERS_DIR, 'OFM_encoded_columns.txt')
_OFM_AUTOENCODER_FILE = os.path.join(_DECODERS_DIR, 'OFM__AutoEncoder.h5')

# --- MatMiner (MM) Model Files ---
_MM_SCALER_FILE = os.path.join(_DECODERS_DIR, 'MM_scaler_autoencoder.pkl')
_MM_COLUMNS_FILE = os.path.join(_DECODERS_DIR, 'MM_encoded_columns.txt')
_MM_AUTOENCODER_FILE = os.path.join(_DECODERS_DIR, 'MM__AutoEncoder.h5')


# ----------------------------------------------------------------------------
# OFM Encoder/Decoder Functions
# ----------------------------------------------------------------------------

def encode_ofm(ofm_dataframe: pd.DataFrame, save_path: str = None, verbose: bool = True) -> pd.DataFrame:
    """
    Encodes a DataFrame from the Original Feature Manifold (OFM) to the latent space (l-OFM).

    Args:
        ofm_dataframe (pd.DataFrame): The input data with original OFM features.
        save_path (str, optional): Path to save the encoded .pkl file. Defaults to None.
        verbose (bool, optional): If True, prints status messages. Defaults to True.

    Returns:
        pd.DataFrame: The encoded data in the latent space.
    """
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

    Args:
        lofm_dataframe (pd.DataFrame): The input data in the l-OFM latent space.
        save_path (str, optional): Path to save the decoded .pkl file. Defaults to None.
        verbose (bool, optional): If True, prints status messages. Defaults to True.

    Returns:
        pd.DataFrame: The decoded data in the original OFM space.
    """
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

    Args:
        mm_dataframe (pd.DataFrame): The input data with original MatMiner features.
        save_path (str, optional): Path to save the encoded .pkl file. Defaults to None.
        verbose (bool, optional): If True, prints status messages. Defaults to True.

    Returns:
        pd.DataFrame: The encoded data in the latent space.
    """
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

    Args:
        lmm_dataframe (pd.DataFrame): The input data in the l-MM latent space.
        save_path (str, optional): Path to save the decoded .pkl file. Defaults to None.
        verbose (bool, optional): If True, prints status messages. Defaults to True.

    Returns:
        pd.DataFrame: The decoded data in the original MatMiner feature space.
    """
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