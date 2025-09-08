# test_shap_download.py
import tempfile
import shutil
from mattervial.interpreter import Interpreter

def test_shap_download():
    """Test SHAP plots download functionality"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize interpreter with temporary directory
        interpreter = Interpreter(base_dir=temp_dir)
        
        # This should trigger download
        try:
            interpreter.display_svg("MEGNet_MatMiner_1")
            print("✓ Download and display test passed")
        except Exception as e:
            print(f"✗ Test failed: {e}")

if __name__ == "__main__":
    test_shap_download()
