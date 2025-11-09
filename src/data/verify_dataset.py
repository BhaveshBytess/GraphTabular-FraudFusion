"""Dataset verification utility for Elliptic++ files."""
import os
from pathlib import Path
from typing import List, Tuple


def verify_dataset(data_root: str) -> Tuple[bool, List[str]]:
    """
    Verify that all required Elliptic++ dataset files are present.
    
    Args:
        data_root: Path to 'Elliptic++ Dataset' folder
        
    Returns:
        (success: bool, messages: List[str])
    """
    required_files = [
        "txs_features.csv",
        "txs_classes.csv",
        "txs_edgelist.csv"
    ]
    
    messages = []
    all_present = True
    
    root_path = Path(data_root)
    
    if not root_path.exists():
        messages.append(f"❌ Dataset root not found: {data_root}")
        return False, messages
    
    messages.append(f"✅ Dataset root found: {data_root}")
    
    for filename in required_files:
        filepath = root_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            messages.append(f"✅ {filename} found ({size_mb:.2f} MB)")
        else:
            messages.append(f"❌ {filename} MISSING")
            all_present = False
    
    if all_present:
        messages.append("✅ All required dataset files present")
    else:
        messages.append("❌ Dataset incomplete - please provide missing files")
        messages.append("   Download from: https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l")
    
    return all_present, messages


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = "data/Elliptic++ Dataset"
    
    success, messages = verify_dataset(data_path)
    
    for msg in messages:
        print(msg)
    
    sys.exit(0 if success else 1)
