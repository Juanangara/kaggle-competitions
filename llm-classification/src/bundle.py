"""Copy the inference code next to the trained artifacts: one folder to upload to Kaggle."""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["data.py", "features.py", "predict.py"]

if __name__ == "__main__":
    out = ROOT / "artifacts"
    assert (out / "lgbm.txt").exists(), "run `python src/train.py --embeddings` first"
    for name in FILES:
        shutil.copy(ROOT / "src" / name, out / name)
    print("bundle:", sorted(p.name for p in out.iterdir()))
