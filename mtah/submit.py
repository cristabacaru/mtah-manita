"""Write the submission: labels.npy -> Manita.csv (+ Manita.csv.sha256), then run the official validator.

    python -m mtah.submit                                  # $MTAH_CACHE/BAR-ATH/labels.npy -> ./Manita.csv
    python -m mtah.submit --labels other.npy               # any (n,) H/A/D/X array
    python -m mtah.submit --labels baseline_smooth50.csv   # or any frame,label CSV (fallback)
    python -m mtah.submit --out /tmp/try.csv               # write somewhere else

What it guarantees (README "Before submitting", guide sections 1 and 2):
- exactly config.N_FRAMES rows, header "frame,label", frame 0..n-1, one label per row, LF line ends
  (the format of starter/baseline.py's write_submission);
- X on every frame outside the graded halves (config.HALF_BLOCKS), whatever the input says there;
- no X inside the halves: the scorer forward-fills X for spellF1 and counts it as a miss on macroF1,
  so a graded X is always a bug upstream and the run stops instead of writing it;
- the same labels always give the same bytes, so `shasum -a 256 Manita.csv` is reproducible;
- evaluation/validate_submission.py (official, unedited) passes, or the command exits with 1.

The CSV goes to the repo root (config.SUBMISSION_CSV), which .gitignore keeps out of git.
Prints aggregates only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from mtah import config

VALID = ("H", "A", "D", "X")
VALIDATOR = config.REPO_ROOT / "evaluation" / "validate_submission.py"


def load_labels(path: str | Path) -> np.ndarray:
    """(n,) array of label strings from a .npy (decode.py's labels.npy) or a frame,label .csv."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.asarray(np.load(path, allow_pickle=False)).astype(str)
    if path.suffix == ".csv":
        eval_dir = str(config.REPO_ROOT / "evaluation")
        if eval_dir not in sys.path:
            sys.path.insert(0, eval_dir)
        import possession_metrics as pm   # official reader, so we read CSVs as the scorer does
        table = pm.read_label_csv(path)
        if not np.array_equal(table["frame"].to_numpy(), np.arange(len(table))):
            raise ValueError(f"{path}: frame column is not 0..n-1 in order")
        return np.array([str(v).strip().upper() for v in table["label"]], dtype=str)
    raise ValueError(f"{path}: expected a .npy or .csv file")


def prepare(labels, n_frames: int = config.N_FRAMES, half_blocks=config.HALF_BLOCKS) -> tuple[np.ndarray, dict]:
    """Checked labels with X forced outside the halves, plus counts of what was fixed.

    Raises ValueError on a wrong length, a label outside H/A/D/X, or any X inside a half.
    """
    lab = np.asarray(labels).astype(str)
    if lab.shape != (n_frames,):
        raise ValueError(f"labels have shape {lab.shape}, the submission needs ({n_frames},)")
    bad = sorted(set(np.unique(lab).tolist()) - set(VALID))
    if bad:
        raise ValueError(f"labels outside {list(VALID)}: {bad[:5]}")

    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in half_blocks:
        graded[first:last + 1] = True
    graded_x = int(np.sum(graded & (lab == "X")))
    if graded_x:
        raise ValueError(f"{graded_x} graded frames are X. X inside a half is always scored as wrong; "
                         "fix the stage that produced the labels (decode.py) instead of submitting it")
    forced = int(np.sum(~graded & (lab != "X")))
    out = lab.copy()
    out[~graded] = "X"
    return out.astype("<U1"), {"forced_x_outside": forced}


def csv_bytes(labels: np.ndarray) -> bytes:
    """The whole file: header, then 'frame,label' per frame, LF line ends, ASCII."""
    body = "".join(f"{i},{lab}\n" for i, lab in enumerate(labels.tolist()))
    return ("frame,label\n" + body).encode("ascii")


def write(labels: np.ndarray, path: str | Path) -> str:
    """Write the CSV atomically and <path>.sha256 next to it (shasum format). Returns the sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = csv_bytes(labels)
    digest = hashlib.sha256(data).hexdigest()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def validate(path: str | Path, n_frames: int = config.N_FRAMES) -> tuple[bool, str]:
    """Run the official validator on the file; (passed, its output)."""
    result = subprocess.run([sys.executable, str(VALIDATOR), str(path), "--rows", str(n_frames)],
                            capture_output=True, text=True)
    return result.returncode == 0 and "Valid." in result.stdout, result.stdout + result.stderr


def run(labels_path: str | Path | None = None, out: str | Path | None = None) -> dict:
    labels_path = Path(labels_path) if labels_path else config.game_cache(config.GAME) / "labels.npy"
    out = Path(out) if out else config.SUBMISSION_CSV
    if not labels_path.is_file():
        raise SystemExit(f"no labels at {labels_path}: run the decoder first (bash run.sh), "
                         "or pass --labels <file>")
    labels, fixes = prepare(load_labels(labels_path))
    digest = write(labels, out)
    passed, report = validate(out)
    counts = {k: int(np.sum(labels == k)) for k in VALID}
    return {"path": out, "sha256": digest, "valid": passed, "validator": report,
            "counts": counts, "source": labels_path, **fixes}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", help="labels .npy or frame,label .csv (default: $MTAH_CACHE/<game>/labels.npy)")
    parser.add_argument("--out", help=f"output CSV (default: {config.SUBMISSION_CSV.name} in the repo root)")
    args = parser.parse_args(argv)
    try:
        info = run(args.labels, args.out)
    except ValueError as err:
        print(f"NOT WRITTEN: {err}")
        return 1
    print(f"labels from  {info['source']}")
    print("counts       " + "  ".join(f"{k} {v}" for k, v in info["counts"].items()))
    if info["forced_x_outside"]:
        print(f"note         {info['forced_x_outside']} frames outside the halves were not X; written as X")
    print(f"wrote        {info['path']}\nsha256       {info['sha256']}  (also in {Path(info['path']).name}.sha256)")
    print("\nofficial validator:\n" + "\n".join("  " + line for line in info["validator"].rstrip().splitlines()))
    if not info["valid"]:
        print("\nFAILED: the official validator rejected the file. Do not submit it.")
        return 1
    print("\nNext: README 'Before submitting' (run twice, same sha256; repo public; Google form).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
