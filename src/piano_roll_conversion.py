"""
Project: MIDI Composer Classification
Role: Person B - Deep Learning Pipeline (Piano-Roll Conversion)
Description: Converts every MIDI file listed in midi_features_vertical.csv to a fixed-size
piano-roll matrix (64 notes x 1536 time steps at 10 Hz, velocity as value) and saves them
all in one file, in the SAME row order as Person A's feature CSV.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pretty_midi

ROOT = Path(__file__).resolve().parent.parent
MIDI_DIR = ROOT / "data" / "midi"
FEATURES_CSV = ROOT / "data" / "processed" / "midi_features_vertical.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "piano_rolls"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "piano_rolls.npz"

TARGET_TIME_STEPS = 1536
TARGET_NOTES = 64
FS = 10            # 10 Hz -> 1536 steps = 153.6 s
MIN_PITCH = 24     # C1
MAX_PITCH = MIN_PITCH + TARGET_NOTES - 1   # 87 = D#6... (64 semitones)


def midi_to_pianoroll(midi_path):
    """Return a (64, 1536) uint8 array (velocity 0-127), or None if the file can't be parsed."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"  could not parse {midi_path.name}: {e}")
        return None

    rolls = [inst.get_piano_roll(fs=FS) for inst in pm.instruments if not inst.is_drum]
    if not rolls:
        return None

    # instruments can have different lengths -> pad to the longest, then sum
    length = max(r.shape[1] for r in rolls)
    pr = np.zeros((128, length), dtype=np.float32)
    for r in rolls:
        pr[:, : r.shape[1]] += r
    pr = np.clip(pr, 0, 127)

    # keep the 64-note range
    pr = pr[MIN_PITCH : MAX_PITCH + 1]

    # crop or zero-pad to a fixed number of time steps
    if pr.shape[1] >= TARGET_TIME_STEPS:
        pr = pr[:, :TARGET_TIME_STEPS]
    else:
        pad = TARGET_TIME_STEPS - pr.shape[1]
        pr = np.pad(pr, ((0, 0), (0, pad)))

    return pr.astype(np.uint8)


def main():
    df = pd.read_csv(FEATURES_CSV, usecols=["composer", "filename"])
    X = np.zeros((len(df), TARGET_NOTES, TARGET_TIME_STEPS), dtype=np.uint8)
    ok = np.ones(len(df), dtype=bool)

    for i, (composer, filename) in enumerate(zip(df["composer"], df["filename"])):
        roll = midi_to_pianoroll(MIDI_DIR / composer / filename)
        if roll is None:
            ok[i] = False
        else:
            X[i] = roll
        if (i + 1) % 25 == 0:
            print(f"processed {i + 1}/{len(df)}")

    if not ok.all():
        print(f"WARNING: {(~ok).sum()} files failed and are all-zero rolls")

    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        composer=df["composer"].to_numpy(dtype=str),
        filename=df["filename"].to_numpy(dtype=str),
    )
    nonzero = (X > 0).mean()
    print(f"saved {X.shape} to {OUTPUT_PATH}  (fraction of non-zero cells: {nonzero:.4f})")


if __name__ == "__main__":
    main()
