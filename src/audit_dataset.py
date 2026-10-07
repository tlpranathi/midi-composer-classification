from pathlib import Path
import hashlib
import pandas as pd
import pretty_midi

# locate project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIDI_DIR = PROJECT_ROOT / "data" / "midi"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []
hashes = {}

# check whether the MIDI directory exists
if not MIDI_DIR.exists():
    raise FileNotFoundError(f"MIDI directory not found: {MIDI_DIR}\n" "Place your composer folders inside data/midi/.")

# scan each composer folder
for composer_dir in sorted(MIDI_DIR.iterdir()):
    if not composer_dir.is_dir():
        continue

    composer = composer_dir.name

    for midi_path in sorted(composer_dir.rglob("*")):
        if midi_path.suffix.lower() not in {".mid", ".midi"}:
            continue

        relative_path = str(midi_path.relative_to(PROJECT_ROOT))

        record = {
            "composer": composer,
            "filename": midi_path.name,
            "path": relative_path,
            "valid": False,
            "notes": 0,
            "duration_seconds": 0.0,
            "duplicate_of": "",
            "error": "",
        }

        try:
            # detect exact duplicates by file contents
            file_hash = hashlib.sha256(midi_path.read_bytes()).hexdigest()

            if file_hash in hashes:
                record["duplicate_of"] = hashes[file_hash]
            else:
                hashes[file_hash] = relative_path

            # parse the MIDI file
            midi = pretty_midi.PrettyMIDI(str(midi_path))

            note_count = sum(len(instrument.notes) for instrument in midi.instruments)

            record["valid"] = True
            record["notes"] = note_count
            record["duration_seconds"] = round(midi.get_end_time(), 2)

            if note_count == 0:
                record["error"] = "No notes found"

        except Exception as exc:
            record["error"] = str(exc)

        records.append(record)

# save audit results
df = pd.DataFrame(records)

if df.empty:
    print("No MIDI files found. Check data/midi/.")
else:
    output_path = OUTPUT_DIR / "dataset_audit.csv"
    df.to_csv(output_path, index=False)

    usable = df[df["valid"] & (df["notes"] > 0)]

    print("\nFiles per composer:")
    print(df.groupby("composer").size().to_string())

    print("\nValid files containing notes:")
    print(usable.groupby("composer").size().to_string())

    print("\nSummary:")
    print("Total MIDI files:", len(df))
    print("Valid MIDI files:", int(df["valid"].sum()))
    print("Files containing notes:", len(usable))
    print("Exact duplicate files:", int(df["duplicate_of"].ne("").sum()))
    print("Files with parsing errors:", int((~df["valid"]).sum()))

    print("\nAudit saved to:", output_path)