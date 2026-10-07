from pathlib import Path
import numpy as np
import pandas as pd
import pretty_midi

# project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MIDI_DIR = PROJECT_ROOT / "data" / "midi"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []

# major and natural-minor scale templates, represented as pitch classes
MAJOR_TEMPLATE = {0, 2, 4, 5, 7, 9, 11}
MINOR_TEMPLATE = {0, 2, 3, 5, 7, 8, 10}


def calculate_chromatic_ratio(pitches):
    """estimate the proportion of notes outside the best-fitting major/minor scale."""
    pitch_classes = pitches % 12
    counts = np.bincount(pitch_classes, minlength=12)

    best_in_scale_count = 0

    for root in range(12):
        major_scale = {(root + note) % 12 for note in MAJOR_TEMPLATE}
        minor_scale = {(root + note) % 12 for note in MINOR_TEMPLATE}
        best_in_scale_count = max(best_in_scale_count, sum(counts[pitch_class] for pitch_class in major_scale), sum(counts[pitch_class] for pitch_class in minor_scale),)

    return float(1 - best_in_scale_count / len(pitches))


def calculate_polyphony_ratio(notes):
    """proportion of notes that overlap at least one other note."""
    if not notes:
        return 0.0

    # a note overlaps another if another note starts before it ends and ends after it starts.
    sorted_notes = sorted(notes, key=lambda note: note.start)
    overlapping_count = 0

    for i, note in enumerate(sorted_notes):
        for other in sorted_notes:
            if other.start >= note.end:
                break

            if other is not note and other.end > note.start:
                overlapping_count += 1
                break

    return float(overlapping_count / len(notes))


def calculate_vertical_interval_histogram(notes):
    """histogram of pitch intervals between overlapping note pairs."""

    interval_counts = np.zeros(25, dtype=float)
    total_pairs = 0

    for i, note_a in enumerate(notes):
        for note_b in notes[i + 1:]:
            # notes must overlap in time
            if note_a.start < note_b.end and note_b.start < note_a.end:
                # calculate signed pitch interval
                interval = note_b.pitch - note_a.pitch

                # skip unisons
                if interval == 0:
                    continue

                # limit intervals to -12 ... +12
                interval = int(np.clip(interval, -12, 12))

                interval_counts[interval + 12] += 1
                total_pairs += 1

    if total_pairs > 0:
        interval_counts /= total_pairs

    return interval_counts


# process each composer folder
for composer_dir in sorted(MIDI_DIR.iterdir()):
    if not composer_dir.is_dir():
        continue

    composer = composer_dir.name

    for midi_path in sorted(composer_dir.iterdir()):
        if midi_path.suffix.lower() not in {".mid", ".midi"}:
            continue

        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))

            # collect notes from all instruments
            notes = [note for instrument in midi.instruments for note in instrument.notes]

            if not notes:
                print(f"Skipping file with no notes: {midi_path.name}")
                continue

            pitches = np.array([note.pitch for note in notes])
            durations = np.array([note.end - note.start for note in notes])
            vertical_interval_histogram = (calculate_vertical_interval_histogram(notes))

            # sort notes by start time
            notes_by_time = sorted(notes, key=lambda note: note.start)
            ordered_pitches = np.array([note.pitch for note in notes_by_time])

            # existing interval statistics
            intervals = np.diff(ordered_pitches)

            # new melodic-interval histogram: 25 bins for intervals -12 to +12.
            # values below -12 or above +12 are grouped into the edge bins.
            if len(intervals) > 0:
                clipped_intervals = np.clip(intervals, -12, 12)

                interval_counts = np.bincount((clipped_intervals + 12).astype(int),minlength=25,)

                melodic_interval_histogram = (interval_counts / len(intervals))
            else:
                melodic_interval_histogram = np.zeros(25)

            # duration of the complete MIDI piece
            duration_seconds = midi.get_end_time()

            # pitch-class distribution
            pitch_classes = pitches % 12
            pitch_class_counts = np.bincount(pitch_classes, minlength=12)
            pitch_class_distribution = (pitch_class_counts / len(pitches))

            # note density
            note_density = (len(notes) / duration_seconds if duration_seconds > 0 else 0.0)

            # pitch-class entropy
            nonzero_probs = pitch_class_distribution[pitch_class_distribution > 0]
            pitch_class_entropy = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))

            # build feature row: existing features retained
            record = {
                "composer": composer,
                "filename": midi_path.name,

                # existing general features
                "mean_pitch": float(np.mean(pitches)),
                "std_pitch": float(np.std(pitches)),
                "mean_note_duration": float(np.mean(durations)),
                "std_note_duration": float(np.std(durations)),
                "note_density": float(note_density),
                "mean_interval": (float(np.mean(intervals)) if len(intervals) > 0 else 0.0),
                "std_interval": (float(np.std(intervals)) if len(intervals) > 0 else 0.0),
                "instrument_count": len(midi.instruments),

                # new pitch and duration features
                "min_pitch": float(np.min(pitches)),
                "max_pitch": float(np.max(pitches)),
                "pitch_range": float(np.max(pitches) - np.min(pitches)),
                "median_pitch": float(np.median(pitches)),
                "median_note_duration": float(np.median(durations)),

                # new tonal and overlap features
                "pitch_class_entropy": pitch_class_entropy,
                "chromatic_note_ratio": calculate_chromatic_ratio(pitches),
                "polyphony_ratio": calculate_polyphony_ratio(notes),
            }

            # add 12 pitch-class features
            for i in range(12):
                record[f"pitch_class_{i}"] = float(pitch_class_distribution[i])
            # add 25 melodic-interval histogram features
            for i in range(25):
                interval_value = i - 12
                record[f"melodic_interval_{interval_value}"] = float(melodic_interval_histogram[i])

            # Add 25 vertical-interval histogram features
            for i in range(25):
                interval_value = i - 12
                record[f"vertical_interval_{interval_value}"] = float(vertical_interval_histogram[i])

            records.append(record)

        except Exception as exc:
            print(f"Error processing {midi_path}: {exc}")


# save expanded features separately
df = pd.DataFrame(records)

if df.empty:
    print("No features extracted. Check your MIDI files.")
else:
    output_path = OUTPUT_DIR / "midi_features_vertical.csv"
    df.to_csv(output_path, index=False)

    print("\nFeature extraction complete!")
    print("Pieces processed:", len(df))
    print("Number of composers:", df["composer"].nunique())
    print("Number of numerical features:", df.shape[1] - 2)
    print("Saved to:", output_path)

    print("\nPieces per composer:")
    print(df["composer"].value_counts().sort_index().to_string())

    print("\nNew features:")
    new_features = ["min_pitch", "max_pitch", "pitch_range", "median_pitch", "median_note_duration", "pitch_class_entropy", "chromatic_note_ratio", "polyphony_ratio",]
    print(df[new_features].describe().round(3).to_string())