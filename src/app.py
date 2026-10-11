"""
Demo app:  streamlit run src/app.py
Upload a MIDI file or pick one from the dataset -> predicted composer + probabilities.
Uses the piano-roll CNN (works for any upload). If you pick a dataset file, it also shows the
FCNN prediction, since that model needs Person A's features (looked up from the CSV).
"""

import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch

from dl_common import FEATURES_CSV, MODELS_DIR, ROOT
from piano_roll_conversion import midi_to_pianoroll
from train_cnn import PianoRollCNN
from train_fcnn import FCNN

st.set_page_config(page_title="MIDI Composer Classifier", layout="centered")
st.title("MIDI Composer Classifier")


@st.cache_resource
def load_cnn():
    path = MODELS_DIR / "cnn_aug.pt"
    if not path.exists():
        path = MODELS_DIR / "cnn_noaug.pt"
    ck = torch.load(path, map_location="cpu")
    model = PianoRollCNN(len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck["classes"], path.stem


@st.cache_resource
def load_fcnn():
    if not (MODELS_DIR / "fcnn.pt").exists():
        return None
    ck = torch.load(MODELS_DIR / "fcnn.pt", map_location="cpu")
    model = FCNN(ck["n_in"], len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    prep = joblib.load(MODELS_DIR / "fcnn_prep.joblib")
    return model.eval(), ck["classes"], prep


def show_probs(classes, probs, title):
    st.subheader(title)
    st.write(f"**Predicted composer: {classes[int(np.argmax(probs))]}**  "
             f"({probs.max() * 100:.1f}%)")
    st.bar_chart(pd.Series(probs, index=classes).sort_values(ascending=False))


cnn, cnn_classes, cnn_name = load_cnn()
features = pd.read_csv(FEATURES_CSV)

source = st.radio("MIDI source", ["Pick from dataset", "Upload a file"], horizontal=True)
midi_path, row = None, None

if source == "Upload a file":
    up = st.file_uploader("MIDI file", type=["mid", "midi"])
    if up is not None:
        tmp = Path(tempfile.gettempdir()) / up.name
        tmp.write_bytes(up.getbuffer())
        midi_path = tmp
else:
    composer = st.selectbox("Composer folder", sorted(features["composer"].unique()))
    files = features[features["composer"] == composer]["filename"].tolist()
    fname = st.selectbox("File", files)
    midi_path = ROOT / "data" / "midi" / composer / fname
    row = features[(features["composer"] == composer) & (features["filename"] == fname)]
    st.caption("Note: if this piece was in the training set, the prediction is optimistic.")

if midi_path is not None:
    roll = midi_to_pianoroll(midi_path)
    if roll is None:
        st.error("Could not read notes from this MIDI file.")
    else:
        st.image(np.flipud(roll).astype(np.float32) / 127.0, caption="Piano roll (64 notes x 1536 steps)",
                 use_container_width=True, clamp=True)
        x = torch.from_numpy(roll.astype(np.float32) / 127.0)[None, None]
        with torch.no_grad():
            probs = torch.softmax(cnn(x), dim=1)[0].numpy()
        show_probs(cnn_classes, probs, f"CNN on piano roll ({cnn_name})")

        fc = load_fcnn()
        if fc is not None and row is not None:
            model, classes, bundle = fc
            feats = bundle["prep"].transform(row[bundle["columns"]])
            with torch.no_grad():
                p = torch.softmax(model(torch.tensor(feats, dtype=torch.float32)), 1)[0].numpy()
            show_probs(classes, p, "FCNN on 78 hand-crafted features")
