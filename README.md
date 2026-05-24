# Voice Clarity Playground

Record a short voice clip in the browser, then compare how different speech-enhancement methods clean it up. Processing runs locally in the app—no audio is sent to an external API.

## Run locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Optional: run a quick smoke test of every installed processor:

```bash
python test_algos.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (include `models/DeepFilterNet3/` so DeepFilterNet can run without downloads).
2. Open [share.streamlit.io](https://share.streamlit.io), connect the repo, and set the main file to `app.py`.
3. Community Cloud reads `requirements.txt` for Python packages and `packages.txt` for system libraries (`libsndfile1`, Rust toolchain for `deepfilterlib`).

The first deploy may take several minutes while PyTorch and DeepFilterNet build. If the DL stack fails to install, the app still runs—the DSP and ML methods work without DeepFilterNet.

## Layout

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit UI and enhancement algorithms |
| `requirements.txt` | Python dependencies (CPU PyTorch) |
| `packages.txt` | Debian packages for Community Cloud |
| `models/DeepFilterNet3/` | Bundled DeepFilterNet weights |
| `test_algos.py` | Processor smoke tests |

## Methods

Sixteen techniques across classical DSP, statistical/learning methods, and optional deep learning (DeepFilterNet). Pick a preset package or choose methods by category after recording.
