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
