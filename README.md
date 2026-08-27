# 🎬 Bollywood Look-Alike AI

Upload a selfie and discover which Bollywood celebrity you visually resemble
most, powered by AI face embeddings.

> ⚠️ **For entertainment and portfolio demonstration only.** Similarity
> scores are a visual-similarity indicator produced by this application —
> not an identity probability, authentication result, or proof that the
> uploaded person is the celebrity shown.

---

## ✨ Features

- **Face detection** on uploaded selfies using InsightFace
- **Face embeddings** compared against a precomputed celebrity gallery
- **Top 3 visual matches** with similarity scores
- **Downloadable result card** (PNG) combining your photo and your best match
- Clean, dark-themed Streamlit UI

---

## 🧠 How It Works

```
User Photo
    ↓
Face Detection
    ↓
Face Quality Check
    ↓
Face Embedding
    ↓
Compare Against Celebrity Gallery
    ↓
Top 3 Visual Matches
    ↓
Result Card
```

Each celebrity can have multiple reference photos. Their individual face
embeddings are averaged into a single, more stable representation. When a
user uploads a selfie, the same embedding process runs on their face and is
compared (via cosine similarity) against every celebrity in the gallery.

---

## 📁 Project Structure

```
.
├── app.py                      # Main Streamlit application
├── build_embeddings.py         # One-time / offline script to build the
│                                # celebrity embedding database
├── requirements.txt
├── utils/
│   ├── face_detection.py       # InsightFace loading, detection, cropping
│   ├── embeddings.py           # Embedding extraction, normalization, averaging
│   └── similarity.py           # Cosine similarity + ranking + scoring
├── celebrity_images/           # Your celebrity reference photos (you provide this)
│   ├── deepika_padukone/
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   └── shah_rukh_khan/
│       └── ...
└── models/
    └── celebrity_embeddings.pkl  # Generated automatically — do not edit by hand
```

### Adding celebrities

Create one folder per celebrity inside `celebrity_images/`, named with
underscores (it's auto-formatted for display, e.g. `deepika_padukone` →
"Deepika Padukone"). Add a handful of clear, varied reference photos
(`.jpg`, `.jpeg`, `.png`, or `.webp`) inside each folder.

> 💡 **Tip:** 8–12 varied, clear photos per celebrity give results nearly
> identical to 30+ photos, at a fraction of the processing time.

---

## 🚀 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add celebrity reference photos

Populate `celebrity_images/<celebrity_name>/` as described above. You need
**at least 2 celebrities** for the app to run.

### 3. Build the embedding database (recommended, do this first)

```bash
python build_embeddings.py
```

This processes every celebrity photo once, using all available CPU cores in
parallel, and saves the result to `models/celebrity_embeddings.pkl`. Doing
this ahead of time means the Streamlit app itself starts instantly instead
of rebuilding the gallery on first load.

> If you're deploying to a hosting platform with limited CPU (e.g. many
> free tiers only offer 1 core), run this step on your own machine first,
> then upload the generated `models/celebrity_embeddings.pkl` alongside
> your app — the deployed app only needs to *load* that file, not build it.

### 4. Run the app

```bash
streamlit run app.py
```

If `models/celebrity_embeddings.pkl` doesn't exist yet, `app.py` will build
it automatically on first run (with a live progress bar) — but running
`build_embeddings.py` ahead of time is faster and recommended.

---

## 🖥️ Usage

1. Upload a clear selfie (one visible face, good lighting, facing the
   camera).
2. Click **✨ Find My Bollywood Look-Alike**.
3. View your top 3 matches with similarity scores.
4. Download a shareable result card as a PNG.

---

## ⚙️ Configuration

A few constants you may want to tune, in `app.py` / `build_embeddings.py`:

| Setting | File | Default | Notes |
|---|---|---|---|
| `TOP_K` | `app.py` | `3` | Number of matches shown |
| `IMAGE_FOLDER` | both | `celebrity_images` | Source photo folder |
| `EMBEDDING_FILE` / `OUTPUT_FILE` | both | `models/celebrity_embeddings.pkl` | Saved database path |
| `NUM_WORKERS` | `build_embeddings.py` | `None` (auto) | Parallel worker processes for building embeddings |

Model choice, in `utils/face_detection.py`:

| Model | Size | Speed | Accuracy |
|---|---|---|---|
| `buffalo_l` | 326MB | Slowest | Highest |
| `buffalo_s` *(current default)* | 159MB | Faster | Slightly lower |
| `buffalo_sc` | 16MB | Fastest | Lower, no landmarks/age/gender |

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — web UI
- [InsightFace](https://github.com/deepinsight/insightface) — face detection & embeddings
- [ONNX Runtime](https://onnxruntime.ai/) — model inference
- [Pillow](https://python-pillow.org/) — image processing
- [NumPy](https://numpy.org/) — numerical operations

---

## ❗ Limitations & Disclaimer

- This is a **visual similarity** tool, not a facial-recognition or
  identity-verification system.
- Match quality depends on photo clarity, lighting, and the diversity of
  each celebrity's reference photos.
- Similarity scores are normalized for readability and are **not** raw
  probabilities.

---

## 👩‍💻 Credits

**Developed by Shrusti Diggavi**
IPEC Solutions Private Limited, Bangalore
