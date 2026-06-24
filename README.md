# VisionMap

A computer vision pipeline that analyzes and visualizes (in 3d) any folder of photos using CLIP embeddings.

## What it does

- **Zero-shot classification** - each photo is automatically labeled by terrain, weather, activity type, and content type using OpenAI's CLIP model. No training, no hand-labeling.
- **Emergent clustering** - HDBSCAN finds natural groupings in the UMAP layout and auto-names each cluster by projecting its centroid back into CLIP's text space against 21,843 ImageNet labels. Cluster names like "mountain bike / trail bike" or "celebration / party" emerge without any predefined categories.
- **Semantic search** - describe a scene in plain English ("foggy mountain trail", "group at a summit") and find matching photos instantly.
- **Interactive UMAP explorer** - all photos projected into 2D and 3D interactive scatter plots, where proximity = visual/semantic similarity. Color by terrain, weather, activity, content type, or emergent cluster. Click individual points in 3D for a photo thumbnail overlay; click points in 2D for a full photo grid below.
- **Stats dashboard** - breakdowns by terrain, weather, activity, and content type, plus a terrain × activity heatmap.

## How it works

CLIP (Contrastive Language-Image Pretraining) is trained on 400M image-text pairs to map images and text into the same 512-dimensional vector space. Classification is based on cosine similarity between an image embedding and a set of text label embeddings. Semantic search works the same way: embed the query string, dot-product against all image embeddings, return top-k.

UMAP reduces those 512D embeddings to 2D/3D by building a nearest-neighbor graph and optimizing a layout that preserves its topology. Photos that look or feel similar end up near each other.

HDBSCAN then runs on the UMAP 2D coordinates to find density-based clusters. Each cluster is named by computing its centroid in the original 512D embedding space and finding the nearest ImageNet-21k synset text embeddings. Because of this, the labels come from CLIP's geometry, not from anything hand-written.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Step 1** - point the pipeline at your photos. By default it reads from a `media/` folder in the project directory. Drop your `.jpg` files there (or edit `MEDIA_DIR` in `pipeline.py` to point anywhere on disk).

**Step 2** - embed, classify, and compute UMAP projections (run once):

```bash
python pipeline.py
```

Takes ~30 seconds per few hundred images on CPU. Saves results to `data/results.pkl`.

**Step 3** - discover emergent clusters and auto-name them (optional, run once after Step 2):

```bash
python cluster.py
```

Takes a few minutes (encodes 21k ImageNet labels with CLIP). Adds `cluster` and `cluster_id` columns to `data/results.pkl`.

**Step 4** - launch the app:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). If `cluster.py` has been run, a "cluster" option appears in the "Color by" dropdown on the map tabs.

## Project structure

```
VisionMap/
├── pipeline.py       # embed + classify + UMAP (run once)
├── cluster.py        # HDBSCAN + auto-naming via ImageNet-21k (run once after pipeline)
├── app.py            # Streamlit app
├── requirements.txt
├── media/            # your photos (.jpg)
└── data/
    └── results.pkl   # generated embeddings + labels (gitignored)
```

## Customizing labels

All classification prompts live at the top of `pipeline.py`. Edit `TERRAIN_LABELS`, `WEATHER_LABELS`, `ACTIVITY_LABELS`, or `CONTENT_LABELS` to suit your collection and re-run `pipeline.py` to regenerate. The prompts are natural language. More descriptive phrases generally produce better results because they move the text embedding closer to the visual cluster you're targeting.

`MULTI_LABEL_RATIO` (default `0.92`) controls how aggressively secondary labels are assigned. Lower values produce more multi-label tags; higher values make classification more exclusive.

## Tuning clusters

Cluster granularity is controlled in `cluster.py`:

- `MIN_CLUSTER_SIZE` (default `10`) - minimum number of photos to form a cluster; smaller groups become "Uncategorized"
- `min_samples` (default `5`) - how many neighbors a point needs to be considered a core point; lower values make the algorithm more sensitive to small dense regions
- `TOP_LABELS` (default `3`) - how many ImageNet synsets to join into each cluster's display name
