"""
cluster.py — run once after pipeline.py to add emergent cluster labels to results.pkl.

Steps:
  1. HDBSCAN on UMAP 2D coords derived from 512-dim CLIP embeddings
  2. Auto-name each cluster by finding nearest ImageNet-21k labels to the
     cluster centroid in CLIP's joint text-image embedding space
  3. Saves cluster / cluster_id columns back into data/results.pkl

Usage:
  python cluster.py
"""
import pickle
import numpy as np
import torch
import open_clip
import hdbscan
from timm.data.imagenet_info import ImageNetInfo
from tqdm import tqdm

DATA_PATH = "data/results.pkl"
MIN_CLUSTER_SIZE = 10
TOP_LABELS = 3          # how many ImageNet labels to join into the cluster name
VOCAB_BATCH = 512       # text encoding batch size


def load_data():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def load_clip():
    print("Loading CLIP ViT-B/32…")
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, tokenizer


def encode_vocab(model, tokenizer, labels: list[str]) -> np.ndarray:
    """Encode all vocab labels in batches, return L2-normalised embeddings."""
    all_feats = []
    for i in tqdm(range(0, len(labels), VOCAB_BATCH), desc="Encoding vocab"):
        batch = labels[i : i + VOCAB_BATCH]
        # Use only the first synonym (before the first comma) to keep prompts short
        short = [l.split(",")[0].strip() for l in batch]
        tokens = tokenizer(short)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().numpy())
    return np.vstack(all_feats)


def name_cluster(centroid: np.ndarray, vocab_embs: np.ndarray, labels: list[str], top_k: int = TOP_LABELS) -> str:
    sims = vocab_embs @ centroid
    top_idx = sims.argsort()[::-1][:top_k]
    # Use only the first synonym of each matched label
    parts = [labels[i].split(",")[0].strip() for i in top_idx]
    return " / ".join(parts)


def main():
    data = load_data()
    df = data["df"]
    embeddings = np.array(data["embeddings"])  # (N, 512), already L2-normed

    # --- HDBSCAN on UMAP 2D coords ---
    # Running on raw 512-dim embeddings fails due to the curse of dimensionality
    # (all points equidistant). UMAP coords already encode the cluster structure
    # visually, so HDBSCAN finds exactly what the eye sees.
    umap_coords = df[["x", "y"]].values
    print(f"Running HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE}) on UMAP 2D coords…")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="leaf",
    )
    labels = clusterer.fit_predict(umap_coords)
    n_clusters = int(labels.max()) + 1
    n_noise = int((labels == -1).sum())
    print(f"Found {n_clusters} clusters, {n_noise} noise points (→ 'Uncategorized')")

    # --- ImageNet-21k vocabulary ---
    print("Loading ImageNet-21k descriptions from timm…")
    info = ImageNetInfo("imagenet21k")
    vocab = info.label_descriptions()
    # Filter out very abstract / non-visual synsets (short 1-word entries that are
    # too broad, e.g. "organism, being") — keep those with at least 3 chars after split
    vocab = [v for v in vocab if len(v.split(",")[0].strip()) >= 3]
    print(f"Vocabulary: {len(vocab)} synsets")

    model, tokenizer = load_clip()
    vocab_embs = encode_vocab(model, tokenizer, vocab)  # (V, 512)

    # --- Name each cluster ---
    print("Naming clusters…")
    cluster_names_map = {-1: "Uncategorized"}
    for cid in range(n_clusters):
        mask = labels == cid
        centroid = embeddings[mask].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        cluster_names_map[cid] = name_cluster(centroid, vocab_embs, vocab)
        print(f"  Cluster {cid:3d} ({mask.sum():4d} photos): {cluster_names_map[cid]}")

    # --- Write back ---
    df["cluster_id"] = labels
    df["cluster"] = df["cluster_id"].map(cluster_names_map)

    data["df"] = df
    with open(DATA_PATH, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSaved updated results to {DATA_PATH}")
    print(f"Added columns: cluster_id, cluster")


if __name__ == "__main__":
    main()
