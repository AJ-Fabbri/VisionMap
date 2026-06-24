"""
pipeline.py — run once to embed all photos in MEDIA_DIR, classify them, and compute UMAP coords.
Output: data/results.pkl

Edit MEDIA_DIR below to point at any folder of .jpg files on disk.
Edit the label lists to customize classification categories for your collection.
"""
import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch
import open_clip
from tqdm import tqdm
import umap

MEDIA_DIR = "media"
DATA_DIR = "data"
BATCH_SIZE = 32

TERRAIN_LABELS = [
    "high alpine mountain rocky summit peak",
    "dense forest woodland trees canopy",
    "urban city street road buildings pavement",
    "coastal ocean sea beach shoreline",
    "desert dry arid sandy scrubland",
    "open rolling hills countryside farmland fields",
    "snow and ice covered winter landscape",
    "river lake pond waterfall",
]

WEATHER_LABELS = [
    "sunny clear blue sky bright daylight",
    "cloudy overcast gray dull sky",
    "foggy misty low visibility haze",
    "golden sunset or sunrise orange pink sky",
    "rainy wet glistening rain drops puddles",
]

ACTIVITY_LABELS = [
    "trail running athlete moving fast",
    "road cycling bicycle on paved road",
    "mountain biking off-road dirt single track rocky terrain",
    "hiking backpacking trekking on foot",
    "swimming open water",
    "skiing or snowboarding on slope",
    "group of people friends together outdoors",
    "scenic landscape no people",
]

CONTENT_LABELS = [
    "outdoor nature landscape or sport activity photo",
    "portrait or selfie of a person or group of people",
    "meme joke funny image text overlay screenshot",
    "food drink meal restaurant",
    "map route GPS track data visualization",
    "gear equipment clothing product flat lay",
]

SHORT_TERRAIN = ["Mountain", "Forest", "Urban", "Coastal", "Desert", "Countryside", "Snow", "Water"]
SHORT_WEATHER = ["Sunny", "Cloudy", "Foggy", "Sunset/Sunrise", "Rainy"]
SHORT_ACTIVITY = ["Trail Run", "Road Cycling", "MTB", "Hiking", "Swimming", "Skiing", "People", "Scenery"]
SHORT_CONTENT = ["Outdoor/Sport", "Portrait", "Meme/Text", "Food", "Map/Data", "Gear"]

# Secondary label included if its similarity is within this fraction of the top score
MULTI_LABEL_RATIO = 0.92


def load_model():
    print("Loading CLIP ViT-B/32 (openai)…")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, preprocess, tokenizer


def embed_images(model, preprocess, image_paths):
    all_embeddings = []
    valid_paths = []

    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Embedding images"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        tensors = []
        batch_valid = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
                batch_valid.append(p)
            except Exception as e:
                print(f"  Skipping {os.path.basename(p)}: {e}")

        if not tensors:
            continue

        batch = torch.stack(tensors)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        all_embeddings.extend(feats.cpu().numpy())
        valid_paths.extend(batch_valid)

    return np.array(all_embeddings), valid_paths


def zero_shot_classify(model, tokenizer, embeddings, label_groups):
    results = {}
    for group_name, (raw_labels, short_labels) in label_groups.items():
        texts = tokenizer(raw_labels)
        with torch.no_grad():
            text_feats = model.encode_text(texts)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        sims = embeddings @ text_feats.cpu().numpy().T  # (N, K)
        top_sims = sims.max(axis=1)
        pred_idx = sims.argmax(axis=1)

        primary = [short_labels[i] for i in pred_idx]
        # Multi-label: include any label within MULTI_LABEL_RATIO of the top score
        tags = [
            ", ".join(
                short_labels[j]
                for j in range(len(short_labels))
                if sims[i, j] >= top_sims[i] * MULTI_LABEL_RATIO
            )
            for i in range(len(embeddings))
        ]

        results[group_name] = primary
        results[f"{group_name}_tags"] = tags
        results[f"{group_name}_conf"] = top_sims.tolist()

    return results


def run_umap(embeddings, n_components=2):
    print(f"Running UMAP {n_components}D…")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        verbose=True,
    )
    return reducer.fit_transform(embeddings)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    jpg_paths = sorted(
        os.path.join(MEDIA_DIR, f)
        for f in os.listdir(MEDIA_DIR)
        if f.lower().endswith(".jpg")
    )
    print(f"Found {len(jpg_paths)} JPG images (skipping MP4s)")

    model, preprocess, tokenizer = load_model()

    embeddings, valid_paths = embed_images(model, preprocess, jpg_paths)
    print(f"Embedded {len(valid_paths)} images → shape {embeddings.shape}")

    label_groups = {
        "terrain": (TERRAIN_LABELS, SHORT_TERRAIN),
        "weather": (WEATHER_LABELS, SHORT_WEATHER),
        "activity": (ACTIVITY_LABELS, SHORT_ACTIVITY),
        "content": (CONTENT_LABELS, SHORT_CONTENT),
    }
    classifications = zero_shot_classify(model, tokenizer, embeddings, label_groups)

    coords2d = run_umap(embeddings, n_components=2)
    coords3d = run_umap(embeddings, n_components=3)

    df = pd.DataFrame(
        {
            "path": valid_paths,
            "filename": [os.path.basename(p) for p in valid_paths],
            "x": coords2d[:, 0],
            "y": coords2d[:, 1],
            "x3d": coords3d[:, 0],
            "y3d": coords3d[:, 1],
            "z3d": coords3d[:, 2],
            **classifications,
        }
    )

    output = {"df": df, "embeddings": embeddings}
    out_path = os.path.join(DATA_DIR, "results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved to {out_path}")
    print("\nLabel distribution:")
    for col in ["terrain", "weather", "activity", "content"]:
        print(f"\n  {col}:")
        print(df[col].value_counts().to_string(index=True))
