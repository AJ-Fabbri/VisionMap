"""
app.py — Streamlit app for exploring a folder of photos with AI.
Run: streamlit run app.py
"""
import base64
import io
import json
import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import torch
import open_clip
from PIL import Image

DATA_DIR = "data"
RESULTS_PATH = os.path.join(DATA_DIR, "results.pkl")

COLOR_PALETTES = {
    "terrain": px.colors.qualitative.Bold,
    "weather": px.colors.qualitative.Pastel,
    "activity": px.colors.qualitative.Vivid,
    "content": px.colors.qualitative.Safe,
    "cluster": px.colors.qualitative.Light24,
}

st.set_page_config(
    page_title="VisionMap",
    layout="wide",
)


# Data loading

@st.cache_resource
def load_data():
    with open(RESULTS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["df"], np.array(data["embeddings"])


@st.cache_resource
def load_clip_model():
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    return model, tokenizer


@st.cache_data
def generate_thumbnails(paths: list, size: int = 80) -> list:
    thumbs = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((size, size))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=65)
            thumbs.append(base64.b64encode(buf.getvalue()).decode())
        except Exception:
            thumbs.append("")
    return thumbs


# Helpers

def tags_match(tags_series: pd.Series, selections: list) -> pd.Series:
    """Return boolean mask: row matches if any selected label appears in its tags string."""
    if not selections:
        return pd.Series([True] * len(tags_series), index=tags_series.index)
    pattern = "|".join(selections)
    return tags_series.str.contains(pattern, na=False)


def semantic_search(query: str, model, tokenizer, embeddings: np.ndarray, df: pd.DataFrame, top_k: int = 24):
    texts = tokenizer([query])
    with torch.no_grad():
        text_feat = model.encode_text(texts)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    sims = (embeddings @ text_feat.cpu().numpy().T).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return df.iloc[top_idx].copy(), sims[top_idx]


def photo_grid(subset: pd.DataFrame, scores=None, cols: int = 5, max_photos: int = 25):
    rows = [subset.iloc[i : i + cols] for i in range(0, min(len(subset), max_photos), cols)]
    for row_idx, row_df in enumerate(rows):
        grid_cols = st.columns(cols)
        for col_idx, (_, photo) in enumerate(row_df.iterrows()):
            with grid_cols[col_idx]:
                score_str = f" · {scores[row_idx * cols + col_idx]:.2f}" if scores is not None else ""
                caption = f"{photo['terrain_tags']} · {photo['weather_tags']}{score_str}"
                st.image(photo["path"], caption=caption, width="stretch")


# App

st.title("VisionMap")
st.caption("CLIP-powered visual AI analysis of your photo collection")

if not os.path.exists(RESULTS_PATH):
    st.error("**No embeddings found.** Run the pipeline first:")
    st.code(".venv/bin/python pipeline.py", language="bash")
    st.stop()

df, embeddings = load_data()

# Sidebar — filters use tags columns so multi-label photos are matched correctly

def unique_tags(col: pd.Series) -> list:
    """Collect all unique individual labels from a comma-separated tags column."""
    seen = set()
    for cell in col.dropna():
        for tag in cell.split(", "):
            seen.add(tag.strip())
    return sorted(seen)

with st.sidebar:
    st.header("Filters")
    sel_terrain = st.multiselect("Terrain", unique_tags(df["terrain_tags"]), default=[])
    sel_weather = st.multiselect("Weather", unique_tags(df["weather_tags"]), default=[])
    sel_activity = st.multiselect("Activity", unique_tags(df["activity_tags"]), default=[])
    sel_content = st.multiselect("Content type", unique_tags(df["content_tags"]), default=[])
    st.metric("Total photos", len(df))
    st.metric("Terrain types", df["terrain"].nunique())

mask = (
    tags_match(df["terrain_tags"], sel_terrain)
    & tags_match(df["weather_tags"], sel_weather)
    & tags_match(df["activity_tags"], sel_activity)
    & tags_match(df["content_tags"], sel_content)
)

filtered_df = df[mask].copy()

# Search bar (top-level)

query = st.text_input(
    "Semantic search — describe a photo in natural language",
    placeholder="e.g.  snowy mountain trail   ·   sunset on the coast   ·   group celebrating at the finish",
)

# Tabs

tab_map, tab_stats, tab_search = st.tabs(["Explore Map", "Stats", "Search Results"])

# Tab: Map

with tab_map:
    col_left, col_right = st.columns([3, 1])
    with col_right:
        color_options = ["terrain", "weather", "activity", "content"]
        if "cluster" in df.columns:
            color_options.append("cluster")
        color_by = st.selectbox("Color by", color_options, index=0)

    subtab_2d, subtab_3d = st.tabs(["2D", "3D"])

    with subtab_2d:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color=color_by,
            color_discrete_sequence=COLOR_PALETTES[color_by],
            hover_data={
                "filename": False,
                "terrain_tags": True,
                "weather_tags": True,
                "activity_tags": True,
                "content": True,
                "x": False,
                "y": False,
            },
            opacity=0.65,
            title=f"{len(df)} photos — colored by {color_by}",
        )
        fig.update_traces(marker_size=6)
        fig.update_layout(
            height=560,
            xaxis_title="",
            yaxis_title="",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            legend_title_text=color_by.capitalize(),
            plot_bgcolor="rgba(240,240,240,0.3)",
        )

        if mask.sum() < len(df):
            fig.add_scatter(
                x=filtered_df["x"],
                y=filtered_df["y"],
                mode="markers",
                marker=dict(size=10, symbol="circle-open", color="black", line_width=2),
                name="Filtered",
                hoverinfo="skip",
            )

        selection = st.plotly_chart(fig, on_select="rerun", key="umap_chart_2d", width="stretch")

        point_indices = (selection or {}).get("selection", {}).get("point_indices", [])
        if point_indices:
            st.subheader(f"Clicked: {len(point_indices)} photo(s)")
            selected_df = df.iloc[point_indices]
            photo_grid(selected_df, cols=min(len(point_indices), 5))
        else:
            count = len(filtered_df)
            label = " (filtered)" if mask.sum() < len(df) else ""
            st.subheader(f"Showing sample of {min(20, count)} / {count} photos{label}")
            sample = filtered_df.sample(min(20, count), random_state=42)
            photo_grid(sample)

    with subtab_3d:
        df_plot = df.reset_index(drop=True)
        df_plot["_idx"] = df_plot.index
        fig3d = px.scatter_3d(
            df_plot,
            x="x3d",
            y="y3d",
            z="z3d",
            color=color_by,
            color_discrete_sequence=COLOR_PALETTES[color_by],
            custom_data=["_idx"],
            hover_data={
                "terrain_tags": True,
                "weather_tags": True,
                "activity_tags": True,
                "content": True,
                "x3d": False,
                "y3d": False,
                "z3d": False,
                "_idx": False,
            },
            opacity=0.7,
            title=f"{len(df)} photos in 3D — colored by {color_by}",
        )
        fig3d.update_traces(marker_size=3)
        fig3d.update_layout(
            height=650,
            legend_title_text=color_by.capitalize(),
            scene=dict(
                xaxis=dict(showticklabels=False, title=""),
                yaxis=dict(showticklabels=False, title=""),
                zaxis=dict(showticklabels=False, title=""),
            ),
        )

        if mask.sum() < len(df):
            fig3d.add_trace(go.Scatter3d(
                x=filtered_df["x3d"],
                y=filtered_df["y3d"],
                z=filtered_df["z3d"],
                mode="markers",
                marker=dict(size=6, symbol="circle-open", color="black", line_width=2),
                name="Filtered",
                hoverinfo="skip",
            ))

        st.plotly_chart(fig3d, width="stretch", key="umap_chart_3d")
        st.caption("Drag to rotate · scroll to zoom · click a point for a preview")

        thumbs = generate_thumbnails(df["path"].tolist(), size=120)
        labels = (df["terrain_tags"] + " · " + df["weather_tags"]).tolist()
        thumbs_json = json.dumps(thumbs)
        labels_json = json.dumps(labels)

        # Single JS component: smooth spinner + click-to-thumbnail overlay
        components.html(f"""
<style>
  body {{ margin: 0; font-family: sans-serif; }}
  button {{
    padding: 5px 16px; margin: 0 4px; cursor: pointer;
    border: 1px solid #ccc; border-radius: 4px;
    background: white; font-size: 13px;
  }}
  button:hover {{ background: #f0f0f0; }}
  button.active {{ background: #1f77b4; color: white; border-color: #1f77b4; }}
  #status {{ display:inline; margin-left:10px; font-size:11px; color:#888; }}
</style>
<button id="btn-spin">Spin</button>
<button id="btn-stop">Stop</button>
<span id="status">searching for chart…</span>
<script>
(function () {{
  const THUMBS = {thumbs_json};
  const LABELS = {labels_json};
  const R = 1.8, Z = 0.3, SPEED = 0.0036;
  let angle = 0, animating = false, rafId = null;
  const status = document.getElementById('status');

  // Detect the 3D chart by checking trace types
  function getPlot() {{
    const plots = window.parent.document.querySelectorAll('.js-plotly-plot');
    for (const p of plots) {{
      try {{
        if (p.data && Array.from(p.data).some(t => t.type === 'scatter3d')) return p;
      }} catch (_) {{}}
    }}
    return null;
  }}

  // Track cursor in parent so we know where to place the overlay
  let mouseX = 0, mouseY = 0;
  window.parent.document.addEventListener('mousemove', e => {{ mouseX = e.clientX; mouseY = e.clientY; }});

  // --- Spinner ---
  function tick() {{
    if (!animating) return;
    angle += SPEED;
    const plot = getPlot();
    if (plot) {{
      window.parent.Plotly.relayout(plot, {{
        'scene.camera': {{
          eye: {{ x: R * Math.cos(angle), y: R * Math.sin(angle), z: Z }},
          center: {{ x: 0, y: 0, z: 0 }},
          up: {{ x: 0, y: 0, z: 1 }},
        }}
      }});
    }}
    rafId = requestAnimationFrame(tick);
  }}

  document.getElementById('btn-spin').addEventListener('click', function () {{
    if (animating) return;
    animating = true;
    this.classList.add('active');
    document.getElementById('btn-stop').classList.remove('active');
    tick();
  }});

  document.getElementById('btn-stop').addEventListener('click', function () {{
    animating = false;
    if (rafId) cancelAnimationFrame(rafId);
    this.classList.add('active');
    document.getElementById('btn-spin').classList.remove('active');
  }});

  // --- Floating thumbnail overlay (created once in parent doc) ---
  let overlay = window.parent.document.getElementById('thumb-overlay-3d');
  if (!overlay) {{
    overlay = window.parent.document.createElement('div');
    overlay.id = 'thumb-overlay-3d';
    overlay.style.cssText = [
      'position:fixed', 'z-index:9999', 'background:white',
      'border:1px solid #ddd', 'border-radius:4px', 'padding:3px',
      'box-shadow:0 2px 8px rgba(0,0,0,0.18)', 'display:none',
      'pointer-events:none', 'max-width:115px', 'text-align:center',
      'font-size:10px', 'color:#666', 'line-height:1.3'
    ].join(';');
    window.parent.document.body.appendChild(overlay);
  }}

  function showThumb(idx) {{
    const b64 = THUMBS[idx];
    if (!b64) return;
    overlay.innerHTML =
      '<img src="data:image/jpeg;base64,' + b64 +
      '" style="width:108px;height:auto;border-radius:2px;display:block;margin-bottom:2px">' +
      (LABELS[idx] || '');
    overlay.style.left = (mouseX + 12) + 'px';
    overlay.style.top  = Math.max(10, mouseY - 90) + 'px';
    overlay.style.display = 'block';
    clearTimeout(overlay._t);
    // dismiss on the next mousedown after this interaction completes (mouseup)
    window.parent.document.addEventListener('mouseup', function waitForRelease() {{
      window.parent.document.removeEventListener('mouseup', waitForRelease);
      window.parent.document.addEventListener('mousedown', () => {{ overlay.style.display = 'none'; }}, {{ once: true }});
    }}, {{ once: true }});
  }}

  function onPlotClick(data) {{
    if (!data || !data.points || !data.points.length) return;
    const pt = data.points[0];
    const idx = (pt.customdata && pt.customdata[0] != null) ? pt.customdata[0] : pt.pointNumber;
    status.textContent = 'point ' + idx;
    showThumb(idx);
  }}

  // Poll continuously so we re-attach after Streamlit replaces the chart on re-render
  let knownPlot = null;
  function pollForPlot() {{
    const plot = getPlot();
    if (plot && plot !== knownPlot) {{
      knownPlot = plot;
      try {{ plot.removeAllListeners('plotly_click'); }} catch(_) {{}}
      plot.on('plotly_click', onPlotClick);
      status.textContent = 'ready — click a point';
    }}
    setTimeout(pollForPlot, 800);
  }}
  pollForPlot();
}})();
</script>
""", height=40)


# Tab: Stats

with tab_stats:
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        fig_t = px.pie(df, names="terrain", title="Terrain (primary)", color_discrete_sequence=px.colors.qualitative.Bold)
        fig_t.update_traces(textposition="inside", textinfo="percent+label")
        fig_t.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_t, width="stretch")

    with c2:
        fig_w = px.pie(df, names="weather", title="Weather (primary)", color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_w.update_traces(textposition="inside", textinfo="percent+label")
        fig_w.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_w, width="stretch")

    with c3:
        fig_a = px.pie(df, names="activity", title="Activity type", color_discrete_sequence=px.colors.qualitative.Vivid)
        fig_a.update_traces(textposition="inside", textinfo="percent+label")
        fig_a.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_a, width="stretch")

    with c4:
        fig_c = px.pie(df, names="content", title="Content type", color_discrete_sequence=px.colors.qualitative.Safe)
        fig_c.update_traces(textposition="inside", textinfo="percent+label")
        fig_c.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_c, width="stretch")

    st.subheader("Cross-tab: terrain × activity")
    cross = pd.crosstab(df["terrain"], df["activity"])
    fig_heat = px.imshow(
        cross,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Photo count by terrain + activity type",
        aspect="auto",
    )
    fig_heat.update_layout(height=400)
    st.plotly_chart(fig_heat, width="stretch")


# Tab: Search

with tab_search:
    if query.strip():
        with st.spinner("Searching…"):
            model, tokenizer = load_clip_model()
            results_df, scores = semantic_search(query, model, tokenizer, embeddings, df)
        st.success(f"Top {len(results_df)} results for: **{query}**")
        photo_grid(results_df, scores=scores, cols=5, max_photos=25)
    else:
        st.info("Enter a description in the search bar above — try 'foggy mountain trail' or 'sunny beach'.")
