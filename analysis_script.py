import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.stats import kstest, expon, lognorm, powerlaw, norm, chi2
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from collections import Counter

with open("data/results.pkl", "rb") as f:
    data = pickle.load(f)

df = data["df"]
embeddings = np.array(data["embeddings"])

coords3d = df[["x3d","y3d","z3d"]].values
coords2d = df[["x","y"]].values

print(f"Dataset: {len(df)} photos")
print(f"Embedding shape: {embeddings.shape}")
print(f"\n--- 3D coordinate stats ---")
print(f"x3d: [{coords3d[:,0].min():.3f}, {coords3d[:,0].max():.3f}]  mean={coords3d[:,0].mean():.3f}")
print(f"y3d: [{coords3d[:,1].min():.3f}, {coords3d[:,1].max():.3f}]  mean={coords3d[:,1].mean():.3f}")
print(f"z3d: [{coords3d[:,2].min():.3f}, {coords3d[:,2].max():.3f}]  mean={coords3d[:,2].mean():.3f}")

# Center of mass
com = coords3d.mean(axis=0)
print(f"\nCenter of mass: {com}")

# Radius from center of mass
radii = np.linalg.norm(coords3d - com, axis=1)
print(f"\n--- Radii from center of mass ---")
print(f"  min: {radii.min():.3f}")
print(f"  max: {radii.max():.3f}")
print(f"  mean: {radii.mean():.3f}")
print(f"  std: {radii.std():.3f}")
print(f"  median: {np.median(radii):.3f}")

df = data["df"]
embeddings = np.array(data["embeddings"])
coords3d = df[["x3d","y3d","z3d"]].values
com = coords3d.mean(axis=0)
radii = np.linalg.norm(coords3d - com, axis=1)

print("=== 1. RADIAL DISTRIBUTION (shell density) ===")
bins = np.linspace(radii.min(), radii.max(), 10)
counts, edges = np.histogram(radii, bins=bins)
shell_vols = (4/3)*np.pi*(edges[1:]**3 - edges[:-1]**3)
density = counts / shell_vols
for i, (c, d) in enumerate(zip(counts, density)):
    r_mid = (edges[i]+edges[i+1])/2
    print(f"  r={r_mid:.2f}  count={c:3d}  density={d:.4f}")

print("\n=== 2. NEAREST-NEIGHBOR DISTANCES ===")
nbrs = NearestNeighbors(n_neighbors=6).fit(coords3d)
dists, _ = nbrs.kneighbors(coords3d)
nn1 = dists[:,1]  # skip self
print(f"  1-NN dist: mean={nn1.mean():.4f}  std={nn1.std():.4f}  min={nn1.min():.4f}  max={nn1.max():.4f}")
nn5 = dists[:,5]
print(f"  5-NN dist: mean={nn5.mean():.4f}  std={nn5.std():.4f}")

# Test if NN distances follow exponential (Poisson point process)
stat, p = kstest(nn1, 'expon', args=(nn1.mean(), nn1.std()))
print(f"  KS test vs exponential: stat={stat:.4f}  p={p:.4f}  ({'likely Poisson/random' if p>0.05 else 'NOT random/Poisson'})")

print("\n=== 3. DBSCAN CLUSTERS ===")
db = DBSCAN(eps=0.7, min_samples=5).fit(coords3d)
labels = db.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"  Clusters found: {n_clusters}  (noise pts: {n_noise} = {n_noise/len(labels)*100:.1f}%)")
cluster_sizes = Counter(labels[labels >= 0])
print(f"  Cluster sizes: {sorted(cluster_sizes.values(), reverse=True)}")

# What labels do clusters have?
for cid, size in sorted(cluster_sizes.items(), key=lambda x: -x[1]):
    mask = labels == cid
    terrain_dist = df[mask]["terrain"].value_counts().head(3).to_dict()
    weather_dist = df[mask]["weather"].value_counts().head(2).to_dict()
    print(f"  Cluster {cid} (n={size}): terrain={terrain_dist}  weather={weather_dist}")

print("\n=== 4. PRINCIPAL AXES (shape of cloud) ===")
centered = coords3d - com
cov = np.cov(centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Explained variance ratios: {eigenvalues/eigenvalues.sum()}")
print(f"  Axis ratios (a:b:c): 1 : {np.sqrt(eigenvalues[1]/eigenvalues[0]):.3f} : {np.sqrt(eigenvalues[2]/eigenvalues[0]):.3f}")
# Prolateness / oblateness: T = (a^2 - b^2) / (a^2 - c^2) where a>b>c
a2, b2, c2 = eigenvalues
T = (a2 - b2) / (a2 - c2)
print(f"  Triaxiality T = {T:.3f}  (0=oblate/pancake, 0.5=triaxial, 1=prolate/cigar)")

df = data["df"]
embeddings = np.array(data["embeddings"])
coords3d = df[["x3d","y3d","z3d"]].values
com = coords3d.mean(axis=0)
centered = coords3d - com
radii = np.linalg.norm(centered, axis=1)

print("=== 5. RADIAL DISTRIBUTION FIT ===")
# If data were drawn from a 3D Gaussian, radii follow a Maxwell-Boltzmann (chi with 3 dof)
# Test: chi2 with 3 dof on radii^2/sigma^2
sigma2 = np.var(radii)
stat_chi2, p_chi2 = kstest(radii**2 / sigma2 * 3, 'chi2', args=(3,))
print(f"  KS vs chi(3)/Maxwell-Boltzmann: stat={stat_chi2:.4f}  p={p_chi2:.4f}  ({'consistent' if p_chi2>0.05 else 'NOT consistent'})")

# Lognormal fit
shape, loc, scale = lognorm.fit(radii, floc=0)
stat_ln, p_ln = kstest(radii, 'lognorm', args=(shape, loc, scale))
print(f"  KS vs lognormal:               stat={stat_ln:.4f}  p={p_ln:.4f}  ({'consistent' if p_ln>0.05 else 'NOT consistent'})")

print("\n=== 6. LOCAL DENSITY VARIATION (fractal signature?) ===")
nbrs = NearestNeighbors(n_neighbors=20).fit(coords3d)
dists, _ = nbrs.kneighbors(coords3d)

# Local density proxy: inverse of distance to k-th neighbor
local_density = 1.0 / dists[:, -1]
print(f"  Local density: mean={local_density.mean():.3f}  std={local_density.std():.3f}")
print(f"  Density CV (std/mean): {local_density.std()/local_density.mean():.3f}")

# Log-log slope of count(r<R) vs R — if fractal: slope = fractal dimension
R_vals = np.percentile(dists[:,1:], range(10, 100, 10), axis=0).mean(axis=1)
counts = [(dists[:,1:] < r).sum() / len(coords3d) for r in R_vals]
log_r = np.log(R_vals)
log_c = np.log(counts)
slope = np.polyfit(log_r, log_c, 1)[0]
print(f"  Log-log slope (correlation dimension estimate): {slope:.3f}")
print(f"  (For uniform 3D fill = 3.0; clustered < 3.0; fractal structures are non-integer)")

print("\n=== 7. ANGULAR DISTRIBUTION (is cloud isotropic?) ===")
# Convert centered coords to spherical angles
norms = np.linalg.norm(centered, axis=1, keepdims=True)
unit = centered / norms

# Multipole moments: if isotropic, all moments after monopole should vanish
# Simple check: mean direction and its magnitude
mean_dir = unit.mean(axis=0)
print(f"  Mean unit direction: {mean_dir}  |magnitude|={np.linalg.norm(mean_dir):.4f}")
print(f"  (0 = isotropic, 1 = all pointing same way)")

# Variance per axis after centering on unit sphere
print(f"  Variance of unit vectors per axis: x={unit[:,0].var():.4f}  y={unit[:,1].var():.4f}  z={unit[:,2].var():.4f}")

print("\n=== 8. PAIRWISE DISTANCE DISTRIBUTION ===")
# Sample pairwise distances
rng = np.random.default_rng(42)
idx = rng.choice(len(coords3d), size=200, replace=False)
sub = coords3d[idx]
pw = []
for i in range(len(sub)):
    for j in range(i+1, len(sub)):
        pw.append(np.linalg.norm(sub[i]-sub[j]))
pw = np.array(pw)
print(f"  Pairwise distances: mean={pw.mean():.3f}  std={pw.std():.3f}  min={pw.min():.3f}  max={pw.max():.3f}")

# In a uniform sphere, pairwise distances peak around 1.0-1.3x diameter
sphere_radius = radii.max()
print(f"  Sphere radius: {sphere_radius:.3f}")
print(f"  Mean pairwise / (2*radius): {pw.mean()/(2*sphere_radius):.3f}  (uniform sphere ~0.75)")

print("\n=== 9. CONVEX HULL ===")
hull = ConvexHull(coords3d)
print(f"  Hull volume:   {hull.volume:.4f}")
print(f"  Hull surface:  {hull.area:.4f}")
sphere_vol = (4/3)*np.pi*sphere_radius**3
print(f"  Enclosing sphere volume: {sphere_vol:.4f}")
print(f"  Fill fraction (hull/sphere): {hull.volume/sphere_vol:.3f}")
# Isoperimetric ratio: 36*pi*V^2 / S^3  =  1 for sphere, <1 for non-spherical
iso = 36 * np.pi * hull.volume**2 / hull.area**3
print(f"  Isoperimetric ratio (1=sphere): {iso:.4f}")