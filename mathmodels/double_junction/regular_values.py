import numpy as np

def sample_outputs(family, time_value, f, samples=61):
    N = family["N"]
    outputs = []
    for index in np.ndindex(*(samples for _ in range(N))):
        x = np.array(index, dtype=float) / (samples - 1)
        p = np.concatenate([x, [time_value]])
        q = f(p)
        if q is not None:
            outputs.append(q)
    return np.array(outputs) if outputs else np.empty((0, N))

# Pick a deep interior point

def pick_deep_interior_point(outputs, k=50):
    """
    Pick a point from outputs that has many neighbors within a small radius.
    This is a proxy for being in a dense, interior region of the image.
    """
    if len(outputs) == 0:
        raise ValueError("No outputs.")

    # Compute pairwise distances (approximate, using a subset if needed)
    n = len(outputs)
    if n > 2000:
        # Subsample for speed
        idx = np.random.choice(n, 2000, replace=False)
        sub = outputs[idx]
    else:
        sub = outputs

    # For each point, count neighbors within radius r
    r = 0.15  # tune if needed
    best_point = None
    best_count = -1

    for i, pt in enumerate(sub):
        dists = np.linalg.norm(sub - pt, axis=1)
        count = int(np.sum((dists < r) & (dists > 1e-12)))
        if count > best_count:
            best_count = count
            best_point = pt.copy()

    # If we subsampled, map back to a nearby point in the full set
    if n > 2000:
        dists_full = np.linalg.norm(outputs - best_point, axis=1)
        best_point = outputs[int(np.argmin(dists_full))]

    print(f"Selected point with {best_count} neighbors within r={r}")
    return best_point
