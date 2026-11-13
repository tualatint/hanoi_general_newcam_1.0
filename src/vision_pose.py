"""Vision-only disk pose estimation for the Hanoi ghost-servo experiment.

No simulator state is used here: the input is the same depth+RGB camera that
produces the policy's point cloud.  The output is the moving disk's centre and
tilt in world coordinates, so the caller can build

    e = ghost_pose - estimated_disk_pose

without touching ``env.data`` / ``env.disk_center``.

Geometry
--------
Disks are flat rings: outer radii 38/32/26 mm, hole 7.7 mm, thickness 20 mm.
The top face is an annulus; from the fixed camera it is the dominant planar
surface.  The estimator

1. segments the disk by hue (the three disks are red / green / blue),
2. RANSAC-fits the top-face plane,
3. fits a circle of *known* outer radius to the visible outer boundary,
4. returns centre = plane centroid - normal * half_thickness, plus the tilt
   vector (the world xy components of the top-face normal).

The fixed-radius circle fit is the key to sub-millimetre centring: the disk's
outer radius is a known constant, so a partially occluded arc still determines
the centre without the bias a free centroid would have.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Nominal RGB (0-255) from hanoi_scene.xml.  Hue is stable under shading.
DISK_RGB = {
    "disk_l": np.array([0.90, 0.12, 0.12], dtype=np.float64) * 255.0,
    "disk_m": np.array([0.12, 0.72, 0.22], dtype=np.float64) * 255.0,
    "disk_s": np.array([0.18, 0.32, 0.95], dtype=np.float64) * 255.0,
}
DISK_OUTER_R = {"disk_l": 0.038, "disk_m": 0.032, "disk_s": 0.026}
DISK_HOLE_R = 0.0077
DISK_HALF_T = 0.010
PEG_TOP_Z = 0.19 + 0.060


# ----------------------------------------------------------------------
# colour segmentation
# ----------------------------------------------------------------------
def recolor_robot_primaries(model, grey=(0.16, 0.16, 0.16)) -> int:
    """Make the robot's pure-primary visual geoms grey.

    The incumbent ARX XML paints the finger tips pure red (1,0,0) and the
    finger pads pure blue (0,0,1) -- the *same hue* as disk_l and disk_s.
    Any colour-based disk segmentation then locks onto the gripper exactly
    when the disk is held, which is the only time the servo error matters.
    This changes only ``geom_rgba`` (visual), never collision geometry.
    """
    n = 0
    for g in range(model.ngeom):
        name = model.geom(g).name or ""
        if name.startswith("disk_"):
            continue
        rgba = np.array(model.geom_rgba[g], dtype=np.float64)
        rgb = rgba[:3]
        if rgb.max() - rgb.min() > 0.90 and rgb.min() < 0.02:
            model.geom_rgba[g] = [grey[0], grey[1], grey[2], rgba[3]]
            n += 1
    return n


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) RGB in 0-255 -> (..., 3) HSV with H in [0, 1)."""
    x = np.asarray(rgb, dtype=np.float64) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = np.max(x, axis=-1)
    mn = np.min(x, axis=-1)
    d = mx - mn
    h = np.zeros_like(mx)
    # standard hue formula; undefined when d == 0 (grey), left at 0
    m = d > 1e-9
    rm = m & (mx == r)
    gm = m & (mx == g)
    bm = m & (mx == b)
    h[rm] = ((g[rm] - b[rm]) / d[rm]) % 6.0
    h[gm] = (b[gm] - r[gm]) / d[gm] + 2.0
    h[bm] = (r[bm] - g[bm]) / d[bm] + 4.0
    h = h / 6.0
    s = np.where(mx > 1e-9, d / np.maximum(mx, 1e-9), 0.0)
    return np.stack([h, s, mx], axis=-1)


def hue_of(rgb: np.ndarray) -> float:
    return float(rgb_to_hsv(np.asarray(rgb, dtype=np.float64))[..., 0])


def segment_disk(points: np.ndarray, name: str,
                 hue_tol: float = 0.055,
                 s_min: float = 0.35, v_min: float = 0.10) -> np.ndarray:
    """Boolean mask of points belonging to ``name`` (xyz+rgb, RGB 0-255).

    Points whose ghost-flag channel (column 6, if present) is set are never
    treated as real disk points.
    """
    if points.shape[1] >= 7:
        real = points[:, 6] < 0.5
    else:
        real = np.ones(len(points), dtype=bool)
    hsv = rgb_to_hsv(points[:, 3:6])
    target = hue_of(DISK_RGB[name])
    dh = np.abs(hsv[:, 0] - target)
    dh = np.minimum(dh, 1.0 - dh)  # hue wraps
    return (real & (dh <= hue_tol) & (hsv[:, 1] >= s_min)
            & (hsv[:, 2] >= v_min))


def segment_all_disks(points: np.ndarray) -> dict:
    return {name: segment_disk(points, name) for name in DISK_RGB}


# ----------------------------------------------------------------------
# plane + fixed-radius circle fit
# ----------------------------------------------------------------------
def _plane_from_3(p: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    n = np.cross(p[1] - p[0], p[2] - p[0])
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        return None
    n = n / nn
    return n, float(n @ p[0])


def _plane_inliers(xyz: np.ndarray, n: np.ndarray, d: float,
                   tol: float) -> np.ndarray:
    return np.abs(xyz @ n - d) <= tol


def fit_plane_ransac(xyz: np.ndarray, tol: float = 0.0012,
                     iters: int = 300, normal_z_min: float = 0.60,
                     rng: Optional[np.random.Generator] = None
                     ) -> Tuple[Optional[np.ndarray], Optional[float], np.ndarray]:
    """Largest near-horizontal planar patch (the visible annulus).

    ``normal_z_min`` rejects the vertical side-wall facets: the top face of an
    upright disk has n_z ~ 1, a side facet n_z ~ 0.  Without this prior, a
    heavily occluded small disk (few top-face points) is often fitted to one
    of its 32 vertical box faces.
    """
    if len(xyz) < 3:
        return None, None, np.zeros(len(xyz), dtype=bool)
    rng = rng or np.random.default_rng(0)
    best_in = np.zeros(len(xyz), dtype=bool)
    best_n = None
    best_d = None
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(len(xyz), size=3, replace=False)
        pl = _plane_from_3(xyz[idx])
        if pl is None:
            continue
        n, d = pl
        if n[2] < 0:
            n, d = -n, -d
        if n[2] < normal_z_min:
            continue
        inl = _plane_inliers(xyz, n, d, tol)
        c = int(inl.sum())
        if c > best_count:
            best_count, best_in, best_n, best_d = c, inl, n, d
    if best_n is None or best_count < 6:
        return None, None, np.zeros(len(xyz), dtype=bool)
    # refine with SVD on the inliers
    P = xyz[best_in]
    c = P.mean(axis=0)
    u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    d = float(n @ c)
    inl = _plane_inliers(xyz, n, d, tol)
    return n, d, inl


def _basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, a)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def ransac_circle_known_radius(uv: np.ndarray, radius: float,
                               tol: float = 0.0015, iters: int = 600,
                               rng: Optional[np.random.Generator] = None
                               ) -> Tuple[Optional[np.ndarray], int, float]:
    """Fixed-radius circle RANSAC.

    Each pair of points defines two possible centres (circle of radius R
    through both); the true centre is the one with the most points on the
    outer rim.  This does not need a good initial centre, which matters when
    the visible arc of a stacked disk is narrow and the centroid is biased by
    tens of millimetres.

    Returns (centre, n_inliers, angular coverage in radians).
    """
    if len(uv) < 4:
        return None, 0, 0.0
    rng = rng or np.random.default_rng(0)
    n = len(uv)
    best_c = None
    best_count = 0
    for _ in range(iters):
        i, j = rng.choice(n, size=2, replace=False)
        p, q = uv[i], uv[j]
        d = q - p
        dist = float(np.linalg.norm(d))
        if dist < 0.15 * radius or dist > 2.0 * radius:
            continue
        mid = 0.5 * (p + q)
        h2 = radius * radius - 0.25 * dist * dist
        if h2 <= 0:
            continue
        h = float(np.sqrt(h2))
        perp = np.array([-d[1], d[0]]) / dist
        for c in (mid + h * perp, mid - h * perp):
            r = np.linalg.norm(uv - c, axis=1)
            count = int((np.abs(r - radius) <= tol).sum())
            if count > best_count:
                best_count, best_c = count, c
    if best_c is None or best_count < 6:
        return None, best_count, 0.0
    # refine on inliers
    c = best_c
    for _ in range(15):
        r = np.linalg.norm(uv - c, axis=1)
        inl = np.abs(r - radius) <= tol
        if int(inl.sum()) < 6:
            break
        P = uv[inl]
        dvec = P - c
        rr = np.maximum(np.linalg.norm(dvec, axis=1), 1e-9)
        res = rr - radius
        J = -dvec / rr[:, None]
        H = J.T @ J + 1e-8 * np.eye(2)
        g = J.T @ res
        step = np.linalg.solve(H, -g)
        c = c + step
        if np.linalg.norm(step) < 1e-7:
            break
    r = np.linalg.norm(uv - c, axis=1)
    inl = np.abs(r - radius) <= tol
    ang = np.sort(np.arctan2(uv[inl, 1] - c[1], uv[inl, 0] - c[0]))
    # visible angular coverage (largest gap complement)
    if len(ang) >= 2:
        gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
        coverage = float(2 * np.pi - gaps.max())
    else:
        coverage = 0.0
    return c, int(inl.sum()), coverage


def fit_circle_known_radius(uv: np.ndarray, radius: float, n_bins: int = 72,
                            gate: float = 0.12
                            ) -> Tuple[Optional[np.ndarray], int]:
    """Fit a circle of known radius to the outer boundary of an annulus.

    ``uv`` are in-plane coordinates of the top-face inliers.  Boundary points
    are the farthest point in each angular bin around a robust initial centre,
    but a bin is only accepted if its farthest point is within ``gate*R`` of
    the known radius.  This rejects the *inner* rim / occlusion boundary
    points that appear when the arm or gripper hides part of the outer rim;
    those outliers otherwise pull the fitted centre by several millimetres.

    Returns (2-D centre, number of accepted bins).
    """
    if len(uv) < 6:
        return None, 0
    c = np.median(uv, axis=0)
    kept = 0
    for _ in range(4):
        ang = np.arctan2(uv[:, 1] - c[1], uv[:, 0] - c[0])
        radial = np.linalg.norm(uv - c, axis=1)
        bins = ((ang + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
        boundary = []
        for b in range(n_bins):
            m = bins == b
            if not np.any(m):
                continue
            j = np.argmax(radial[m])
            p = uv[m][j]
            if abs(np.linalg.norm(p - c) - radius) <= gate * radius:
                boundary.append(p)
        kept = len(boundary)
        if kept < 5:
            return None, kept
        B = np.asarray(boundary)
        # robust Gauss-Newton for min sum (||p - c|| - R)^2
        cc = B.mean(axis=0)
        for _ in range(20):
            d = B - cc
            r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            res = r - radius
            w = 1.0 / np.maximum(1.0, np.abs(res) / 0.0015)
            J = -d / r[:, None]
            H = (J * w[:, None]).T @ J + 1e-6 * np.eye(2)
            g = (J * w[:, None]).T @ res
            step = np.linalg.solve(H, -g)
            cc = cc + step
            if np.linalg.norm(step) < 1e-7:
                break
        c = cc
    return c, kept


def sample_disk_model(name: str, n: int = 3000,
                      rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Sample the known ring surface in the disk's local frame (z = axis)."""
    rng = rng or np.random.default_rng(1)
    R = DISK_OUTER_R[name]
    r_in = DISK_HOLE_R
    h = DISK_HALF_T
    n_top = int(0.40 * n)
    n_bot = int(0.15 * n)
    n_out = int(0.35 * n)
    n_in = n - n_top - n_bot - n_out
    # uniform on an annulus: r = sqrt(u*(R^2-r_in^2)+r_in^2)
    def annulus(k, z):
        u = rng.random(k)
        th = rng.random(k) * 2 * np.pi
        r = np.sqrt(u * (R * R - r_in * r_in) + r_in * r_in)
        return np.stack([r * np.cos(th), r * np.sin(th),
                         np.full(k, z)], axis=-1)
    top = annulus(n_top, h)
    bot = annulus(n_bot, -h)
    th = rng.random(n_out) * 2 * np.pi
    z = rng.uniform(-h, h, n_out)
    out = np.stack([R * np.cos(th), R * np.sin(th), z], axis=-1)
    th = rng.random(n_in) * 2 * np.pi
    z = rng.uniform(-h, h, n_in)
    inn = np.stack([r_in * np.cos(th), r_in * np.sin(th), z], axis=-1)
    return np.concatenate([top, bot, out, inn], axis=0)


def _rot_from_normal(n: np.ndarray) -> np.ndarray:
    """Rotation whose local +z maps to ``n`` (yaw arbitrary)."""
    n = n / np.linalg.norm(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return np.stack([u, v, n], axis=1)


def icp_disk_pose(xyz: np.ndarray, name: str,
                  centre0: np.ndarray, normal0: np.ndarray,
                  rng: Optional[np.random.Generator] = None
                  ) -> Optional[dict]:
    """Rigid ICP of the known ring model to the observed disk points.

    The coarse plane/circle fit can be several millimetres off when the
    visible arc is narrow.  ICP uses the whole visible surface (top annulus +
    outer/inner wall) and the exact model radius, so the side-wall points pin
    the lateral position even when the top-face boundary is eroded.
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None
    model = sample_disk_model(name, rng=rng)
    R = _rot_from_normal(normal0)
    t = np.asarray(centre0, dtype=np.float64).copy()
    thresh = 0.018
    tree = None
    kept = 0
    rms = float("inf")
    for it in range(40):
        Q = model @ R.T + t
        tree = cKDTree(Q)
        import os as _os
        d, idx = tree.query(
            xyz, k=1, workers=int(_os.environ.get('VSERVO_ICP_WORKERS', '-1')))
        keep = d < thresh
        kept = int(keep.sum())
        if kept < 12:
            return None
        P = xyz[keep]
        Qm = Q[idx[keep]]
        pc = P.mean(axis=0)
        qc = Qm.mean(axis=0)
        H = (Qm - qc).T @ (P - pc)
        U, S, Vt = np.linalg.svd(H)
        D = np.eye(3)
        if np.linalg.det(Vt.T @ U.T) < 0:
            D[2, 2] = -1.0
        dR = Vt.T @ D @ U.T
        dt = pc - dR @ qc
        R = dR @ R
        t = dR @ t + dt
        rms = float(np.sqrt(np.mean(d[keep] ** 2)))
        thresh = max(0.0025, thresh * 0.88)
        if np.linalg.norm(dt) < 1e-6 and it > 10:
            break
    # orthonormalise R (accumulated numerical drift)
    U, S, Vt = np.linalg.svd(R)
    R = U @ Vt
    n = R[:, 2]
    if n[2] < 0:
        n = -n
    return {
        "centre": t,
        "normal": n,
        "icp_rms": rms,
        "icp_kept": kept,
    }


def fit_disk_pose(xyz: np.ndarray, name: str,
                  rng: Optional[np.random.Generator] = None
                  ) -> Optional[dict]:
    """Estimate (centre, normal, visible bins, n_inliers) of one disk."""
    if len(xyz) < 8:
        return None
    n, d, inl = fit_plane_ransac(xyz, rng=rng)
    if n is None or int(inl.sum()) < 8:
        return None
    P = xyz[inl]
    u, v = _basis_from_normal(n)
    uv = np.stack([(P - P.mean(axis=0)) @ u, (P - P.mean(axis=0)) @ v], axis=-1)
    c2, n_inl, coverage = ransac_circle_known_radius(
        uv, DISK_OUTER_R[name], rng=rng)
    if c2 is None:
        return None
    # back to 3-D: centre of the top face, then shift down by half thickness
    origin = P.mean(axis=0)
    top_centre = origin + c2[0] * u + c2[1] * v
    centre0 = top_centre - n * DISK_HALF_T
    out = {
        "centre": centre0.astype(np.float64),
        "normal": n.astype(np.float64),
        "tilt_xy": n[:2].astype(np.float64),
        "tilt_rad": float(np.arccos(np.clip(n[2], -1.0, 1.0))),
        "n_plane_inliers": int(inl.sum()),
        "n_circle_inliers": int(n_inl),
        "arc_coverage_rad": float(coverage),
        "arc_coverage_deg": float(np.degrees(coverage)),
        "centre_coarse": centre0.astype(np.float64),
    }
    icp = icp_disk_pose(xyz, name, centre0, n, rng=rng)
    if icp is not None:
        out.update(icp)
        out["tilt_xy"] = icp["normal"][:2].astype(np.float64)
        out["tilt_rad"] = float(
            np.arccos(np.clip(icp["normal"][2], -1.0, 1.0)))
    return out


def estimate_table_z(points: np.ndarray,
                     rng: Optional[np.random.Generator] = None
                     ) -> Optional[dict]:
    """Estimate the table-top height from the grey table points.

    The post-side minimum z is NOT a reliable table height: once the ring
    surrounds the post, the visible yellow starts at the ring top, so a
    ``post_z_min``-based ghost rises by ~10 mm exactly when the ring seats.
    The table plane remains visible around the post, so fit it instead.
    """
    hsv = rgb_to_hsv(points[:, 3:6])
    m = (hsv[:, 1] < 0.25) & (hsv[:, 2] > 0.08)
    m &= (points[:, 2] > 0.180) & (points[:, 2] < 0.202)
    xyz = points[m][:, :3]
    if len(xyz) < 50:
        return None
    n, d, inl = fit_plane_ransac(xyz, tol=0.0012, iters=250,
                                 normal_z_min=0.97, rng=rng)
    if n is None or int(inl.sum()) < 50:
        return None
    z = float(np.median(xyz[inl, 2]))
    return {"table_z": z, "n_points": int(len(xyz)),
            "n_inliers": int(inl.sum()),
            "spread": float(np.percentile(xyz[inl, 2], 90)
                            - np.percentile(xyz[inl, 2], 10))}


def estimate_peg_axis(points: np.ndarray, peg_xy_hint: np.ndarray,
                      radius: float = 0.007,
                      rng: Optional[np.random.Generator] = None
                      ) -> Optional[dict]:
    """Estimate a post axis from the yellow side wall (vision only).

    The fixture layout supplies only a search hint; the returned axis is what
    the ghost pose and the servo error use.  Both the disk estimate and the
    post estimate share the same point-cloud frame, so the systematic
    camera/back-projection offset (~-2.5 mm x in this scene) cancels in the
    relative error.
    """
    hsv = rgb_to_hsv(points[:, 3:6])
    # yellow post (0.85, 0.68, 0.18) -> hue ~0.124
    m = (np.abs(hsv[:, 0] - 0.124) < 0.05) & (hsv[:, 1] > 0.35) \
        & (hsv[:, 2] > 0.10)
    # side wall only: below the spherical cap, above the table
    m &= (points[:, 2] > 0.185) & (points[:, 2] < 0.258)
    m &= (np.hypot(points[:, 0] - peg_xy_hint[0],
                   points[:, 1] - peg_xy_hint[1]) < 0.025)
    xyz = points[m][:, :3]
    if len(xyz) < 8:
        return None
    c2, n_inl, coverage = ransac_circle_known_radius(
        xyz[:, :2] - peg_xy_hint, radius, tol=0.0012, iters=500, rng=rng)
    if c2 is None or coverage < 0.90:
        return None
    axis = np.asarray(peg_xy_hint, dtype=np.float64) + c2
    yellow_z = xyz[:, 2]
    return {
        "axis_xy": axis.astype(np.float64),
        "z_min": float(yellow_z.min()),
        "z_max": float(yellow_z.max()),
        "n_points": int(len(xyz)),
        "n_circle_inliers": int(n_inl),
        "arc_coverage_rad": float(coverage),
    }


def estimate_disk(points: np.ndarray, name: str,
                  rng: Optional[np.random.Generator] = None
                  ) -> Optional[dict]:
    m = segment_disk(points, name)
    out = fit_disk_pose(points[m][:, :3], name, rng=rng)
    if out is not None:
        out["n_color_points"] = int(m.sum())
    return out


def error_to_ghost(est: dict, ghost_centre: np.ndarray) -> np.ndarray:
    """5-D servo error: [ex, ey, ez, nx, ny] (tilt in the world frame)."""
    e = np.asarray(ghost_centre, dtype=np.float64) - est["centre"]
    return np.concatenate([e, est["tilt_xy"]]).astype(np.float32)
