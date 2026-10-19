# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import os
import time

import numpy as np
import trimesh
import xatlas
from scipy.spatial import cKDTree

# Above this face count, xatlas runs on a quadric copy and UVs are transferred.
_FAST_MIN_FACES = 100_000
_CANDIDATES = 16


def mesh_uv_wrap(mesh):
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    if len(mesh.faces) > 500000000:
        raise ValueError("The mesh has more than 500,000,000 faces, which is not supported.")

    if len(mesh.faces) > _FAST_MIN_FACES:
        return _fast_uv_wrap(mesh)

    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

    mesh.vertices = mesh.vertices[vmapping]
    mesh.faces = indices
    mesh.visual.uv = uvs

    return mesh


def _fast_uv_wrap(mesh):
    """Unwrap a decimated copy, then copy UVs onto the full-resolution mesh."""
    started = time.perf_counter()
    n_faces = int(len(mesh.faces))
    target = int(os.environ.get("HUNYUAN_REMESH_FACES", "40000"))
    if target < 1:
        raise ValueError("HUNYUAN_REMESH_FACES must be a positive face count")

    low = _decimate_copy(mesh, target)
    vmapping, indices, uvs = xatlas.parametrize(low.vertices, low.faces)
    low_vertices = np.asarray(low.vertices, dtype=np.float64)[np.asarray(vmapping)]
    low_faces = np.asarray(indices, dtype=np.int64)
    low_uv = np.asarray(uvs, dtype=np.float64)
    wrapped = _transfer_uvs(mesh, low_vertices, low_faces, low_uv)
    elapsed = time.perf_counter() - started
    print(
        f"fast uv: faces={n_faces} decimated={len(low.faces)} seconds={elapsed:.2f}",
        flush=True,
    )
    return wrapped


def _decimate_copy(mesh, target):
    src = trimesh.Trimesh(
        vertices=np.array(mesh.vertices, dtype=np.float64, copy=True),
        faces=np.array(mesh.faces, dtype=np.int64, copy=True),
        process=False,
    )
    if len(src.faces) <= target:
        return src
    try:
        low = src.simplify_quadric_decimation(face_count=int(target))
    except TypeError:
        reduction = 1.0 - (float(target) / float(len(src.faces)))
        reduction = min(0.99, max(0.0, reduction))
        low = src.simplify_quadric_decimation(percent=reduction)
    if len(low.faces) < 1:
        raise RuntimeError("quadric decimation produced an empty mesh")
    return low


def _transfer_uvs(full, low_vertices, low_faces, low_uv):
    vertices = np.asarray(full.vertices, dtype=np.float64)
    faces = np.asarray(full.faces, dtype=np.int64)
    closest, dist, tri_id = _closest_points(low_vertices, low_faces, vertices)
    bary = _normalize_bary(
        trimesh.triangles.points_to_barycentric(low_vertices[low_faces[tri_id]], closest)
    )
    tri_uv = low_uv[low_faces[tri_id]]
    uv_at_vert = np.clip((bary[:, :, None] * tri_uv).sum(axis=1), 0.0, 1.0)

    face_island = _face_islands(low_faces)
    vert_island = face_island[tri_id]
    corner_uv = uv_at_vert[faces].copy()
    corner_island = vert_island[faces]
    dist_corner = dist[faces]

    islands_differ = (corner_island[:, 0] != corner_island[:, 1]) | (
        corner_island[:, 1] != corner_island[:, 2]
    )
    pos = vertices[faces]
    pos_e = np.linalg.norm(pos - np.roll(pos, -1, axis=1), axis=2)
    uv_e = np.linalg.norm(corner_uv - np.roll(corner_uv, -1, axis=1), axis=2)
    extent = max(float(np.ptp(vertices, axis=0).max()), 1e-8)
    ratio = uv_e / np.maximum(pos_e / extent, 1e-8)
    jumps = (uv_e.max(axis=1) > 0.2) | (ratio.max(axis=1) > 25.0)
    seam = islands_differ | jumps
    if np.any(seam):
        _reproject_seams(
            seam,
            faces,
            vertices,
            corner_uv,
            corner_island,
            dist_corner,
            low_vertices,
            low_faces,
            low_uv,
            face_island,
        )

    corner_uv = np.clip(corner_uv, 0.0, 1.0)
    new_vertices, new_faces, new_uv = _weld_corners(vertices, faces, corner_uv)
    if len(new_faces) != len(faces):
        raise RuntimeError("UV transfer changed the face count")
    out = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    out.visual.uv = np.asarray(new_uv, dtype=np.float32)
    return out


def _reproject_seams(
    seam,
    faces,
    vertices,
    corner_uv,
    corner_island,
    dist_corner,
    low_vertices,
    low_faces,
    low_uv,
    face_island,
):
    """Give seam-crossing faces one chart, then split those vertices at weld time."""
    face_ids = np.flatnonzero(seam)
    targets = _majority_island(corner_island[face_ids], dist_corner[face_ids])
    groups = _group_ids(face_island)
    for island in np.unique(targets):
        sel = face_ids[targets == island]
        chart_faces = groups.get(int(island))
        if chart_faces is None or len(chart_faces) == 0:
            continue
        corners = faces[sel]
        uniq, inv = np.unique(corners.reshape(-1), return_inverse=True)
        uv_u, _dist, _tri = _project_chart(
            low_vertices, low_faces[chart_faces], low_uv, vertices[uniq]
        )
        corner_uv[sel] = uv_u[inv.reshape(-1, 3)]


def _project_chart(low_vertices, chart_faces, low_uv, points):
    closest, dist, local_tri = _closest_points(low_vertices, chart_faces, points)
    tri_vidx = chart_faces[local_tri]
    bary = _normalize_bary(trimesh.triangles.points_to_barycentric(low_vertices[tri_vidx], closest))
    uv = np.clip((bary[:, :, None] * low_uv[tri_vidx]).sum(axis=1), 0.0, 1.0)
    return uv, dist, local_tri


def _majority_island(corner_island, dist_corner):
    a = corner_island[:, 0]
    b = corner_island[:, 1]
    c = corner_island[:, 2]
    ab = a == b
    bc = b == c
    ac = a == c
    maj = np.where(ab, a, np.where(bc, b, np.where(ac, a, a)))
    all_diff = ~(ab | bc | ac)
    if np.any(all_diff):
        da = dist_corner[:, 0]
        db = dist_corner[:, 1]
        dc = dist_corner[:, 2]
        pick_a = (da <= db) & (da <= dc)
        pick_b = (db < da) & (db <= dc)
        choice = np.where(pick_a, a, np.where(pick_b, b, c))
        maj = np.where(all_diff, choice, maj)
    return maj


def _group_ids(labels):
    order = np.argsort(labels, kind="mergesort")
    sorted_labels = labels[order]
    cuts = np.flatnonzero(np.diff(sorted_labels)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(sorted_labels)]])
    return {int(sorted_labels[s]): order[s:e] for s, e in zip(starts, ends)}


def _face_islands(faces):
    """Charts are connected components of the seam-split low mesh."""
    faces = np.asarray(faces, dtype=np.int64)
    n_verts = int(faces.max()) + 1 if len(faces) else 0
    parent = np.arange(n_verts, dtype=np.int64)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for tri in faces:
        ra = find(int(tri[0]))
        rb = find(int(tri[1]))
        rc = find(int(tri[2]))
        parent[rb] = ra
        parent[rc] = ra
    roots = np.empty(n_verts, dtype=np.int64)
    for i in range(n_verts):
        roots[i] = find(i)
    _uniq, inv = np.unique(roots[faces[:, 0]], return_inverse=True)
    return inv.astype(np.int64)


def _closest_points(vertices, faces, points, k=_CANDIDATES):
    """Closest point on a triangle, using the k nearest triangle centroids."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    points = np.asarray(points, dtype=np.float64)
    tris = vertices[faces]
    centroids = tris.mean(axis=1)
    n_tri = len(faces)
    if n_tri < 1:
        raise RuntimeError("UV transfer source has no faces")
    kk = min(int(k), n_tri)
    tree = cKDTree(centroids)
    n = len(points)
    tri_id = np.empty(n, dtype=np.int64)
    closest = np.empty((n, 3), dtype=np.float64)
    dist = np.empty(n, dtype=np.float64)
    step = 40000
    for start in range(0, n, step):
        end = min(n, start + step)
        pts = points[start:end]
        _d, idx = tree.query(pts, k=kk, workers=4)
        if kk == 1:
            idx = np.asarray(idx, dtype=np.int64).reshape(-1, 1)
        else:
            idx = np.asarray(idx, dtype=np.int64)
        m = end - start
        flat_pts = np.repeat(pts, kk, axis=0)
        flat_tris = tris[idx.reshape(-1)]
        hit = trimesh.triangles.closest_point(flat_tris, flat_pts)
        delta = flat_pts - hit
        d2 = np.einsum("ij,ij->i", delta, delta).reshape(m, kk)
        best = d2.argmin(axis=1)
        rows = np.arange(m)
        tri_id[start:end] = idx[rows, best]
        closest[start:end] = hit.reshape(m, kk, 3)[rows, best]
        dist[start:end] = np.sqrt(d2[rows, best])
    return closest, dist, tri_id


def _normalize_bary(bary):
    bary = np.nan_to_num(np.asarray(bary, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    bary = np.clip(bary, 0.0, None)
    total = bary.sum(axis=1, keepdims=True)
    total[total < 1e-12] = 1.0
    return bary / total


def _weld_corners(vertices, faces, corner_uv):
    """One UV per vertex, duplicating a vertex where charts disagree."""
    quant = np.rint(np.clip(corner_uv, 0.0, 1.0) * 100000.0).astype(np.int64)
    flat_vi = faces.reshape(-1)
    keys = np.column_stack([flat_vi, quant.reshape(-1, 2)])
    _uniq, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    new_faces = inv.reshape(-1, 3).astype(np.int64)
    new_vertices = vertices[flat_vi[first]]
    new_uv = corner_uv.reshape(-1, 2)[first]
    return new_vertices, new_faces, new_uv
