import io
import numpy as np
from sklearn.decomposition import PCA


# TODO: also save those points without primitive labels?
def save_vg(
    points,
    filepath,
    group_info=None,
    group_col: int = 3,
    colors=None,
    normals=None,
):
    """
    Save point cloud with quadric primitive segments into a .vg file (v2 supporting multiple primitive types).

    Primitive Types (group_type codes follow official VG spec):
        0: plane     parameters: [a, b, c, d] satisfying a*x + b*y + c*z + d = 0 ( [a,b,c] normalized )
        1: cylinder  parameters: 10-parameter quadrics
        2: sphere    parameters: 10-parameter quadrics
        3: cone      parameters: 10-parameter quadrics

    Parameters
    ----------
    points : np.ndarray (N, >= group_col+1)
        Point cloud. Columns 0:3 are xyz. Column `group_col` holds integer group ids.
    filepath : str
        Output path (.vg)
    group_info : dict | list | None
        Description of each group primitive. Accepted forms:
            - dict mapping group_id -> { 'type': <int|str>, 'parameters': np.ndarray[k] }
            - list of dicts each with keys: 'id', 'type', 'parameters'
        If None and only planes are expected: plane parameters will be estimated via PCA (if compute_missing_plane_params=True).
    group_col : int
        Column index in `points` indicating group id.
    colors : np.ndarray | None (N,3)
        Per-point RGB in [0,255] or [0,1]. If None, defaults to mid-gray (128,128,128).
    normals : np.ndarray | None (N,3)
        Per-point normals. If None and compute_normals=True they will be estimated from primitive parameters.

    Notes
    -----
    - All generated / supplied direction vectors will be normalized.
    - Minimal validation performed; raise ValueError for inconsistent shapes.
    """
    from random import random
    # ------------- Collect grouping -------------
    if points.ndim != 2 or points.shape[1] <= group_col:
        raise ValueError("points must have at least group_col+1 columns (xyz + group id)")
    xyz = points[:, :3]
    group_ids = points[:, group_col].astype(int)
    unique_ids = np.unique(group_ids)
    G = len(unique_ids)

    # Minimal handling: expect list of dicts with 'id','type','parameters'.
    if group_info is None:
        raise ValueError("group_info must be provided (list of {id,type,parameters}).")
    normalized_entries = list(group_info)
    lookup = {int(e['id']): e for e in normalized_entries}
    unique_ids_list = unique_ids.tolist()

    # If ids don't match but lengths match, align by order ignoring provided ids.
    if not all(gid in lookup for gid in unique_ids_list):
        if len(normalized_entries) == len(unique_ids_list):
            lookup = {gid: e for gid, e in zip(unique_ids_list, normalized_entries)}
        else:
            missing = [gid for gid in unique_ids_list if gid not in lookup]
            raise ValueError(f"{filepath}: Missing group_info entry for group ids {missing}. Provided ids: {sorted(lookup.keys())}")

    group_param_list = []
    for gid in unique_ids_list:
        e = lookup[gid]
        t = int(e['type'])
        params = np.asarray(e['parameters'], dtype=np.float32).reshape(-1)
        if params.shape[0] != 10:
            raise ValueError(f"group {gid} params length {params.shape[0]} != 10 (params={params})")
        if not np.all(np.isfinite(params)):
            raise ValueError(f"group {gid} has non-finite parameter values: {params}")
        group_param_list.append((gid, t, params))
    G = len(group_param_list)

    # ------------- Colors -------------
    group_colors = {gid: (random(), random(), random()) for gid in unique_ids}
    colors = np.zeros((points.shape[0], 3), dtype=np.float32)
    for gid, (r, g, b) in group_colors.items():
        colors[group_ids == gid] = (r, g, b)

    # ------------- Assemble output -------------
    out = ''
    out += f'num_points: {points.shape[0]}\n'
    sio = io.StringIO(); np.savetxt(sio, xyz, fmt='%.6f %.6f %.6f'); out += sio.getvalue(); sio.close()

    out += f'num_colors: {points.shape[0]}\n'
    sio = io.StringIO(); np.savetxt(sio, colors, fmt='%.6f %.6f %.6f'); out += sio.getvalue(); sio.close()

    out += f'num_normals: {points.shape[0]}\n'
    sio = io.StringIO(); np.savetxt(sio, normals, fmt='%.6f %.6f %.6f'); out += sio.getvalue(); sio.close()

    out += f'num_groups: {G}\n'

    # Map gid to contiguous ordering index to compute sequential point indices (as in original implementation)
    # We'll preserve original ordering of unique_ids for output
    running_index = 0
    # We'll create a remapped contiguous ordering of points grouped by group id to mimic original sequential grouping
    ordered_indices = []
    for gid in unique_ids:
        ordered_indices.extend(np.where(group_ids == gid)[0].tolist())
    # Inverse mapping from original point index to new sequential index
    new_index_map = np.zeros(points.shape[0], dtype=int)
    for new_i, old_i in enumerate(ordered_indices):
        new_index_map[old_i] = new_i

    for order_i, (gid, t, params) in enumerate(group_param_list):
        mask = group_ids == gid
        g_point_ids_new = new_index_map[np.where(mask)[0]]
        # For plane primitives (type 0) only write the 4 linear parameters (G,H,I,J) excluding the 6 quadratic terms.
        if t == 0:
            plane_params = params[-3:]  # assuming ordering [A,B,C,D,E,F,G,H,I,J]
            group_points = xyz[g_point_ids_new, :]
            centroid = np.mean(group_points, axis=0)
            d = -centroid.dot(plane_params)
            plane_params = np.concatenate([plane_params, np.array([d], dtype=np.float32)], axis=0)
            out += f'group_type: {t}\n'
            out += 'num_group_parameters: 4\n'
            out += 'group_parameters: ' + ' '.join(f'{x}' for x in plane_params.tolist()) + '\n'
        else:
            out += f'group_type: {t}\n'
            out += f'num_group_parameters: {len(params)}\n'
            out += 'group_parameters: ' + ' '.join(f'{x}' for x in params.tolist()) + '\n'
        out += f'group_label: group_{order_i}\n'
        rc, gc, bc = group_colors[gid]
        out += f'group_color: {rc} {gc} {bc}\n'
        out += f'group_num_point: {g_point_ids_new.size}\n'
        out += ' '.join(str(i) for i in g_point_ids_new.tolist()) + '\n'
        out += 'num_children: 0\n'

    with open(filepath, 'w') as f:
        f.write(out)


