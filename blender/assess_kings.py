"""Measure the built kings against the concept sheet.

Run:  blender -b <dir>/ringhold_kings.blend --python blender/assess_kings.py -- <out.json>
Every value is a fraction of H, the height from the feet to the top of the helm, so it can be
compared with the same proportion measured on the concept art (targets below, +/- tolerance).
"""
import bpy, sys, json, math
import numpy as np

# measured on the concept sheet (Blue King front view, back view), in units of H
TARGETS = {
    'crown_height':      (0.11, 0.025, 'crown height'),
    'crown_top':         (1.08, 0.04,  'crown top above the feet'),
    'fur_width':         (0.34, 0.04,  'fur mantle width (front)'),
    'cape_bottom_width': (0.44, 0.06,  'cape width seen from behind, at the hem'),
    'cape_top_ratio':    (0.62, 0.08,  'cape width at the top / at the hem (behind)'),
    'belt_height':       (0.56, 0.03,  'belt height'),
    'sword_length':      (0.47, 0.05,  'sword length, pommel to tip'),
    'flame_top':         (1.02, 0.06,  'top of the torch flame'),
    'fur_side_low':      (0.70, 0.04,  'lowest fur at the sides, over the upper arms (item 2)'),
    'helm_ridge_m':      (0.04, 0.01,  'helm centre ridge stands proud of the face, metres (item 4)'),
    'helm_sunk_m':       (0.14, 0.04,  'fur rises above the bottom of the helm, metres (item 4)'),
    'crown_flare':       (1.40, 0.20,  'crown prong tips radius / band base radius (item 4)'),
    'crown_prongs':      (5, 0,        'number of main crown prongs (item 4)'),
    'blade_width':       (0.04, 0.005, 'sword blade width (item 7: at least 0.035 H)'),
}

def verts(name):
    ob = bpy.data.objects[name]
    dg = bpy.context.evaluated_depsgraph_get()
    me = ob.evaluated_get(dg).to_mesh()
    co = np.array([(ob.matrix_world @ v.co)[:] for v in me.vertices])
    root = ob
    while root.parent: root = root.parent
    co -= np.array(root.matrix_world.translation[:])
    return co, me, ob

def measure(k):
    K = 'king_%s_' % k
    arm, me, ob = verts(K + 'armor')
    helm = verts(K + 'helm')[0] if (K + 'helm') in bpy.data.objects else arm
    H = helm[:, 2].max()
    out = {'H_m': round(H, 3)}
    cr, _, _ = verts(K + 'crown_mesh')
    out['crown_height'] = (cr[:, 2].max() - cr[:, 2].min()) / H
    out['crown_top'] = cr[:, 2].max() / H
    fur, _, _ = verts(K + 'fur')
    out['fur_width'] = (fur[:, 0].max() - fur[:, 0].min()) / H
    out['fur_side_low'] = fur[np.abs(fur[:, 0]) > 0.2][:, 2].min() / H
    cape, _, _ = verts(K + 'cape')
    w = cape[:, 0].max() - cape[:, 0].min()
    top = cape[cape[:, 2] > cape[:, 2].max() - 0.06]
    out['cape_bottom_width'] = w / H
    out['cape_top_ratio'] = (top[:, 0].max() - top[:, 0].min()) / w
    # belt: dark-leather faces around the waist
    mats = [m.name for m in ob.data.materials]
    zs = []
    for p in me.polygons:
        if mats[p.material_index] == 'k_leather_dk':
            c = np.mean([arm[i] for i in p.vertices], axis=0)
            if abs(c[0]) < 0.3 and 0.8 < c[2] < 1.2 and c[1] > -0.25: zs.append(c[2])
    root = bpy.data.objects['king_' + k]
    out['belt_height'] = float(np.median(zs)) / H if zs else float(root.get('belt_z', 0.0)) / H
    sw, _, _ = verts(K + 'sword')
    c = sw - sw.mean(0); axis = np.linalg.svd(c, full_matrices=False)[2][0]
    proj = c @ axis
    out['sword_length'] = (proj.max() - proj.min()) / H
    u = (proj - proj.min()) / (proj.max() - proj.min())
    perp = c - np.outer(proj, axis)
    tipward = u if np.abs(perp[u > 0.9]).max() < np.abs(perp[u < 0.1]).max() else 1 - u   # the tip end is the narrow one
    blade = (tipward > 0.35) & (tipward < 0.75)
    out['blade_width'] = 2 * np.linalg.norm(perp[blade], axis=1).max() / H
    band = (helm[:, 2] > 1.71) & (helm[:, 2] < 1.76)   # bare plate between the breaths and the eye slit
    ys = np.unique(np.round(helm[band & (np.abs(helm[:, 0]) < 0.006)][:, 1], 4))
    ridge = ys[0]; face = ys[ys > ridge + 0.003][0]          # ridge front, then the face plate behind it
    out['helm_ridge_m'] = face - ridge
    out['helm_sunk_m'] = fur[:, 2].max() - helm[:, 2].min()
    cz = cr[:, 2] - cr[:, 2].min(); rad = np.hypot(cr[:, 0], cr[:, 1] - cr[:, 1].mean())
    out['crown_flare'] = rad[cz > cz.max() * 0.8].mean() / rad[cz < 0.01].mean()
    tips = cr[cz > cz.max() - 0.035]
    ang = np.sort(np.arctan2(tips[:, 1] - cr[:, 1].mean(), tips[:, 0]))
    gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
    out['crown_prongs'] = int((gaps > 0.4).sum())
    fl, _, _ = verts(K + 'torch_flame')
    out['flame_top'] = fl[:, 2].max() / H
    return out

res = {}
for k in ('blue', 'red'):
    m = measure(k)
    res[k] = m
rows = []
for key, (t, tol, label) in TARGETS.items():
    vals = [res[k][key] for k in ('blue', 'red')]
    ok = all(abs(v - t) <= tol for v in vals)
    rows.append({'key': key, 'label': label, 'target': t, 'tol': tol, 'blue': round(vals[0], 3), 'red': round(vals[1], 3), 'pass': ok})
out = sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else None
if out:
    json.dump({'rows': rows, 'raw': res}, open(out, 'w'), indent=1)
print('| criterion | target | blue | red | result |')
print('|---|---|---|---|---|')
for r in rows:
    print('| %s | %.2f ± %.2f | %.3f | %.3f | %s |' % (r['label'], r['target'], r['tol'], r['blue'], r['red'], 'PASS' if r['pass'] else 'FAIL'))
