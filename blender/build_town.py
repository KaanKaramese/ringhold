"""Ringhold town and valley: the environment assets, painted to match the kings.

Run:  blender -b --factory-startup --python blender/build_town.py -- <out_dir>
Writes <out_dir>/ringhold_town.glb (+ .blend). The game loads it after ringhold_assets.glb, so every
asset here replaces the old one of the same name: keep, house1, house2, tower, brazier, wall_marker,
wall_wood, wall_stone, wall_iron, tree_pine, tree_dead, rock, portal, chest, bowrack, plus new props
(grave, fence, lamppost).

Conventions the game relies on: each asset is a root empty at the origin on the ground, front facing
Blender -Y (the game's +Z). Child names containing "flame" flicker (pivot at their base), keep_banner
swings on its pole, portal_core spins (direct child of portal), chest_lid hinges at its back edge, the
archer tower's platform is 4.45 m up, and wall pieces are 10.7 m long along X.
"""
import bpy, bmesh, math, os, sys, random
import numpy as np
from mathutils import Vector, Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_kings as K          # shared modelling, painting and baking helpers

V, ell, tube = K.V, K.ell, K.tube
TAU = math.tau
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- textures
def blocks(n, rows, cols, seed, base, mortar=0.3, gap=(0.035, 0.07)):
    """Tileable masonry: staggered blocks with dark mortar, every block its own value."""
    rng = np.random.default_rng(seed)
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n
    r = np.floor(y * rows).astype(int); off = (r % 2) * 0.5
    cx = x * cols + off; c = np.floor(cx).astype(int) % cols
    fx, fy = cx % 1, (y * rows) % 1
    table = rng.random((rows, cols))
    val = table[r % rows, c]
    g = K.fbm(n, 8, 5, seed + 1)
    col = np.asarray(base)[None, None, :] * (0.62 + 0.55 * val[..., None]) * (0.75 + 0.5 * g[..., None])
    mort = (np.minimum(fx, 1 - fx) < gap[0]) | (np.minimum(fy, 1 - fy) < gap[1])
    col[mort] *= mortar
    edge = np.clip(1 - np.minimum(np.minimum(fx, 1 - fx) / 0.12, np.minimum(fy, 1 - fy) / 0.2), 0, 1)
    return col * (1 - 0.25 * edge[..., None])

def grime(col, n, seed, amt=0.35):
    """Dirt gathering towards the bottom of the texture plus random stains."""
    vv = np.linspace(0, 1, n)[:, None]
    stain = np.clip((K.fbm(n, 6, 4, seed) - 0.55) * 4, 0, 1)
    return col * (1 - amt * (0.6 * vv ** 2 + 0.4 * stain))[..., None]

CLOTH = {'blue': np.array([0.16, 0.19, 0.38]), 'red': np.array([0.46, 0.08, 0.06])}
if K.FLAT: CLOTH = dict(K.CLOTH)   # so cloth_base recognises them and paints flat heraldic colours

def paint():
    n = 512; T = {}
    T['stone'] = K.image('t_stone', grime(blocks(n, 8, 4, 1, [0.46, 0.43, 0.39]), n, 2))
    T['stone_dk'] = K.image('t_stone_dk', grime(blocks(n, 8, 4, 3, [0.27, 0.26, 0.26]), n, 4))
    T['cobble'] = K.image('t_cobble', blocks(n, 6, 6, 5, [0.36, 0.33, 0.29], 0.35, (0.06, 0.06)))
    s = K.fbm(n, 6, 5, 6)
    cr = K.scratches(n, 120, 7, (6, 30))[..., None]
    T['plaster'] = K.image('t_plaster', grime(np.clip(K.shade([0.66, 0.60, 0.50], s, 0.35) - cr * 0.12, 0, 1), n, 8, 0.45))
    T['timber'] = K.image('t_timber', K.shade([0.20, 0.13, 0.08], K.streaks(n, 9, 50).T, 0.7))
    pl = K.streaks(n, 10, 40)
    seams = (((np.arange(n) / n * 6) % 1) < 0.04)[None, :]
    T['planks'] = K.image('t_planks', np.where(seams[..., None], 0.35, 1.0) * K.shade([0.30, 0.20, 0.12], pl, 0.6))
    sl = blocks(n, 10, 7, 11, [0.20, 0.22, 0.27], 0.25, (0.03, 0.12))
    T['slate'] = K.image('t_slate', sl)
    th = K.streaks(n, 12, 120)
    bands = 0.75 + 0.25 * np.cos(np.linspace(0, TAU * 7, n))[:, None]
    T['thatch'] = K.image('t_thatch', K.shade([0.38, 0.31, 0.20], th, 0.8) * bands[..., None])
    T['iron'] = K.image('t_iron', K.shade([0.16, 0.16, 0.17], K.fbm(n, 8, 5, 13), 0.6) * (1 - 0.4 * np.clip((K.fbm(n, 10, 4, 14) - 0.6) * 4, 0, 1))[..., None])
    T['bone'] = K.image('t_bone', grime(K.shade([0.72, 0.66, 0.55], K.fbm(n, 6, 5, 15), 0.4), n, 16, 0.5))
    T['bark'] = K.image('t_bark', K.shade([0.18, 0.14, 0.11], K.streaks(n, 17, 70).T, 0.9))
    nd = K.streaks(n, 18, 160)
    T['needles'] = K.image('t_needles', K.shade([0.15, 0.19, 0.14], nd, 0.9) * (0.7 + 0.5 * K.fbm(n, 8, 4, 19))[..., None])
    rk = K.fbm(n, 4, 6, 20)
    moss = np.clip((K.fbm(n, 6, 4, 21) - 0.5) * 3, 0, 1) * np.linspace(1, 0, n)[:, None]
    rc = K.shade([0.42, 0.40, 0.37], rk, 0.7)
    T['rock'] = K.image('t_rock', rc * (1 - moss[..., None]) + np.array([0.20, 0.23, 0.15]) * moss[..., None] * (0.7 + 0.6 * rk[..., None]))
    T['rope'] = K.image('t_rope', K.shade([0.45, 0.38, 0.26], K.streaks(256, 22, 30), 0.6))
    T['earth'] = K.image('t_earth', K.shade([0.25, 0.20, 0.15], K.fbm(n, 6, 5, 23), 0.7))
    # heraldic banners: navy with the gold fleur-de-lis, crimson with the gold lion, torn hems
    for cname, em in (('blue', 'fleur.png'), ('red', 'lion.png')):
        mask = K.load_mask(em, 512)
        col = K.cloth_base(n, CLOTH[cname], 30 + len(cname))
        e = np.zeros((n, n)); K.stamp(e, mask, n // 2, int(n * 0.42), int(n * 0.6), sx=0.8)
        m, gcol = K.gold_paint(n, e, 40 + len(cname))
        col = col * (1 - m[..., None]) + gcol * m[..., None]
        col, _ = K.worn_hem(col, n, 50 + len(cname), 0.8)
        T['banner_' + cname] = K.image('t_banner_' + cname, col)
    return T

# ---------------------------------------------------------------- helpers
MATS = {}
def mat(name, tex=None, color=(1, 1, 1)):
    MATS[name] = K.material(name, tex, color); return name

def tbox(P, c, size, m, rot=None, s=0.6):
    """Box with world-scaled uvs (s tiles per metre) so tiled textures keep their scale."""
    rot = rot or Matrix.Identity(3)
    hx, hy, hz = (d / 2 for d in size)
    L = [(sx * hx, sy * hy, sz * hz) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
    vs = [P.bm.verts.new(Vector(c) + rot @ Vector(p)) for p in L]
    faces = [((0, 2, 3, 1), (0, 1)), ((4, 5, 7, 6), (0, 1)), ((0, 1, 5, 4), (0, 2)), ((2, 6, 7, 3), (0, 2)), ((0, 4, 6, 2), (1, 2)), ((1, 3, 7, 5), (1, 2))]
    for f, (a, b) in faces:
        P.face([vs[k] for k in f], [((L[k][a] + 50) * s, (L[k][b] + 50) * s) for k in f], m)
    return vs

def prism(P, c, w, d, h, m, rot=None, s=0.6):
    """Triangular gable prism: base w (along X) by d (along Y), apex h above the base, centred at c."""
    rot = rot or Matrix.Identity(3)
    pts = [(-w / 2, -d / 2, 0), (w / 2, -d / 2, 0), (0, -d / 2, h), (-w / 2, d / 2, 0), (w / 2, d / 2, 0), (0, d / 2, h)]
    vs = [P.bm.verts.new(Vector(c) + rot @ Vector(p)) for p in pts]
    for f, ax in (((0, 1, 2), (0, 2)), ((3, 5, 4), (0, 2)), ((0, 3, 4, 1), (0, 1))):
        P.face([vs[k] for k in f], [((pts[k][ax[0]] + 50) * s, (pts[k][ax[1]] + 50) * s) for k in f], m)

def cyl(P, c, r0, r1, h, m, n=10, caps=(False, True), us=2, vs=1, rfn=None):
    rings = [ell(c.x, c.y, c.z + h, r1, r1, n, rfn=rfn), ell(c.x, c.y, c.z, r0, r0, n, rfn=rfn)]
    P.loft(rings, m, cap_top=caps[1], cap_bot=caps[0], uscale=us, vscale=vs)

def cone(P, c, r, h, m, n=10, us=3, vs=2):
    P.loft([ell(c.x, c.y, c.z + h, 0.001, 0.001, n), ell(c.x, c.y, c.z, r, r, n)], m, uscale=us, vscale=vs)

def flame(P, base, sc=1.0):
    """Ragged fire: orange tongues with yellow-white lower bands (same look as the king's torch)."""
    rnd = random.Random(int(base.x * 100 + base.z * 10))
    for ti in range(5):
        ox, oy = rnd.uniform(-0.05, 0.05) * sc, rnd.uniform(-0.05, 0.05) * sc
        h, r, lean = rnd.uniform(0.2, 0.38) * sc, rnd.uniform(0.04, 0.08) * sc, rnd.uniform(-0.5, 0.5)
        rings = []
        for t in (1.0, 0.75, 0.5, 0.25, 0.0):
            w = r * math.sin(math.pi * (0.15 + 0.85 * (1 - t))) ** 0.8 * (1 - t) ** 0.25 if t < 1 else 0.0
            rings.append(ell(base.x + ox + lean * 0.1 * sc * t * t, base.y + oy, base.z + h * t, max(w, 0.001), max(w, 0.001) * 0.8, 7,
                             rfn=lambda a, t=t: 1 + 0.25 * math.sin(a * 3 + t * 7)))
        P.loft(rings, 'glow_fire', cap_bot='glow_firehot', mat_fn=lambda i, j: 'glow_firehot' if i >= 3 else 'glow_fire')

def root(name):
    ob = bpy.data.objects.new(name, None); bpy.context.scene.collection.objects.link(ob); return ob

BAKE = []   # (object, texture size, wear, ao distance, ao darkest)
def done(P, parent, size=512, wear=0.5, ao=(0.25, 0.4), smooth=35, location=(0, 0, 0), bake=True):
    ob = P.build(parent, smooth_angle=smooth, location=location)
    if bake: BAKE.append((ob, size, wear, ao))
    return ob

def glow(name, parent, mats, build, location=(0, 0, 0)):
    G = K.Part(name, mats); build(G); return G.build(parent, smooth_angle=89, location=location)

# ---------------------------------------------------------------- assets
def keep():
    R = root('keep')
    P = K.Part('keep_body', ['t_stone', 't_stone_dk', 't_slate', 't_planks', 't_iron', 't_banner_blue', 't_banner_red'])
    tbox(P, (0, 0, 0.4), (8.2, 8.2, 0.8), 't_stone_dk')
    # tapering square tower
    sq = lambda z, h: [V(h, -h, z), V(h, h, z), V(-h, h, z), V(-h, -h, z)]
    P.loft([sq(10.6, 3.0), sq(5.5, 3.25), sq(0.8, 3.45)], 't_stone', uscale=8, vscale=5)
    # stepped buttresses at the corners
    for sx in (-1, 1):
        for sy in (-1, 1):
            for k, (h, w) in enumerate(((3.2, 1.0), (2.2, 0.8), (1.4, 0.6))):
                z = 0.8 + (0 if k == 0 else (3.2 if k == 1 else 5.4))
                tbox(P, (sx * (3.35 + w / 2 - 0.3), sy * (3.35 + w / 2 - 0.3), z + h / 2), (w, w, h), 't_stone')
    # walkway and battlements
    tbox(P, (0, 0, 10.8), (7.0, 7.0, 0.4), 't_stone_dk')
    for k in range(7):
        t = -3.0 + k * 1.0
        for (cx, cy, sxz) in ((t, -3.25, (0.6, 0.5)), (t, 3.25, (0.6, 0.5)), (-3.25, t, (0.5, 0.6)), (3.25, t, (0.5, 0.6))):
            tbox(P, (cx, cy, 11.35), (sxz[0], sxz[1], 0.7), 't_stone')
    # corner turrets with steep slate spires
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = V(sx * 3.2, sy * 3.2, 8.6)
            cyl(P, c, 0.85, 0.95, 3.2, 't_stone', n=10, caps=(True, True))
            cone(P, c + V(0, 0, 3.2), 1.15, 3.4 if sy < 0 else 2.8, 't_slate', n=10)
    # arched door with planks, stone frame, iron studs
    tbox(P, (0, -3.42, 1.9), (2.4, 0.3, 2.6), 't_stone_dk')
    P.loft([ell(0, -3.5, 3.2, 1.2, 0.18, 10, a0=0, a1=math.pi, closed=False), ell(0, -3.5, 3.2, 0.9, 0.18, 10, a0=0, a1=math.pi, closed=False)], 't_stone_dk', closed=False)
    tbox(P, (0, -3.58, 1.75), (1.8, 0.1, 2.3), 't_planks')
    for k in range(3):
        tbox(P, (0, -3.64, 0.9 + k * 0.75), (1.85, 0.05, 0.1), 't_iron')
    # arrow slits
    for z in (5.0, 7.6):
        for x in (-1.6, 1.6):
            tbox(P, (x, -3.18, z), (0.18, 0.12, 1.0), 't_iron')
    # the two kings' banners hang either side of the door: blue fleur west, red lion east
    for x, m in ((-1.9, 't_banner_blue'), (1.9, 't_banner_red')):
        rows = []
        for k in range(6):
            t = k / 5
            z = 9.6 - t * 4.4 - (0.35 * ((k % 2) * 2 - 1) if k == 5 else 0)
            rows.append([V(x + 0.55, -3.36 - 0.03 * math.sin(t * 3), z), V(x, -3.38 - 0.05 * math.sin(t * 3), z), V(x - 0.55, -3.36 - 0.03 * math.sin(t * 3), z)])
        rows[-1][1].z -= 0.45   # swallowtail point
        P.loft(rows, m, closed=False, flip=True)
        tbox(P, (x, -3.36, 9.7), (1.4, 0.12, 0.1), 't_iron')
    done(P, R, size=1024, wear=0.55)
    # beacon brazier on the roof, torches by the door
    B = K.Part('keep_brazier', ['t_iron'])
    B.loft([ell(0, 0, 12.0, 0.95, 0.95, 10), ell(0, 0, 11.6, 0.7, 0.7, 10), ell(0, 0, 11.0, 0.3, 0.3, 10)], 't_iron', cap_bot=True)
    for x in (-1.2, 1.2):
        tube(B, [V(x, -3.5, 2.6), V(x, -3.75, 2.9)], [0.04, 0.04], 't_iron', n=6)
        B.loft([ell(x, -3.78, 3.05, 0.11, 0.11, 8), ell(x, -3.78, 2.88, 0.06, 0.06, 8)], 't_iron', cap_bot=True)
    done(B, R, size=256, wear=0.4)
    glow('keep_flame', R, ['glow_fire', 'glow_firehot'], lambda G: flame(G, V(0, 0, 11.85), 2.6), location=V(0, 0, 11.85))
    for k, x in enumerate((-1.2, 1.2)):
        glow('keep_flame_%d' % k, R, ['glow_fire', 'glow_firehot'], lambda G, x=x: flame(G, V(x, -3.78, 3.0), 0.9), location=V(x, -3.78, 3.0))
    W = K.Part('keep_windows', ['glow_window'])
    for x in (-1.6, 1.6):
        tbox(W, (x, -3.2, 5.0), (0.12, 0.05, 0.8), 'glow_window')
    W.build(R, smooth_angle=10)
    # banner on a pole above the battlements; its pivot is the pole so it can swing
    pole = K.Part('keep_pole', ['t_timber'])
    tube(pole, [V(2.3, 2.3, 11.0), V(2.3, 2.3, 16.2)], [0.07, 0.05], 't_timber', n=6)
    done(pole, R, size=128, wear=0.2)
    F = K.Part('keep_banner', ['t_banner_red'])
    rows = []
    for k in range(5):
        t = k / 4
        rows.append([V(2.3, 2.3, 16.0 - t * 2.6), V(3.3 + 0.1 * math.sin(t * 4), 2.3 + 0.15 * math.sin(t * 3), 16.0 - t * 2.6 - 0.3 * t), V(4.0, 2.3 + 0.2 * math.sin(t * 2), 15.95 - t * 2.4 - 0.6 * t)])
    F.loft(rows, 't_banner_red', closed=False)
    F.build(R, smooth_angle=89, location=V(2.3, 2.3, 16.0))

def house(name, two):
    R = root(name)
    P = K.Part(name + '_body', ['t_stone', 't_plaster', 't_timber', 't_thatch', 't_slate', 't_planks'])
    w, d = (3.3, 2.9) if two else (3.0, 2.6)
    roof_m = 't_slate' if two else 't_thatch'
    tbox(P, (0, 0, 0.25), (w + 0.2, d + 0.2, 0.5), 't_stone')
    floors = [(0.5, 2.4, 0.0)] + ([(2.4, 4.3, 0.28)] if two else [])
    top = floors[-1][1]
    for z0, z1, jet in floors:
        ww, dd = w + jet * 2, d + jet * 2
        tbox(P, (0, -jet * 0.0, (z0 + z1) / 2), (ww, dd, z1 - z0), 't_plaster')
        # timber frame proud of the plaster
        for sx in (-1, 0, 1):
            tbox(P, (sx * (ww / 2 - 0.08) if sx else 0, -dd / 2 - 0.03, (z0 + z1) / 2), (0.16, 0.08, z1 - z0), 't_timber')
            tbox(P, (sx * (ww / 2 - 0.08) if sx else 0, dd / 2 + 0.03, (z0 + z1) / 2), (0.16, 0.08, z1 - z0), 't_timber')
        for sy in (-1, 1):
            tbox(P, (sy * (ww / 2 + 0.03), 0, (z0 + z1) / 2), (0.08, 0.16, z1 - z0), 't_timber')
            tbox(P, (sy * (ww / 2 + 0.03), 0, (z0 + z1) / 2), (0.08, dd, 0.14), 't_timber')
        for zz in (z0 + 0.08, z1 - 0.08):
            tbox(P, (0, -dd / 2 - 0.04, zz), (ww + 0.1, 0.1, 0.16), 't_timber')
            tbox(P, (0, dd / 2 + 0.04, zz), (ww + 0.1, 0.1, 0.16), 't_timber')
        for sx in (-1, 1):   # diagonal braces on the front
            L = math.hypot(ww / 2 - 0.3, z1 - z0 - 0.2)
            ang = math.atan2(z1 - z0 - 0.2, ww / 2 - 0.3) * sx
            tbox(P, (sx * ww / 4, -dd / 2 - 0.04, (z0 + z1) / 2), (L, 0.07, 0.12), 't_timber', Matrix.Rotation(ang, 3, 'Y'))
    # steep gable roof along X with deep overhang, slightly crooked
    rh = 2.0 if two else 1.9
    over = 0.35
    for sy in (-1, 1):
        half = (d + 0.56 if two else d) / 2 + over
        L = math.hypot(half, rh)
        ang = math.atan2(rh, half)
        tbox(P, (0.03, sy * half / 2, top + rh / 2), (w + 0.9, L, 0.22), roof_m, Matrix.Rotation(-sy * ang, 3, 'X') @ Matrix.Rotation(0.015, 3, 'Z'))
    prism(P, (0, 0, top), d + (0.56 if two else 0), w + (0.56 if two else 0), rh, 't_plaster', Matrix.Rotation(math.pi / 2, 3, 'Z'))   # gable ends face +-X
    tbox(P, (0.03, 0, top + rh + 0.05), (w + 0.95, 0.3, 0.2), 't_timber')
    # chimney
    tbox(P, (w / 2 - 0.5, d / 4, top + rh * 0.6), (0.6, 0.6, rh * 1.5), 't_stone')
    # door and shutters on the front
    tbox(P, (-0.6 if two else -0.55, -d / 2 - 0.06, 1.3), (0.9, 0.08, 1.6), 't_planks')
    for x in ((0.55, 0.95) if not two else (0.6, 1.0)):
        tbox(P, (x + 0.3, -d / 2 - 0.07, 1.6), (0.18, 0.06, 0.7), 't_planks')
    done(P, R, size=512, wear=0.45)
    W = K.Part(name + '_windows', ['glow_window'])
    tbox(W, (0.9 if not two else 0.95, -d / 2 - 0.04, 1.6), (0.5, 0.04, 0.6), 'glow_window')
    if two:
        for x in (-0.9, 0.9):
            tbox(W, (x, -d / 2 - 0.32, 3.4), (0.5, 0.04, 0.6), 'glow_window')
    W.build(R, smooth_angle=10)

def tower():
    R = root('tower')
    P = K.Part('tower_body', ['t_timber', 't_planks', 't_rope', 't_banner_red'])
    for sx in (-1, 1):
        for sy in (-1, 1):
            tube(P, [V(sx * 1.45, sy * 1.45, -0.1), V(sx * 1.15, sy * 1.15, 4.3)], [0.14, 0.12], 't_timber', n=7)
            tube(P, [V(sx * 1.1, sy * 1.1, 4.3), V(sx * 1.0, sy * 1.0, 6.7)], [0.08, 0.07], 't_timber', n=6)
    for (a, b) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
        for z0, z1 in ((0.3, 2.3), (2.3, 4.2)):
            k0, k1 = 1.45 - 0.3 * z0 / 4.3, 1.45 - 0.3 * z1 / 4.3
            tube(P, [V(a[0] * k0, a[1] * k0, z0), V(b[0] * k1, b[1] * k1, z1)], [0.05, 0.05], 't_timber', n=5)
            tube(P, [V(b[0] * k0, b[1] * k0, z0), V(a[0] * k1, a[1] * k1, z1)], [0.05, 0.05], 't_timber', n=5)
    tbox(P, (0, 0, 4.37), (3.0, 3.0, 0.16), 't_planks')
    for sx in (-1, 1):   # railings
        tbox(P, (sx * 1.45, 0, 5.0), (0.08, 2.9, 0.08), 't_timber'); tbox(P, (0, sx * 1.45, 5.0), (2.9, 0.08, 0.08), 't_timber')
        tbox(P, (sx * 1.45, 0, 4.7), (0.06, 2.9, 0.06), 't_timber'); tbox(P, (0, sx * 1.45, 4.7), (2.9, 0.06, 0.06), 't_timber')
    # pyramid roof of crimson cloth on timber
    rings = [ell(0, 0, 8.5, 0.001, 0.001, 4, a0=math.pi / 4), ell(0, 0, 6.6, 1.95, 1.95, 4, a0=math.pi / 4)]
    P.loft(rings, 't_banner_red', uscale=1, vscale=1)
    # ladder at the front
    for x in (-0.3, 0.3):
        tube(P, [V(x, -1.9, 0), V(x, -1.4, 4.4)], [0.04, 0.04], 't_timber', n=5)
    for k in range(9):
        t = (k + 0.5) / 9
        tbox(P, (0, -1.9 + 0.5 * t, 4.4 * t), (0.6, 0.05, 0.05), 't_timber')
    done(P, R, size=512, wear=0.35)

def brazier():
    R = root('brazier')
    P = K.Part('brazier_body', ['t_stone', 't_iron'])
    P.loft([ell(0, 0, 0.9, 0.42, 0.42, 8), ell(0, 0, 0.6, 0.5, 0.5, 8), ell(0, 0, 0.0, 0.62, 0.62, 8)], 't_stone', cap_top=True, uscale=2)
    P.loft([ell(0, 0, 1.45, 0.75, 0.75, 10), ell(0, 0, 1.2, 0.55, 0.55, 10), ell(0, 0, 0.9, 0.25, 0.25, 10)], 't_iron', cap_bot=True)
    for k in range(8):
        a = TAU * k / 8
        tube(P, [V(math.cos(a) * 0.72, math.sin(a) * 0.72, 1.4), V(math.cos(a) * 0.86, math.sin(a) * 0.86, 1.75)], [0.03, 0.015], 't_iron', n=4)
    done(P, R, size=256, wear=0.5)
    glow('brazier_flame', R, ['glow_fire', 'glow_firehot'], lambda G: flame(G, V(0, 0, 1.35), 2.2), location=V(0, 0, 1.35))

def walls():
    rnd = random.Random(3)
    # marker: two stakes, a rope with rag strips, a skull on a post
    R = root('wall_marker')
    P = K.Part('wall_marker_body', ['t_timber', 't_rope', 't_bone', 't_plaster'])
    for x in (-1.9, 1.9):
        tube(P, [V(x, 0, -0.2), V(x, 0, 1.15)], [0.07, 0.05], 't_timber', n=6)
    tube(P, [V(-1.9, 0, 1.0), V(0, 0, 0.82), V(1.9, 0, 1.0)], [0.02, 0.02, 0.02], 't_rope', n=4)
    for k in range(5):
        x = -1.4 + k * 0.7
        tbox(P, (x, 0, 0.7), (0.12, 0.02, 0.28), 't_plaster', Matrix.Rotation(rnd.uniform(-0.2, 0.2), 3, 'Y'))
    tube(P, [V(0, 0.15, -0.2), V(0, 0.15, 1.2)], [0.05, 0.04], 't_timber', n=6)
    P.loft([ell(0, 0.12, 1.42, 0.06, 0.06, 8), ell(0, 0.12, 1.32, 0.13, 0.12, 8), ell(0, 0.12, 1.18, 0.09, 0.1, 8)], 't_bone', cap_top=True, cap_bot=True)
    done(P, R, size=256, wear=0.3)
    # wood: a palisade of sharpened logs bound with beams, an earth bank at its foot
    R = root('wall_wood')
    P = K.Part('wall_wood_body', ['t_bark', 't_timber', 't_rope', 't_earth'])
    x = -5.35
    while x < 5.35:
        r = rnd.uniform(0.15, 0.2); h = rnd.uniform(2.6, 3.1); lean = rnd.uniform(-0.04, 0.04)
        top = V(x + lean, 0, h)
        P.loft([ell(top.x, 0, h + 0.45, 0.001, 0.001, 6), ell(top.x, 0, h, r, r, 6), ell(x, 0, -1.2, r, r, 6)], 't_bark', uscale=1, vscale=3)
        x += r * 2 * 0.95
    for z in (0.6, 1.9):
        tbox(P, (0, 0.24, z), (10.7, 0.14, 0.22), 't_timber')
    P.loft([[V(5.6, -0.9, -1.2), V(5.6, 0.4, 0.5), V(5.6, 1.5, -1.2)], [V(-5.6, -0.9, -1.2), V(-5.6, 0.4, 0.5), V(-5.6, 1.5, -1.2)]], 't_earth', closed=False, uscale=1, vscale=4)
    done(P, R, size=512, wear=0.3)
    # stone and iron: masonry with battlements and torches (iron adds plates, spikes and a red band)
    for tier, (name, h, m) in enumerate((('wall_stone', 3.2, 't_stone'), ('wall_iron', 3.8, 't_stone_dk')), start=2):
        R = root(name)
        mats = ['t_stone', 't_stone_dk', 't_iron', 't_banner_red']
        P = K.Part(name + '_body', mats)
        tbox(P, (0, 0, -0.5), (10.9, 1.6, 1.1), 't_stone_dk')
        tbox(P, (0, 0, h / 2), (10.7, 1.2, h), m)
        n = 9
        for k in range(n):
            xx = -5.35 + (k + 0.5) * 10.7 / n
            tbox(P, (xx, -0.4, h + 0.3), (0.62, 0.42, 0.6), m)
            tbox(P, (xx, 0.4, h + 0.3), (0.62, 0.42, 0.6), m)
        for xx in (-3.6, 0, 3.6):   # arrow slits
            tbox(P, (xx, -0.58, h * 0.6), (0.14, 0.08, 0.8), 't_iron')
        for xx in (-2.7, 2.7):      # torch brackets
            tube(P, [V(xx, -0.6, h - 0.5), V(xx, -0.85, h - 0.2)], [0.04, 0.04], 't_iron', n=5)
            P.loft([ell(xx, -0.88, h - 0.02, 0.1, 0.1, 8), ell(xx, -0.88, h - 0.18, 0.05, 0.05, 8)], 't_iron', cap_bot=True)
        if tier == 3:
            for k in range(4):
                tbox(P, (-4.0 + k * 2.67, -0.62, h * 0.45), (1.6, 0.06, h * 0.7), 't_iron')
            for k in range(22):
                xx = -5.2 + k * 0.495
                P.loft([ell(xx, -0.4, h + 1.15, 0.001, 0.001, 4), ell(xx, -0.4, h + 0.6, 0.06, 0.06, 4)], 't_iron')
            tbox(P, (0, -0.64, h * 0.82), (10.6, 0.04, 0.3), 't_banner_red')
        done(P, R, size=512, wear=0.6)
        for k, xx in enumerate((-2.7, 2.7)):
            glow('%s_flame_%d' % (name, k), R, ['glow_fire', 'glow_firehot'], lambda G, xx=xx: flame(G, V(xx, -0.88, h - 0.05), 1.0), location=V(xx, -0.88, h - 0.05))

def trees():
    rnd = random.Random(9)
    R = root('tree_pine')
    P = K.Part('tree_pine_body', ['t_bark', 't_needles'])
    tube(P, [V(0, 0, -0.1), V(0.1, 0.05, 2.0), V(-0.05, 0.1, 4.0), V(0.05, 0, 5.8)], [0.24, 0.17, 0.11, 0.03], 't_bark', n=7)
    for k in range(7):   # drooping clumps of needles: every bough its own length, tips hanging down
        z = 1.1 + k * 0.72; r = 1.75 * (1 - k / 7.5) + 0.2; n = 16
        jag = [rnd.uniform(0.45, 1.25) if j % 2 == 0 else rnd.uniform(0.3, 0.6) for j in range(n)]
        rf = lambda a, jag=jag: jag[round((a % TAU) / TAU * n) % n]
        P.loft([ell(0, 0, z + 0.9, 0.05, 0.05, n), ell(0, 0, z + 0.3, r * 0.45, r * 0.45, n, rfn=rf),
                ell(0, 0, z - 0.15, r, r, n, rfn=rf, zfn=lambda a, jag=jag: -0.45 * jag[round((a % TAU) / TAU * n) % n])],
               't_needles', uscale=4, vscale=1)
    done(P, R, size=512, wear=0.0, ao=(0.35, 0.35), smooth=25)
    R = root('tree_dead')
    P = K.Part('tree_dead_body', ['t_bark'])
    tube(P, [V(0, 0, -0.1), V(0.12, -0.05, 1.5), V(-0.1, 0.1, 3.0), V(0.15, 0.0, 4.4)], [0.22, 0.15, 0.09, 0.02], 't_bark', n=7)
    for k in range(7):
        z = 1.6 + k * 0.4; a = rnd.uniform(0, TAU); L = rnd.uniform(0.7, 1.3) * (1 - k / 10)
        p0 = V(0, 0, z); p1 = p0 + V(math.cos(a) * L * 0.6, math.sin(a) * L * 0.6, L * 0.5); p2 = p1 + V(math.cos(a + 0.5) * L * 0.4, math.sin(a + 0.5) * L * 0.4, L * 0.15)
        tube(P, [p0, p1, p2], [0.06, 0.035, 0.01], 't_bark', n=5)
    done(P, R, size=256, wear=0.0, ao=(0.2, 0.5), smooth=60)
    R = root('rock')
    P = K.Part('rock_body', ['t_rock'])
    noise = [rnd.uniform(0.75, 1.2) for _ in range(12 * 7)]
    rings = []
    for i in range(7):
        t = i / 6; z = 0.85 * math.cos(math.pi * t) * 0.6 + 0.35
        rr = math.sin(math.pi * t)
        rings.append(ell(0, 0, z, 1.1 * rr + 0.001, 0.9 * rr + 0.001, 12, rfn=lambda a, i=i: noise[i * 12 + int((a % TAU) / TAU * 12) % 12]))
    P.loft(rings, 't_rock', uscale=2, vscale=1)
    done(P, R, size=256, wear=0.3, smooth=40)

def portal():
    rnd = random.Random(5)
    R = root('portal')
    P = K.Part('portal_body', ['t_bone', 't_stone_dk', 't_iron'])
    for sx in (-1, 1):   # two leaning pillars of bone and black stone, ribs and spikes
        tube(P, [V(sx * 2.7, 0, -0.5), V(sx * 2.55, 0, 2.5), V(sx * 2.1, 0, 5.0), V(sx * 1.0, 0, 6.6)], [0.55, 0.45, 0.35, 0.22], 't_stone_dk', n=8)
        for k in range(6):
            z = 0.6 + k * 0.95
            base = V(sx * (2.6 - 0.12 * k * k / 5), 0, z)
            tip = base + V(sx * rnd.uniform(0.5, 0.9), rnd.uniform(-0.3, 0.3), rnd.uniform(0.3, 0.7))
            P.loft([ell(tip.x, tip.y, tip.z, 0.001, 0.001, 5), ell(base.x, base.y, base.z, 0.12, 0.12, 5)], 't_bone')
    P.loft([ell(0, 0, 7.4, 0.001, 0.001, 6), ell(0, 0, 6.4, 0.5, 0.5, 6)], 't_bone')   # central horn
    for k in range(9):   # skulls round the arch
        a = math.pi * (k + 0.5) / 9
        c = V(math.cos(a) * 2.6, -0.35, 3.0 + math.sin(a) * 3.2)
        P.loft([ell(c.x, c.y, c.z + 0.2, 0.08, 0.08, 7), ell(c.x, c.y, c.z + 0.08, 0.2, 0.18, 7), ell(c.x, c.y, c.z - 0.12, 0.13, 0.14, 7)], 't_bone', cap_top=True, cap_bot=True)
    rings = [ell(0, 0, 3.0, 2.45, 3.05, 24, rfn=lambda a: 1.0), ell(0, 0, 3.0, 2.15, 2.75, 24)]   # stone ring (lies in XZ: swap below)
    P.loft([[V(p.x, -0.12, 3.0 + p.y) for p in r] for r in rings], 't_stone_dk', uscale=6)
    P.loft([[V(p.x, 0.12, 3.0 + p.y) for p in r] for r in rings], 't_stone_dk', uscale=6, flip=True)
    for k in range(5):
        a = TAU * k / 5 + 0.3
        tbox(P, (math.cos(a) * 2.8, math.sin(a) * 1.3, 0.1), (0.9, 0.8, 0.6), 't_stone_dk', Matrix.Rotation(a, 3, 'Z'))
    done(P, R, size=512, wear=0.4)
    # the swirling core: a disc in the arch with brighter spiral arms, spinning about the arch's axis
    def core(G):
        disc = [V(2.2 * math.cos(TAU * k / 24), 0, 2.75 * math.sin(TAU * k / 24)) for k in range(24)]
        G.loft([[p * 0.04 for p in disc], disc], 'glow_core', uscale=1)
        G.loft([[p * 0.04 + V(0, 0.01, 0) for p in disc], [p + V(0, 0.01, 0) for p in disc]], 'glow_core', uscale=1, flip=True)
        for arm in range(4):
            pts = []
            for k in range(10):
                t = k / 9; a = arm * TAU / 4 + t * 2.4
                pts.append(V(math.cos(a) * 2.1 * t, -0.03, math.sin(a) * 2.6 * t))
            tube(G, pts, [0.02 + 0.08 * (k / 9) for k in range(10)], 'glow_eye', n=4, hint=V(0, 1, 0))
    G = K.Part('portal_core', ['glow_core', 'glow_eye'])
    core(G)
    G.build(R, smooth_angle=89, location=V(0, 0, 0))
    bpy.data.objects['portal_core'].location = (0, 0, 3.0)

def chest():
    R = root('chest')
    P = K.Part('chest_body', ['t_planks', 't_iron'])
    tbox(P, (0, 0, 0.23), (1.0, 0.68, 0.46), 't_planks', s=1.5)
    for x in (-0.38, 0.38):
        tbox(P, (x, 0, 0.23), (0.06, 0.7, 0.48), 't_iron')
    tbox(P, (0, -0.35, 0.36), (0.14, 0.03, 0.16), 't_iron')
    done(P, R, size=256, wear=0.6)
    L = K.Part('chest_lid', ['t_planks', 't_iron'])
    lid = []
    for k in range(9):   # half-cylinder lid hinged at the back (+Y) edge
        a = math.pi * k / 8
        lid.append([V(-0.5, 0.34 - 0.34 * (1 - math.cos(a)), 0.46 + 0.16 * math.sin(a)), V(0.5, 0.34 - 0.34 * (1 - math.cos(a)), 0.46 + 0.16 * math.sin(a))])
    L.loft(lid, 't_planks', closed=False, uscale=1, vscale=1)
    lob = L.build(R, smooth_angle=50, location=V(0, 0.34, 0.46))
    BAKE.append((lob, 256, 0.5, (0.1, 0.5)))

def bowrack():
    R = root('bowrack')
    P = K.Part('bowrack_body', ['t_timber', 't_planks', 't_rope', 't_banner_blue', 't_banner_red'])
    for x in (-0.95, 0.95):
        tube(P, [V(x, 0, 0), V(x, 0, 2.0)], [0.07, 0.06], 't_timber', n=6)
    tbox(P, (0, 0, 1.65), (2.2, 0.12, 0.12), 't_timber')
    tbox(P, (0, 0, 0.5), (2.0, 0.1, 0.1), 't_timber')
    for k in range(3):
        x = -0.55 + k * 0.55
        pts = [V(x + 0.18 * math.sin(math.pi * t), -0.1, 0.55 + 1.05 * t) for t in [i / 8 for i in range(9)]]
        tube(P, pts, [0.025] * 9, 't_timber', n=4, hint=V(0, 1, 0))
        tube(P, [pts[0], pts[-1]], [0.006, 0.006], 't_rope', n=3)
    for x, m in ((-0.95, 't_banner_blue'), (0.95, 't_banner_red')):
        P.loft([[V(x, 0.02, 2.0), V(x + 0.25 * (1 if x < 0 else -1), 0.02, 1.9)], [V(x, 0.02, 1.35), V(x + 0.25 * (1 if x < 0 else -1), 0.02, 1.5)]], m, closed=False)
    done(P, R, size=256, wear=0.3)

def props():
    rnd = random.Random(11)
    R = root('grave')
    P = K.Part('grave_body', ['t_rock', 't_earth'])
    tilt = Matrix.Rotation(0.12, 3, 'X') @ Matrix.Rotation(0.06, 3, 'Y')
    tbox(P, (0, 0.1, 0.45), (0.6, 0.14, 0.9), 't_rock', tilt, s=1.2)
    rows = [[tilt @ V(0.3 * math.cos(a), 0, 0.9 + 0.28 * math.sin(a)) + V(0, y, 0) for a in [math.pi * k / 8 for k in range(9)]] for y in (0.03, 0.17)]
    P.loft(rows, 't_rock', closed=False)
    P.loft([ell(0, -0.65, 0.12, 0.32, 0.7, 10), ell(0, -0.65, -0.1, 0.45, 0.85, 10)], 't_earth', cap_top=True)
    done(P, R, size=256, wear=0.5)
    R = root('fence')
    P = K.Part('fence_body', ['t_timber'])
    for k in range(4):
        x = -1.5 + k * 1.0
        tube(P, [V(x, 0, -0.2), V(x + rnd.uniform(-0.05, 0.05), 0, 1.0 + rnd.uniform(-0.1, 0.1))], [0.06, 0.05], 't_timber', n=5)
    for z in (0.45, 0.85):
        tbox(P, (0, 0.05, z), (3.2, 0.06, 0.12), 't_timber', Matrix.Rotation(rnd.uniform(-0.05, 0.05), 3, 'Y'))
    done(P, R, size=256, wear=0.3)
    R = root('lamppost')
    P = K.Part('lamppost_body', ['t_timber', 't_iron'])
    tube(P, [V(0, 0, -0.2), V(0, 0, 3.0)], [0.09, 0.07], 't_timber', n=6)
    tube(P, [V(0, 0, 2.8), V(0, -0.6, 2.95)], [0.04, 0.04], 't_timber', n=5)
    P.loft([ell(0, -0.6, 2.85, 0.04, 0.04, 6), ell(0, -0.6, 2.78, 0.17, 0.17, 6), ell(0, -0.6, 2.4, 0.15, 0.15, 6), ell(0, -0.6, 2.33, 0.08, 0.08, 6)], 't_iron', cap_top=True, cap_bot=True)
    done(P, R, size=128, wear=0.3)
    W = K.Part('lamppost_glass', ['glow_window'])
    W.loft([ell(0, -0.6, 2.74, 0.13, 0.13, 6), ell(0, -0.6, 2.42, 0.12, 0.12, 6)], 'glow_window')
    W.build(R, smooth_angle=10)

# ---------------------------------------------------------------- bake + export
def bake_all():
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'; scn.cycles.samples = 12; scn.cycles.device = 'CPU'
    for ob, size, wear, (dist, lo) in BAKE:
        K.BAKE_SIZE[ob.name] = size; K.WEAR[ob.name] = wear; K.AO[ob.name] = (dist, lo)
        K.bake_painted(ob, ob.name)
        m = ob.data.materials[0]; m.name = 't_' + m.name[len('k_'):] if m.name.startswith('k_') else m.name

def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.name = 'Ringhold_Town'
    T = paint()
    for k, tex in T.items(): mat('t_' + k, tex)
    mat('glow_fire', None, (1.0, 0.35, 0.05)); mat('glow_firehot', None, (1.0, 0.75, 0.35))
    mat('glow_window', None, (1.0, 0.62, 0.28)); mat('glow_core', None, (0.42, 0.03, 0.05)); mat('glow_eye', None, (1.0, 0.14, 0.08))
    K.MATS = MATS
    keep(); house('house1', False); house('house2', True); tower(); brazier(); walls(); trees(); portal(); chest(); bowrack(); props()
    bpy.context.view_layer.update()
    if not K.FLAT and '--nobake' not in sys.argv: bake_all()
    for o in bpy.data.objects: o.select_set(True)
    glb = os.path.join(K.OUT, 'ringhold_town.glb')
    kw = dict(filepath=glb, export_format='GLB', use_selection=True, export_apply=True, export_yup=True, export_image_format='JPEG', export_materials='EXPORT')
    try: bpy.ops.export_scene.gltf(**kw, export_jpeg_quality=80)
    except TypeError:
        try: bpy.ops.export_scene.gltf(**kw, export_image_quality=80)
        except TypeError: bpy.ops.export_scene.gltf(**kw)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(K.OUT, 'ringhold_town.blend'), compress=True)
    print('BUILT', glb, os.path.getsize(glb))

main()
