"""Ringhold kings: builds king_blue (player 1) and king_red (player 2) from the concept sheet.

Run:  blender -b --factory-startup --python blender/build_kings.py -- <out_dir>
Writes <out_dir>/ringhold_kings.glb and ringhold_kings.blend.

Blender axes: Z up, the king faces -Y, his right hand is -X (torch), left hand +X (sword).
Every texture is painted here with numpy so the build needs nothing but Blender and the
emblem masks in blender/emblems/.
"""
import bpy, bmesh, math, os, sys, random
import numpy as np
from mathutils import Vector, Matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else HERE
TAU = math.tau

# ---------------------------------------------------------------- textures
def up(g, n):
    """Tileable smooth upsample of a square grid to n x n."""
    f = g.shape[0]
    x = np.arange(n) * f / n
    i0 = np.floor(x).astype(int); t = x - i0; t = t * t * (3 - 2 * t); i1 = (i0 + 1) % f
    rows = g[i0] * (1 - t)[:, None] + g[i1] * t[:, None]
    return rows[:, i0] * (1 - t)[None, :] + rows[:, i1] * t[None, :]

def fbm(n, base=4, oct=5, seed=0, pers=0.5, aspect=(1, 1)):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n)); amp = 1.0; tot = 0.0; f = base
    for _ in range(oct):
        fy, fx = min(n, f * aspect[0]), min(n, f * aspect[1])
        g = rng.random((max(fy, fx), max(fy, fx)))
        layer = up(g, n)
        if fy != fx:  # stretched noise: resample one axis
            gy = rng.random((fy, fx))
            yi = np.arange(n) * fy // n; xi = np.arange(n) * fx // n
            layer = up(np.kron(gy, np.ones((1, 1)))[:min(fy, fx)][:, :min(fy, fx)], n) if False else gy[yi][:, xi]
        out += amp * layer; tot += amp; amp *= pers; f *= 2
    return out / tot

def streaks(n, seed, freq=64):
    """Vertical streak noise (varies along x, slowly along y): fur, wood grain, cloth weave."""
    rng = np.random.default_rng(seed)
    a = up(rng.random((freq, freq)), n)
    b = up(rng.random((freq // 8, freq // 8)), n)
    col = rng.random(freq)
    xi = np.arange(n) * freq // n
    s = np.interp(np.arange(n) * freq / n, np.arange(freq + 1), np.append(col, col[0]))
    return np.clip(0.5 * s[None, :] + 0.3 * a.T + 0.2 * b, 0, 1)

def scratches(n, count, seed, length=(8, 40), val=1.0):
    rng = np.random.default_rng(seed); m = np.zeros((n, n))
    for _ in range(count):
        x, y = rng.random(2) * n; a = rng.random() * TAU; L = rng.uniform(*length)
        for t in np.linspace(0, L, int(L) * 2):
            m[int(y + math.sin(a) * t) % n, int(x + math.cos(a) * t) % n] = val * rng.uniform(0.5, 1)
    return m

def shade(base, v, amt):
    return np.clip(np.asarray(base)[None, None, :] * (1 + amt * (v[..., None] - 0.5) * 2), 0, 1)

def load_mask(name, n):
    path = os.path.join(HERE, 'emblems', name)
    img = bpy.data.images.load(path)
    w, h = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, img.channels)[::-1, :, 0]
    bpy.data.images.remove(img)
    yi = (np.arange(n) * h // n); xi = (np.arange(n) * w // n)
    return px[yi][:, xi]

def stamp(canvas_mask, mask, cx, cy, size, rot=0.0, sx=1.0):
    """Paste mask (square) into canvas_mask centred at pixel (cx, cy) with given pixel size."""
    H, W = canvas_mask.shape; m = mask.shape[0]
    r = int(size / 2) + 2
    ys, xs = np.mgrid[max(0, cy - r):min(H, cy + r), max(0, cx - r):min(W, cx + r)]
    if ys.size == 0: return
    dx, dy = xs - cx, ys - cy
    c, s = math.cos(rot), math.sin(rot)
    u = (c * dx + s * dy) / (size * sx) + 0.5; v = (-s * dx + c * dy) / size + 0.5
    ok = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
    val = np.zeros(u.shape)
    val[ok] = mask[(v[ok] * m).astype(int), (u[ok] * m).astype(int)]
    canvas_mask[ys, xs] = np.maximum(canvas_mask[ys, xs], val)

IMAGES = {}
def image(name, rgb):
    h, w, _ = rgb.shape
    img = bpy.data.images.new(name, w, h, alpha=False)
    px = np.ones((h, w, 4), np.float32); px[..., :3] = np.clip(rgb, 0, 1)
    img.pixels.foreach_set(px[::-1].ravel())
    path = os.path.join(OUT, 'tex', name + '.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.filepath_raw = path; img.file_format = 'PNG'; img.save(); img.pack()
    IMAGES[name] = img
    return img

GOLD = np.array([0.66, 0.48, 0.20])
def gold_paint(n, mask, seed, base=GOLD):
    """Worn painted gold: flaked edges, grime, a few bright strokes."""
    g = fbm(n, 6, 5, seed)
    flake = fbm(n, 24, 3, seed + 1)
    m = mask * np.clip((flake - 0.30) * 6, 0, 1)
    col = base[None, None, :] * (0.65 + 0.6 * g[..., None])
    return m, col

def paint_textures(rng):
    T = {}
    n = 512
    # steel plate: near-black iron with grime, rust blooms and pale scratches
    g = fbm(n, 4, 6, 11); r = fbm(n, 8, 4, 12)
    col = shade([0.32, 0.32, 0.33], g, 0.5)
    rust = np.clip((r - 0.62) * 5, 0, 1)[..., None]
    col = col * (1 - rust * 0.5) + np.array([0.24, 0.13, 0.07]) * rust * 0.5
    sc = scratches(n, 260, 13)[..., None]
    T['steel'] = image('k_steel', np.clip(col + sc * 0.22, 0, 1))
    # blade: lighter worn steel
    col = shade([0.46, 0.46, 0.48], fbm(n, 6, 5, 14), 0.35) + scratches(n, 400, 15, (10, 60))[..., None] * 0.2
    T['blade'] = image('k_blade', col)
    # chain/scale mail: rows of overlapping rings
    y, x = np.mgrid[0:n, 0:n] / n * 24
    x2 = x + (np.floor(y) % 2) * 0.5
    d = np.hypot((x2 % 1) - 0.5, (y % 1) - 0.25)
    ring = np.clip(1 - np.abs(d - 0.36) * 12, 0, 1) * (y % 1 > 0.1)
    col = shade([0.10, 0.10, 0.11], fbm(n, 8, 4, 16), 0.5)
    T['mail'] = image('k_mail', np.clip(col + ring[..., None] * np.array([0.22, 0.21, 0.20]) * (0.5 + fbm(n, 8, 3, 17)[..., None]), 0, 1))
    # brigandine: dark leather covered in riveted plates
    y, x = np.mgrid[0:n, 0:n] / n * 8
    x2 = x + (np.floor(y) % 2) * 0.5
    plate = (np.abs((x2 % 1) - 0.5) < 0.44) & (np.abs((y % 1) - 0.5) < 0.42)
    riv = np.hypot((x2 % 1) - 0.5, (y % 1) - 0.18) < 0.07
    col = shade([0.19, 0.15, 0.11], fbm(n, 6, 5, 18), 0.6)
    col = np.where(plate[..., None], col * 1.15, col * 0.45)
    col = np.where(riv[..., None], np.array([0.42, 0.32, 0.16]) * (0.6 + fbm(n, 16, 2, 19)[..., None]), col)
    T['brig'] = image('k_brigandine', col)
    # leather: brown, cracked
    cr = scratches(n, 300, 20, (4, 18))[..., None]
    T['leather'] = image('k_leather', np.clip(shade([0.24, 0.15, 0.09], fbm(n, 6, 6, 21), 0.6) - cr * 0.1, 0, 1))
    T['leather_dk'] = image('k_leather_dk', np.clip(shade([0.12, 0.08, 0.05], fbm(n, 6, 6, 22), 0.6) - cr * 0.05, 0, 1))
    # brass / gold
    T['brass'] = image('k_brass', shade([0.50, 0.36, 0.16], fbm(n, 6, 5, 23), 0.7))
    gcol = shade(GOLD, fbm(n, 5, 6, 24), 0.6)
    grime = np.clip((fbm(n, 10, 4, 25) - 0.55) * 4, 0, 1)[..., None]
    T['gold'] = image('k_gold', np.clip(gcol * (1 - grime * 0.6) + scratches(n, 200, 26)[..., None] * 0.18, 0, 1))
    # fur: pale grey-beige clumps, dark roots near the neck (top of the uv)
    s = streaks(n, 27, 90)
    vv = np.linspace(1, 0, n)[:, None]  # 1 at top row
    root = np.clip(vv * 1.3 - 0.25, 0, 1)
    col = np.array([0.58, 0.55, 0.50])[None, None, :] * (0.5 + 0.75 * s[..., None])
    tip = np.clip((0.35 - vv) * 3, 0, 1)                     # pale, almost white tips
    col = col * (1 - 0.8 * root[..., None] ** 1.3) + 0.22 * tip[..., None]
    col = col * (0.75 + 0.5 * fbm(n, 6, 4, 28)[..., None])
    T['fur'] = image('k_fur', col)
    T['fur_dk'] = image('k_fur_dk', shade([0.13, 0.12, 0.11], streaks(256, 32, 48), 0.7))
    # wood
    s = streaks(n, 29, 40).T
    T['wood'] = image('k_wood', shade([0.20, 0.12, 0.06], s, 0.7))
    T['iron'] = image('k_iron', shade([0.10, 0.10, 0.10], fbm(n, 8, 5, 30), 0.6))
    T['visor'] = image('k_visor', shade([0.02, 0.02, 0.02], fbm(64, 4, 3, 31), 0.4))
    return T

CLOTH = {'blue': np.array([0.16, 0.19, 0.38]), 'red': np.array([0.46, 0.08, 0.06])}
EMBLEM = {'blue': 'fleur.png', 'red': 'lion.png'}

def cloth_base(n, color, seed):
    weave = streaks(n, seed, 128)
    g = fbm(n, 4, 6, seed + 1)
    col = color[None, None, :] * (0.55 + 0.7 * g[..., None]) * (0.9 + 0.2 * weave[..., None])
    return col

def worn_hem(col, n, seed, start=0.72, pale=np.array([0.62, 0.52, 0.36])):
    """Moth-eaten pale blotches gathering toward the bottom edge, as on the concept capes."""
    vv = np.linspace(0, 1, n)[:, None]  # 0 top row, 1 bottom row
    ramp = np.clip((vv - start) / (1 - start), 0, 1)
    spots = fbm(n, 16, 4, seed)
    m = np.clip((spots + ramp * 0.55 - 0.85) * 7, 0, 1) * (ramp > 0)
    return col * (1 - m[..., None]) + pale * (0.55 + 0.4 * spots[..., None]) * m[..., None], m

def paint_king(color_name, seed):
    T = {}
    color = CLOTH[color_name]
    em = load_mask(EMBLEM[color_name], 512)
    # cape: u runs from his left front edge, across the back, to his right front edge; v=1 at the shoulders
    n = 1024
    col = cloth_base(n, color, seed)
    uu = np.linspace(0, 1, n)[None, :]; vv = np.linspace(0, 1, n)[:, None]
    col *= (0.55 + 0.45 * np.clip(vv * 3, 0, 1))[..., None]               # shadowed under the fur
    col *= (1 - 0.35 * np.clip(np.abs(uu - 0.5) * 2 - 0.6, 0, 1) / 0.4)[..., None]  # darker towards the front edges
    big = np.zeros((n, n)); stamp(big, em, n // 2, int(n * 0.36), int(n * 0.44), sx=0.85)
    small = np.zeros((n, n))
    for k in range(7):
        u = 0.13 + k * 0.123
        stamp(small, em, int(u * n), int(n * (0.80 if k % 2 else 0.76)), int(n * 0.12), (k % 3 - 1) * 0.08, sx=0.80)
    m, gcol = gold_paint(n, np.maximum(big, small * 0.85), seed + 2)
    col = col * (1 - m[..., None]) + gcol * m[..., None]
    col, _ = worn_hem(col, n, seed + 3, 0.80)
    T['cape'] = image('k_cape_' + color_name, col)
    # lining: plain darker cloth
    T['lining'] = image('k_lining_' + color_name, cloth_base(256, color * 0.45, seed + 4))
    # skirt / tabard: u around the waist (front at u=0.5), v=1 at the belt
    n = 512
    col = cloth_base(n, color, seed + 5)
    vv = np.linspace(0, 1, n)[:, None]
    col *= (0.6 + 0.4 * np.clip(vv * 4, 0, 1))[..., None]
    small = np.zeros((n, n))
    for k in range(8):
        stamp(small, em, int((k + 0.5) / 8 * n), int(n * 0.74), int(n * 0.10))
    m, gcol = gold_paint(n, small * 0.85, seed + 6)
    col = col * (1 - m[..., None]) + gcol * m[..., None]
    col, _ = worn_hem(col, n, seed + 7, 0.62)
    T['skirt'] = image('k_skirt_' + color_name, col)
    # brooch: gold rim, enamel field, emblem
    n = 256
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n - 0.5
    d = np.hypot(x, y) * 2
    col = cloth_base(n, color * 0.8, seed + 8)
    e = np.zeros((n, n)); stamp(e, em, n // 2, n // 2, int(n * 0.62))
    m, gcol = gold_paint(n, np.maximum(e, (d > 0.80).astype(float)), seed + 9)
    col = col * (1 - m[..., None]) + gcol * m[..., None]
    col *= (1 - 0.5 * np.clip((d - 0.74) * 20, 0, 1) * np.clip((0.82 - d) * 20, 0, 1))[..., None]
    T['brooch'] = image('k_brooch_' + color_name, col)
    # banner-style emblem plate for the gem colour on the crown
    return T

# ---------------------------------------------------------------- materials
def material(name, tex=None, color=(1, 1, 1)):
    if name in bpy.data.materials: return bpy.data.materials[name]
    m = bpy.data.materials.new(name)
    try: m.use_nodes = True
    except Exception: pass
    nt = m.node_tree; b = nt.nodes.get('Principled BSDF')
    b.inputs['Roughness'].default_value = 0.85
    if tex is not None:
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = tex
        nt.links.new(t.outputs['Color'], b.inputs['Base Color'])
        b.inputs['Base Color'].default_value = (1, 1, 1, 1)
    else:
        b.inputs['Base Color'].default_value = (*color, 1)
    if name.startswith('glow_'):
        b.inputs['Emission Color'].default_value = (*color, 1); b.inputs['Emission Strength'].default_value = 3
    m.diffuse_color = (*color, 1)
    return m

# ---------------------------------------------------------------- geometry helpers
class Part:
    """Accumulates geometry for one object in a bmesh with a uv layer and named materials."""
    def __init__(self, name, mats):
        self.name = name; self.mats = mats; self.bm = bmesh.new()
        self.uv = self.bm.loops.layers.uv.new('UVMap')

    def mi(self, mat):
        return self.mats.index(mat)

    def face(self, verts, uvs, mat):
        f = self.bm.faces.new(verts)
        for l, uv in zip(f.loops, uvs): l[self.uv].uv = uv
        f.material_index = self.mi(mat)
        return f

    def loft(self, rings, mat, closed=True, cap_top=False, cap_bot=False, uscale=1.0, vscale=1.0, mat_fn=None, flip=False):
        """rings top->bottom, each ring counter-clockwise seen from above (or along the tube)."""
        R, n = len(rings), len(rings[0])
        vs = [[self.bm.verts.new(p) for p in r] for r in rings]
        cols = n if closed else n - 1
        for i in range(R - 1):
            for j in range(cols):
                j2 = (j + 1) % n
                q = [vs[i][j], vs[i + 1][j], vs[i + 1][j2], vs[i][j2]]
                u0, u1 = j / cols * uscale, (j + 1) / cols * uscale
                v0, v1 = (1 - i / (R - 1)) * vscale, (1 - (i + 1) / (R - 1)) * vscale
                uvs = [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
                if flip: q.reverse(); uvs.reverse()
                self.face(q, uvs, mat_fn(i, j) if mat_fn else mat)
        for cap, ring, rev in ((cap_top, vs[0], True), (cap_bot, vs[-1], False)):
            if not cap: continue
            c = Vector((0, 0, 0))
            for v in ring: c += v.co
            c /= len(ring)
            cv = self.bm.verts.new(c)
            for j in range(len(ring)):
                a, b = ring[j], ring[(j + 1) % len(ring)]
                tri = [cv, b, a] if rev else [cv, a, b]
                if flip: tri.reverse()
                uvs = [(0.5, 0.5)] + [(0.5 + 0.5 * math.cos(TAU * k / len(ring)), 0.5 + 0.5 * math.sin(TAU * k / len(ring))) for k in ((j + 1, j) if rev else (j, j + 1))]
                self.face(tri, uvs, cap if isinstance(cap, str) else mat)
        return vs

    def box(self, c, size, mat, rot=None, uvs_scale=4.0):
        rot = rot or Matrix.Identity(3)
        hx, hy, hz = (s / 2 for s in size)
        P = [Vector(c) + rot @ Vector((sx * hx, sy * hy, sz * hz)) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
        vs = [self.bm.verts.new(p) for p in P]
        F = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
        for f in F:
            self.face([vs[k] for k in f], [(0, 0), (1, 0), (1, 1), (0, 1)], mat)
        return vs

    def v_from_height(self, ztop, zbot):
        """Re-map v from vertex height so torn hems don't shear the painted emblems."""
        for f in self.bm.faces:
            for l in f.loops:
                l[self.uv].uv.y = (l.vert.co.z - zbot) / (ztop - zbot)

    def build(self, parent=None, smooth_angle=50, solidify=None, location=(0, 0, 0)):
        # geometry is authored in the parent's space; location becomes the object's pivot
        bmesh.ops.translate(self.bm, verts=self.bm.verts, vec=-Vector(location))
        bmesh.ops.remove_doubles(self.bm, verts=self.bm.verts, dist=1e-5)
        me = bpy.data.meshes.new(self.name)
        self.bm.to_mesh(me); self.bm.free()
        for m in self.mats: me.materials.append(MATS[m])
        ob = bpy.data.objects.new(self.name, me)
        bpy.context.scene.collection.objects.link(ob)
        ob.location = location
        if parent: ob.parent = parent
        for p in me.polygons: p.use_smooth = True
        try: me.set_sharp_from_angle(angle=math.radians(smooth_angle))
        except Exception: pass
        if solidify:
            mod = ob.modifiers.new('solid', 'SOLIDIFY'); mod.thickness = solidify[0]; mod.offset = solidify[1]
            if len(solidify) > 2: mod.material_offset = solidify[2]
        return ob

def ell(cx, cy, z, rx, ry, n, a0=0.0, a1=TAU, closed=True, rfn=None, zfn=None, p=2.0):
    pts = []
    cnt = n if closed else n
    for k in range(cnt):
        a = a0 + (a1 - a0) * (k / n if closed else k / (n - 1))
        c, s = math.cos(a), math.sin(a)
        if p != 2.0:  # superellipse
            c = math.copysign(abs(c) ** (2 / p), c); s = math.copysign(abs(s) ** (2 / p), s)
        rr = rfn(a) if rfn else 1.0
        z2 = z + (zfn(a) if zfn else 0.0)
        pts.append(Vector((cx + rx * c * rr, cy + ry * s * rr, z2)))
    return pts

def frame(d, hint=Vector((0, 0, 1))):
    d = d.normalized()
    if abs(d.dot(hint)) > 0.95: hint = Vector((0, 1, 0))
    x = d.cross(hint).normalized(); y = d.cross(x).normalized()
    return x, y, d

def tube(part, pts, radii, mat, n=10, caps=True, hint=Vector((0, -1, 0)), squash=1.0, rfn=None):
    """Tube through pts with (rx, ry) radii per point. Rings ordered start->end."""
    rings = []
    for i, p in enumerate(pts):
        d = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])
        x, y, _ = frame(d, hint)
        rx, ry = radii[i] if isinstance(radii[i], tuple) else (radii[i], radii[i] * squash)
        ring = []
        for k in range(n):
            a = TAU * k / n
            rr = rfn(i, a) if rfn else 1.0
            ring.append(Vector(p) + (x * math.cos(a) * rx + y * math.sin(a) * ry) * rr)
        rings.append(ring)
    return part.loft(rings, mat, cap_top=caps, cap_bot=caps)

def V(*a): return Vector(a)

# ---------------------------------------------------------------- the king
def build_king(cname, T, seed):
    rnd = random.Random(seed)
    root = bpy.data.objects.new('king_' + cname, None); bpy.context.scene.collection.objects.link(root)
    K = 'king_' + cname + '_'
    cape_m, lining_m, skirt_m, brooch_m = 'k_cape_' + cname, 'k_lining_' + cname, 'k_skirt_' + cname, 'k_brooch_' + cname
    gem_m = 'k_gem_' + cname

    # ---- body armour (one object)
    B = Part(K + 'armor', ['k_steel', 'k_mail', 'k_brigandine', 'k_leather', 'k_leather_dk', 'k_brass', 'k_visor', brooch_m, 'k_iron'])
    # torso: brigandine over mail, broad chest
    tz = [(1.60, 0.09, 0.10), (1.53, 0.17, 0.15), (1.46, 0.235, 0.18), (1.36, 0.235, 0.185), (1.24, 0.21, 0.17), (1.12, 0.195, 0.155), (1.02, 0.20, 0.16)]
    rings = [ell(0, 0.01, z, rx, ry, 20) for z, rx, ry in tz]
    B.loft(rings, 'k_brigandine', uscale=3, vscale=1.4)
    # gorget
    B.loft([ell(0, 0.0, 1.66, 0.095, 0.10, 14), ell(0, 0.0, 1.57, 0.13, 0.13, 14)], 'k_steel')
    # crossed straps over the chest (X), as on the concept
    def chest_y(x, z):
        rx, ry = 0.235, 0.185
        return 0.01 - ry * math.sqrt(max(0.02, 1 - (x / rx) ** 2)) - 0.012
    for s in (-1, 1):
        pts = [V(s * 0.17, 0, 1.50), V(s * 0.07, 0, 1.32), V(-s * 0.07, 0, 1.17), V(-s * 0.16, 0, 1.06)]
        for p in pts: p.y = chest_y(p.x, p.z)
        tube(B, pts, [(0.035, 0.008)] * 4, 'k_leather', n=4, caps=False, hint=V(0, -1, 0))
        for p in pts[1:3]:
            B.box(p + V(0, -0.008, 0), (0.025, 0.012, 0.025), 'k_brass')
    # brooch: round medallion on the chest
    bc = V(0.0, -0.205, 1.27)
    nb = 20
    ring_f = [bc + V(0.075 * math.cos(TAU * k / nb), 0, 0.075 * math.sin(TAU * k / nb)) for k in range(nb)]
    ring_b = [p + V(0, 0.03, 0) for p in ring_f]
    vs_f = [B.bm.verts.new(p) for p in ring_f]; vs_b = [B.bm.verts.new(p) for p in ring_b]
    cf = B.bm.verts.new(bc + V(0, -0.008, 0))
    for k in range(nb):
        k2 = (k + 1) % nb
        a0, a1 = TAU * k / nb, TAU * k2 / nb
        B.face([cf, vs_f[k], vs_f[k2]], [(0.5, 0.5), (0.5 + 0.5 * math.cos(a0), 0.5 + 0.5 * math.sin(a0)), (0.5 + 0.5 * math.cos(a1), 0.5 + 0.5 * math.sin(a1))], brooch_m)
        B.face([vs_f[k2], vs_f[k], vs_b[k], vs_b[k2]], [(0, 0), (1, 0), (1, 1), (0, 1)], 'k_brass')
    # belt with buckle and pouches
    B.loft([ell(0, 0.01, 1.095, 0.215, 0.17, 20), ell(0, 0.01, 1.005, 0.22, 0.175, 20)], 'k_leather_dk', uscale=4)
    B.box(V(0, -0.18, 1.05), (0.10, 0.025, 0.085), 'k_brass')
    root['belt_z'] = 1.05  # buckle centre, read by assess_kings.py (the bake merges the belt into one texture)
    B.box(V(0, -0.19, 1.05), (0.065, 0.02, 0.05), 'k_leather_dk')
    for s, w in ((-1, 0.10), (1, 0.08)):
        B.box(V(s * 0.17, -0.12, 0.975), (w, 0.06, 0.12), 'k_leather', Matrix.Rotation(s * -0.5, 3, 'Z'))
        B.box(V(s * 0.17, -0.15, 1.025), (w + 0.01, 0.02, 0.04), 'k_leather_dk', Matrix.Rotation(s * -0.5, 3, 'Z'))
    for x in (-0.1, 0.1):
        B.box(V(x, -0.175, 1.05), (0.018, 0.012, 0.018), 'k_brass')
    # thighs (mail) and legs
    for s in (-1, 1):
        hip, knee, ankle = V(s * 0.10, 0.0, 1.00), V(s * 0.16, -0.02, 0.55), V(s * 0.20, 0.0, 0.13)
        tube(B, [hip, (hip + knee) / 2, knee], [0.10, 0.095, 0.08], 'k_mail', n=10)
        # greave: flared plate shin
        tube(B, [knee + V(0, -0.01, -0.02), (knee + ankle) / 2 + V(0, -0.012, 0), ankle + V(0, 0, 0.02)], [(0.085, 0.09), (0.08, 0.09), (0.065, 0.075)], 'k_steel', n=10)
        # knee cop
        kc = knee + V(0, -0.07, 0.01)
        tube(B, [kc + V(0, 0, 0.06), kc, kc + V(0, 0, -0.06)], [(0.035, 0.03), (0.085, 0.05), (0.04, 0.03)], 'k_steel', n=10, hint=V(1, 0, 0))
        B.box(kc + V(s * 0.06, 0.02, 0), (0.03, 0.05, 0.05), 'k_steel')
        # boot / sabaton
        f = ankle + V(0, -0.05, -0.08)
        B.loft([ell(f.x, f.y + 0.06, 0.17, 0.075, 0.09, 12), ell(f.x, f.y - 0.01, 0.10, 0.085, 0.15, 12, p=2.6),
                ell(f.x, f.y - 0.02, 0.0, 0.09, 0.17, 12, p=2.6)], 'k_leather_dk', cap_bot=True)
        for k in range(3):
            B.loft([ell(f.x, f.y - 0.06 - k * 0.035, 0.12 - k * 0.02, 0.08, 0.06, 10, a0=math.pi, a1=TAU, closed=False),
                    ell(f.x, f.y - 0.08 - k * 0.035, 0.07 - k * 0.02, 0.085, 0.06, 10, a0=math.pi, a1=TAU, closed=False)], 'k_steel', closed=False)

    # arms. right (-X) raises the torch, left (+X) holds the sword low
    def arm(s, shoulder, elbow, hand, hand_rot_hint):
        # pauldron: stacked lames
        for k in range(5):     # pauldron lames, the lowest showing below the fur
            c = shoulder + V(s * (0.015 + 0.012 * k), 0, -0.045 * k)
            B.loft([ell(c.x, c.y, c.z + 0.07, 0.07 - k * 0.01, 0.09, 12, rfn=lambda a: 1.0),
                    ell(c.x, c.y, c.z - 0.03, 0.105 - k * 0.004, 0.115 - k * 0.004, 12)], 'k_steel', cap_top=True)
        tube(B, [shoulder, elbow], [0.07, 0.062], 'k_mail', n=10)
        for k in range(3):     # rerebrace: three overlapping flared plates down the upper arm
            a0, a1 = 0.38 + 0.17 * k, 0.52 + 0.17 * k
            tube(B, [shoulder + (elbow - shoulder) * a0, shoulder + (elbow - shoulder) * a1], [0.08, 0.094], 'k_steel', n=10, caps=False)
        # elbow cop
        tube(B, [elbow + (shoulder - elbow).normalized() * 0.05, elbow, elbow + (hand - elbow).normalized() * 0.05], [0.05, 0.08, 0.05], 'k_steel', n=10)
        # vambrace, flared at the wrist
        d = (hand - elbow)
        tube(B, [elbow + d * 0.12, elbow + d * 0.6, hand - d.normalized() * 0.06], [0.06, 0.068, 0.078], 'k_steel', n=10)
        # gauntlet: clenched fist
        g = hand
        x, y, z = frame(d, hand_rot_hint)
        M3 = Matrix((x, y, z)).transposed()
        B.box(g, (0.085, 0.095, 0.10), 'k_leather_dk', M3)
        B.box(g + y * -0.01 + z * 0.01, (0.095, 0.10, 0.055), 'k_steel', M3)
        for k in range(4):
            B.box(g + x * (-0.03 + k * 0.02) + y * -0.05 + z * 0.03, (0.018, 0.03, 0.04), 'k_steel', M3)
    shR, elR, haR = V(-0.25, 0.01, 1.47), V(-0.34, 0.04, 1.15), V(-0.40, -0.24, 1.25)
    shL, elL, haL = V(0.25, 0.01, 1.47), V(0.31, 0.03, 1.13), V(0.34, -0.07, 0.80)
    arm(-1, shR, elR, haR, V(1, 0, 0))
    arm(1, shL, elL, haL, V(0, 0, 1))

    armor = B.build(root, smooth_angle=40)

    # ---- great helm: two angled face plates meeting at a raised centre ridge (beak), eye slit, breaths
    HM = Part(K + 'helm', ['k_steel', 'k_visor', 'k_iron'])
    hz = [(1.885, 0.075, 0.085), (1.87, 0.105, 0.115), (1.82, 0.115, 0.128), (1.72, 0.115, 0.132), (1.64, 0.11, 0.127), (1.60, 0.105, 0.12)]
    def beak(z):  # how far the front point sticks out; eases off under the crown
        return 0.24 * min(1.0, max(0.0, (1.80 - z) / 0.06))
    def keel(a, z):
        return 1.0 + beak(z) * max(0.0, -math.sin(a)) ** 1.6
    def ring_at(z):
        for k in range(len(hz) - 1):
            if hz[k + 1][0] <= z <= hz[k][0]:
                u = (hz[k][0] - z) / (hz[k][0] - hz[k + 1][0])
                return hz[k][1] * (1 - u) + hz[k + 1][1] * u, hz[k][2] * (1 - u) + hz[k + 1][2] * u
        return hz[-1][1], hz[-1][2]
    def surf(z, x):
        """Point on the helm's front surface at height z nearest to the given x."""
        rx, ry = ring_at(z); best = None
        for k in range(181):
            a = -math.pi + math.pi * k / 180
            pt = ell(0, 0.015, z, rx, ry, 1, a0=a, a1=a + 1, closed=True, rfn=lambda q: keel(q, z), p=2.4)[0]
            if best is None or abs(pt.x - x) < abs(best.x - x): best = pt
        return best
    rings = [ell(0, 0.015, z, rx, ry, 12, rfn=lambda a, z=z: keel(a, z), p=2.4) for z, rx, ry in hz]   # 12 flat plates
    HM.loft(rings, 'k_steel', cap_top=True, uscale=2)
    # centre ridge standing ~3.5 cm proud of the face
    rr = []
    for z in (1.865, 1.83, 1.78, 1.72, 1.66, 1.615):
        c = surf(z, 0.0)
        rr.append([c + V(0.013, 0.012, 0), c + V(0, 0.022, 0), c + V(-0.013, 0.012, 0), c + V(0, -0.034, 0)])
    HM.loft(rr, 'k_steel', cap_top=True, cap_bot=True)
    def plate_box(p0, p1, h, d, mat, out=0.004):
        """Thin box laid on the face between two surface points (follows the angled plate)."""
        mid = (p0 + p1) / 2; dx = p1 - p0
        ang = math.atan2(dx.y, dx.x)
        nrm = V(math.sin(ang), -math.cos(ang), 0)
        HM.box(mid + nrm * out, (dx.length, d, h), mat, Matrix.Rotation(ang, 3, 'Z'))
    for s in (-1, 1):
        plate_box(surf(1.775, 0.0), surf(1.775, s * 0.10), 0.016, 0.02, 'k_visor')            # eye slit
        for k in range(4):                                                                   # breaths
            for m in range(2):
                z = 1.70 - k * 0.02
                plate_box(surf(z, s * (0.035 + m * 0.025)), surf(z, s * (0.047 + m * 0.025)), 0.006, 0.02, 'k_visor')
    HM.loft([ell(0, 0.015, 1.80, 0.119, 0.133, 12, rfn=lambda a: keel(a, 1.80), p=2.4),
             ell(0, 0.015, 1.785, 0.121, 0.135, 12, rfn=lambda a: keel(a, 1.785), p=2.4)], 'k_iron')  # brow band
    helm_ob = HM.build(root, smooth_angle=20)


    # ---- fur mantle: ragged pale fur over the shoulders, pointed clumps hanging down
    F = Part(K + 'fur', ['k_fur', 'k_fur_dk'])
    # bulky dark under-pelt that rises around the lower helm (V-neck at the front so it clears the visor)
    prof = [(1.74, 0.135, 0.13), (1.70, 0.19, 0.175), (1.64, 0.235, 0.21), (1.56, 0.255, 0.23), (1.47, 0.25, 0.225)]
    vneck = lambda a, k: -0.08 * max(0.0, -math.sin(a)) ** 2 * (1 - k / 2) if k < 2 else 0.0
    F.loft([ell(0, 0.03, z, rx, ry, 28, zfn=lambda a, k=k: vneck(a, k)) for k, (z, rx, ry) in enumerate(prof)], 'k_fur_dk', uscale=4)
    def collar(a, t):  # a point on the pelt; t 0 = top ring, 1 = rim
        f = t * (len(prof) - 1); k = min(int(f), len(prof) - 2); u = f - k
        z, rx, ry = [prof[k][m] * (1 - u) + prof[k + 1][m] * u for m in range(3)]
        z += vneck(a, k + u) if k + u < 2 else 0.0
        return V(rx * math.cos(a), 0.03 + ry * math.sin(a), z)
    def clump(a, t, L, w, droop, lean=0.0, split=0):
        """A ragged, flat, pointed lock of matted fur, cupped across, tip off-centre."""
        p0 = collar(a, t); out = (collar(a, min(1.0, t + 0.06)) - collar(a, max(0.0, t - 0.06))).normalized()
        nout = V(math.cos(a), math.sin(a), 0)
        out = (out + nout * 0.6).normalized()
        d = (out * (1 - droop) + V(0, 0, -1) * droop).normalized()
        side = V(-math.sin(a), math.cos(a), 0)
        d = (d + side * lean).normalized()
        nrm = d.cross(side).normalized()
        if nrm.dot(nout) < 0: nrm = -nrm
        base = p0 + nrm * 0.01
        tipoff = side * rnd.uniform(-0.35, 0.35) * w
        rings = []
        for s_, ww in ((0.0, 0.7), (0.22, 1.0), (0.5, 0.85), (0.78, 0.45), (1.0, 0.0)):
            ww *= rnd.uniform(0.8, 1.2) if 0 < s_ < 1 else 1.0
            c = base + d * L * s_ + tipoff * s_ * s_ + nrm * 0.035 * math.sin(math.pi * s_) + side * rnd.uniform(-0.01, 0.01)
            ring = []
            for j in range(6):
                q = TAU * j / 6
                ring.append(c + side * w * ww * math.cos(q) + nrm * (0.02 * math.sin(q) + 0.014 * ww * abs(math.cos(q))))
            rings.append(ring)
        F.loft(rings, 'k_fur')
        for _ in range(split):  # extra shorter points beside it: matted locks part unevenly
            clump(a + rnd.uniform(0.08, 0.16) * rnd.choice((-1, 1)), t, L * rnd.uniform(0.5, 0.75), w * rnd.uniform(0.5, 0.7), droop, -lean)
    def is_front_neck(a, t):
        return t < 0.35 and -math.sin(a) > 0.45   # would poke through the visor
    # 16 large locks on the rim, hanging over the shoulders, longer over the chest
    for k in range(16):
        a = TAU * (k + rnd.uniform(-0.45, 0.45)) / 16
        front = max(0.0, -math.sin(a)); sidew = abs(math.cos(a))
        L = rnd.uniform(0.12, 0.21) * (1 + 0.35 * front) * (1 - 0.15 * sidew)
        clump(a, rnd.uniform(0.85, 1.0), L, rnd.uniform(0.09, 0.13), rnd.uniform(0.55, 0.9), rnd.uniform(-0.35, 0.35),
              split=rnd.choice((0, 1, 1, 2)))
    # scattered locks of every size over the whole pelt, in no rows
    placed = 0
    while placed < 110:
        t = rnd.uniform(0.0, 1.0) ** 0.8; a = rnd.uniform(0, TAU)
        if is_front_neck(a, t): continue
        sz = rnd.uniform(0.6, 1.3) * (0.55 + 0.5 * t)
        clump(a, t, rnd.uniform(0.08, 0.17) * sz * 1.3, rnd.uniform(0.06, 0.10) * sz, rnd.uniform(0.25, 0.6) + 0.3 * t,
              rnd.uniform(-0.45, 0.45), split=rnd.choice((0, 0, 0, 1)))
        placed += 1
    fur = F.build(root, smooth_angle=70)

    # ---- cape: hangs from behind the shoulders, flares wide, torn hem
    C = Part(K + 'cape', [cape_m, lining_m])
    R, n = 26, 60
    hem = []
    for k in range(n):
        base = 0.0 if k % 2 == 0 else 0.10
        hem.append(base + rnd.uniform(0, 0.10) + (0.12 if rnd.random() < 0.18 else 0.0))
    rings = []
    for i in range(R):
        s = i / (R - 1)
        wrap = max(0.0, (s - 0.5) / 0.5)        # below the hands the front edges sweep forward
        a0 = math.radians(25 - 60 * wrap ** 1.3); a1 = math.pi - a0
        rx = 0.27 + 0.13 * s ** 1.1
        ry = 0.17 + 0.25 * s ** 1.3            # tucked under the fur at the top, then falls away
        cy = 0.06 + 0.08 * s
        ring = []
        for k in range(n):
            t = k / (n - 1)
            a = a0 + (a1 - a0) * t
            fold = 1 + (0.13 * s ** 1.1) * math.sin(a * 9 + 0.7) + 0.04 * s * math.sin(a * 21 + 2.0)
            ztop = 1.50  # attached under the fur
            zbot = 0.02 + hem[k] + 0.18 * (1 - math.sin(math.pi * t)) ** 2
            z = ztop + (zbot - ztop) * s
            ring.append(V(rx * math.cos(a) * fold, cy + ry * math.sin(a) * fold, z))
        rings.append(ring)
    # ring order goes +X (his left) -> back -> -X: counter-clockwise from above, outward normals
    C.loft(rings, cape_m, closed=False)
    C.v_from_height(1.50, 0.02)
    cape = C.build(root, smooth_angle=89, solidify=(0.012, -1.0, 1))

    # ---- skirt: cloth hanging from the belt to mid-shin, tattered
    S = Part(K + 'skirt', [skirt_m])
    R, n = 10, 40
    hem = [(0.0 if k % 2 == 0 else 0.07) + rnd.uniform(0, 0.08) for k in range(n)]
    rings = []
    for i in range(R):
        s = i / (R - 1)
        ring = []
        for k in range(n):
            a = TAU * k / n - math.pi / 2  # u=0 at the front... shift so front sits at u=0.5
            a = TAU * k / n + math.pi / 2
            fold = 1 + 0.07 * s * math.sin(a * 9)
            rx, ry = 0.225 + 0.10 * s, 0.18 + 0.10 * s
            z = 1.04 + ((0.32 + hem[k] + 0.30 * abs(math.cos(a)) ** 1.5) - 1.04) * s   # hem rises at the sides: knees show
            ring.append(V(rx * math.cos(a) * fold, 0.01 + ry * math.sin(a) * fold, z))
        rings.append(ring)
    S.loft(rings, skirt_m)
    S.v_from_height(1.04, 0.32)
    skirt = S.build(root, smooth_angle=89, solidify=(0.01, -1.0))

    # ---- sword, held low in his left hand, point toward the ground by the right foot
    W = Part(K + 'sword', ['k_blade', 'k_steel', 'k_brass', 'k_leather_dk'])
    tip = V(0.62, -0.30, 0.10)
    d = (tip - haL).normalized()
    x, y, z = frame(d, V(0, -1, 0))
    def at(t, ox=0, oy=0): return haL + d * t + x * ox + y * oy
    # blade: diamond section, fullered
    L = 0.72
    bl = []
    for t, w in ((0.09, 0.041), (0.40, 0.038), (0.62, 0.029), (L - 0.02, 0.006)):
        bl.append([at(t, w, 0), at(t, 0, 0.007), at(t, -w, 0), at(t, 0, -0.007)])
    W.loft(bl, 'k_blade', cap_top=True)
    v_tip = W.bm.verts.new(at(L))
    # crossguard and grip, pommel
    W.box(at(0.07), (0.20, 0.025, 0.025), 'k_steel', Matrix((x, y, d)).transposed())
    tube(W, [at(0.065), at(-0.10)], [0.016, 0.018], 'k_leather_dk', n=6, hint=V(0, -1, 0))
    tube(W, [at(-0.10), at(-0.13)], [0.03, 0.025], 'k_brass', n=8, hint=V(0, -1, 0))
    sword = W.build(root, smooth_angle=30)

    # ---- torch in his raised right hand
    TP = Part(K + 'torch', ['k_wood', 'k_iron'])
    tb, tt = haR + V(0.0, 0.0, -0.22), haR + V(0.02, -0.03, 0.26)
    tube(TP, [tb, haR, tt], [0.022, 0.025, 0.03], 'k_wood', n=8, hint=V(1, 0, 0))
    cup = tt + V(0, 0, 0.0)
    TP.loft([ell(cup.x, cup.y, cup.z + 0.13, 0.08, 0.08, 10), ell(cup.x, cup.y, cup.z + 0.06, 0.065, 0.065, 10), ell(cup.x, cup.y, cup.z, 0.035, 0.035, 10)], 'k_iron', cap_bot=True)
    for k in range(5):
        a = TAU * k / 5
        TP.box(cup + V(math.cos(a) * 0.075, math.sin(a) * 0.075, 0.15), (0.012, 0.012, 0.07), 'k_iron')
    torch = TP.build(root, smooth_angle=40)
    FL = Part(K + 'torch_flame', ['glow_fire', 'glow_firehot'])
    fc = cup + V(0, 0, 0.08)
    # a ragged fire: a body plus several curling tongues of different heights, hot core inside
    tongues = [(0.0, 0.0, 0.36, 0.085, 0.0), (0.045, 0.02, 0.28, 0.05, 0.5), (-0.05, 0.01, 0.31, 0.05, -0.4),
               (0.02, -0.045, 0.24, 0.045, 0.2), (-0.02, 0.05, 0.22, 0.045, -0.2), (0.06, -0.02, 0.17, 0.035, 0.9)]
    for ti, (ox, oy, h, r, lean) in enumerate(tongues):
        rings = []
        for t in (1.0, 0.75, 0.5, 0.25, 0.0):
            w = r * math.sin(math.pi * (0.15 + 0.85 * (1 - t))) ** 0.8 * (1 - t) ** 0.25 if t < 1 else 0.0
            cx = fc.x + ox + lean * 0.12 * t * t
            cy = fc.y + oy + 0.03 * math.sin(t * 5 + ox * 40)
            rings.append(ell(cx, cy, fc.z + h * t, max(w, 0.001), max(w, 0.001) * 0.8, 8,
                             rfn=lambda a, t=t: 1 + 0.25 * math.sin(a * 3 + t * 7)))
        hot_from = 2 if ti == 0 else 3      # lower bands burn yellow-white: the hot core shows from every side
        FL.loft(rings, 'glow_fire', cap_bot='glow_firehot', mat_fn=lambda i, j, hf=hot_from: 'glow_firehot' if i >= hf else 'glow_fire')
    rings = []
    for t, w in ((1.0, 0.0), (0.7, 0.03), (0.35, 0.05), (0.0, 0.045)):
        rings.append(ell(fc.x, fc.y - 0.01, fc.z + 0.22 * t, max(w, 0.001), max(w, 0.001), 8))
    FL.loft(rings, 'glow_firehot', cap_bot=True)
    flame = FL.build(root, smooth_angle=89, location=fc)  # pivot at the base: the game flickers flames by scaling y

    # ---- crown: tall battered gold crown with alternating spikes, sitting on the helm
    crown = bpy.data.objects.new(K + 'crown', None); bpy.context.scene.collection.objects.link(crown)
    crown.parent = root; crown.location = (0, 0.015, 1.84)
    CR = Part(K + 'crown_mesh', ['k_gold', gem_m])
    n = 60
    band = [(0.07, 0.14, 0.152), (0.03, 0.13, 0.142), (-0.02, 0.122, 0.135)]    # z, rx, ry: tall band hugging the helm
    CR.loft([ell(0, 0, z, rx, ry, n) for z, rx, ry in band], 'k_gold', uscale=3)
    def bandpt(a, z):
        u = (0.07 - z) / 0.09
        rx = band[0][1] * (1 - u) + band[2][1] * u; ry = band[0][2] * (1 - u) + band[2][2] * u
        return V(rx * math.cos(a), ry * math.sin(a), z)
    def prong(a, h, prof, flare):
        """A flat gold prong standing on the band, leaning outward; prof = [(v, half width)] base -> tip."""
        base = bandpt(a, 0.068)
        out = V(math.cos(a), math.sin(a), 0); tan = V(-math.sin(a), math.cos(a), 0)
        rings = []
        for v, w in reversed(prof):   # tip first (rings run top -> bottom)
            c = base + V(0, 0, h * v) + out * (flare * h * v ** 1.2)
            rings.append([c + tan * -w, c + out * 0.006, c + tan * w])
        CR.loft(rings, 'k_gold', closed=False)
    spear = [(0.0, 0.05), (0.45, 0.03), (0.6, 0.03), (0.8, 0.017), (1.0, 0.002)]   # broad triangle, spear tip
    small = [(0.0, 0.022), (0.5, 0.012), (1.0, 0.002)]
    for k in range(5):
        a = -math.pi / 2 + TAU * k / 5                # first prong faces the front (-Y)
        prong(a, 0.145 if k == 0 else 0.13, spear, 0.27)
        prong(a + TAU / 10, 0.05, small, 0.2)
    def stud(c, r, mat):
        rings = [ell(c.x, c.y, c.z + r * math.cos(math.pi * t / 4), r * math.sin(math.pi * t / 4) + 1e-4, r * math.sin(math.pi * t / 4) + 1e-4, 8)
                 for t in range(5)]
        CR.loft(rings, mat)
    for k in range(10):                              # round studs on the band: gems under the prongs
        a = -math.pi / 2 + TAU * k / 10
        p = bandpt(a, 0.025) + V(math.cos(a), math.sin(a), 0) * 0.006
        stud(p, 0.016 if k % 2 == 0 else 0.011, gem_m if k % 2 == 0 else 'k_gold')
    crown_mesh = CR.build(crown, smooth_angle=35, solidify=(0.012, 0.0))
    return root


# ---------------------------------------------------------------- painted wear bake
WEAR = {'helm': 0.6, 'armor': 0.6, 'sword': 0.7, 'torch': 0.35, 'crown_mesh': 0.5, 'fur': 0.0}
AO = {'fur': (0.035, 0.68)}   # per part: occlusion distance, darkest value; default (0.10, 0.35)
BAKE_SIZE = {'helm': 512, 'armor': 1024, 'fur': 1024, 'sword': 512, 'torch': 256, 'crown_mesh': 512}

def prep_bake_material(mat):
    """Route the material's colour through occlusion (dark creases) and edge wear (pale worn edges)
    into an Emission shader, so a Cycles EMIT bake captures the painted look. Done once per material."""
    nt = mat.node_tree; N = nt.nodes; L = nt.links
    if 'bake_wear' in N: return
    bsdf = N.get('Principled BSDF'); out = next(n for n in N if n.type == 'OUTPUT_MATERIAL')
    tex = next((n for n in N if n.type == 'TEX_IMAGE'), None)
    def node(t, **kw):
        n = N.new(t)
        for k, v in kw.items(): setattr(n, k, v)
        return n
    if tex:
        uvn = node('ShaderNodeUVMap', uv_map='UVMap'); L.new(uvn.outputs['UV'], tex.inputs['Vector'])
        col = tex.outputs['Color']
    else:
        rgb = node('ShaderNodeRGB'); rgb.outputs[0].default_value = bsdf.inputs['Base Color'].default_value
        col = rgb.outputs[0]
    ao = node('ShaderNodeAmbientOcclusion', samples=8); ao.name = 'bake_ao'
    aof = node('ShaderNodeMapRange'); aof.name = 'bake_aomin'
    L.new(ao.outputs['AO'], aof.inputs['Value'])
    shaded = node('ShaderNodeMix', data_type='RGBA', blend_type='MULTIPLY'); shaded.inputs['Factor'].default_value = 1.0
    L.new(col, shaded.inputs[6]); L.new(aof.outputs['Result'], shaded.inputs[7])
    bev = node('ShaderNodeBevel', samples=8); bev.inputs['Radius'].default_value = 0.009
    geo = node('ShaderNodeNewGeometry')
    dot = node('ShaderNodeVectorMath', operation='DOT_PRODUCT')
    L.new(bev.outputs['Normal'], dot.inputs[0]); L.new(geo.outputs['True Normal'], dot.inputs[1])
    edge = node('ShaderNodeMapRange'); edge.inputs['From Min'].default_value = 0.93; edge.inputs['From Max'].default_value = 0.76
    L.new(dot.outputs['Value'], edge.inputs['Value'])
    noise = node('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 45.0; noise.inputs['Detail'].default_value = 6.0
    chip = node('ShaderNodeMapRange'); chip.inputs['From Min'].default_value = 0.42; chip.inputs['From Max'].default_value = 0.58
    L.new(noise.outputs['Fac'], chip.inputs['Value'])
    mask = node('ShaderNodeMath', operation='MULTIPLY', use_clamp=True)
    L.new(edge.outputs['Result'], mask.inputs[0]); L.new(chip.outputs['Result'], mask.inputs[1])
    wear = node('ShaderNodeValue'); wear.name = 'bake_wear'
    amt = node('ShaderNodeMath', operation='MULTIPLY', use_clamp=True)
    L.new(mask.outputs['Value'], amt.inputs[0]); L.new(wear.outputs['Value'], amt.inputs[1])
    bright = node('ShaderNodeMix', data_type='RGBA', blend_type='SCREEN'); bright.inputs['Factor'].default_value = 1.0
    L.new(col, bright.inputs[6]); bright.inputs[7].default_value = (0.55, 0.52, 0.47, 1)
    final = node('ShaderNodeMix', data_type='RGBA', blend_type='MIX')
    L.new(amt.outputs['Value'], final.inputs['Factor']); L.new(shaded.outputs[2], final.inputs[6]); L.new(bright.outputs[2], final.inputs[7])
    em = node('ShaderNodeEmission'); L.new(final.outputs[2], em.inputs['Color'])
    L.new(em.outputs['Emission'], out.inputs['Surface'])
    tgt = node('ShaderNodeTexImage'); tgt.name = 'bake_target'

def bake_painted(ob, part):
    scn = bpy.context.scene
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = ob; ob.select_set(True)
    for m in list(ob.modifiers): bpy.ops.object.modifier_apply(modifier=m.name)
    me = ob.data
    for uv in list(me.uv_layers):
        if uv.name != 'UVMap': me.uv_layers.remove(uv)
    buv = me.uv_layers.new(name='bake')
    me.uv_layers.active = buv; buv.active_render = True
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.003)
    bpy.ops.object.mode_set(mode='OBJECT')
    size = BAKE_SIZE[part]
    name = 'k_baked_' + ob.name
    img = bpy.data.images.new(name, size, size, alpha=False)
    for mat in me.materials:
        prep_bake_material(mat)
        N = mat.node_tree.nodes
        N['bake_wear'].outputs[0].default_value = WEAR[part]
        dist, lo = AO.get(part, (0.10, 0.35))
        N['bake_ao'].inputs['Distance'].default_value = dist; N['bake_aomin'].inputs['To Min'].default_value = lo
        N['bake_target'].image = img; N.active = N['bake_target']
    bpy.ops.object.bake(type='EMIT', margin=6)
    path = os.path.join(OUT, 'tex', name + '.png')
    img.filepath_raw = path; img.file_format = 'PNG'; img.save(); img.pack()
    # swap to one plain material on the baked uvs
    newm = material(name, img)
    me.materials.clear(); me.materials.append(newm)
    for p in me.polygons: p.material_index = 0
    me.uv_layers.remove(me.uv_layers['UVMap']); me.uv_layers['bake'].name = 'UVMap'
    print('BAKED', name, size)

def bake_all():
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'; scn.cycles.samples = 12; scn.cycles.device = 'CPU'
    scn.render.bake.use_selected_to_active = False
    for k in ('blue', 'red'):
        for part in ('armor', 'helm', 'fur', 'sword', 'torch', 'crown_mesh'):
            bake_painted(bpy.data.objects['king_%s_%s' % (k, part)], part)

# ---------------------------------------------------------------- main
def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene; scn.name = 'Ringhold_Kings'
    global MATS
    T = paint_textures(random.Random(1))
    MATS = {}
    def add(name, tex=None, color=(1, 1, 1)): MATS[name] = material(name, tex, color)
    for k in ('steel', 'blade', 'mail', 'leather', 'leather_dk', 'brass', 'gold', 'fur', 'fur_dk', 'wood', 'iron', 'visor'):
        add('k_' + k, T[k])
    add('k_brigandine', T['brig'])
    add('glow_fire', None, (1.0, 0.35, 0.05)); add('glow_firehot', None, (1.0, 0.75, 0.35))
    add('k_gem_blue', None, (0.05, 0.08, 0.45)); add('k_gem_red', None, (0.45, 0.02, 0.02))
    kings = []
    for i, cname in enumerate(('blue', 'red')):
        KT = paint_king(cname, 100 + i * 50)
        for key in ('cape', 'lining', 'skirt', 'brooch'):
            add('k_%s_%s' % (key, cname), KT[key])
        root = build_king(cname, KT, 7 + i)
        root.location.x = -0.9 if i == 0 else 0.9
        kings.append(root)
    bpy.context.view_layer.update()
    if '--nobake' not in sys.argv: bake_all()   # painted wear and occlusion, see bake_painted()
    # export: apply modifiers, kings at origin side by side (the game re-centres each root)
    for o in bpy.data.objects: o.select_set(True)
    glb = os.path.join(OUT, 'ringhold_kings.glb')
    kw = dict(filepath=glb, export_format='GLB', use_selection=True, export_apply=True, export_yup=True,
              export_image_format='JPEG', export_materials='EXPORT')
    try: bpy.ops.export_scene.gltf(**kw, export_jpeg_quality=82)
    except TypeError:
        try: bpy.ops.export_scene.gltf(**kw, export_image_quality=82)
        except TypeError: bpy.ops.export_scene.gltf(**kw)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, 'ringhold_kings.blend'), compress=True)
    print('BUILT', glb, os.path.getsize(glb))

if __name__ == '__main__':   # build_town.py imports the helpers above without building the kings
    main()
