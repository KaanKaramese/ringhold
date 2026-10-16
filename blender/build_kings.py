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
    col = shade([0.20, 0.20, 0.21], g, 0.55)
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
    col = np.array([0.74, 0.70, 0.62])[None, None, :] * (0.55 + 0.6 * s[..., None])
    col = col * (1 - 0.35 * root[..., None])
    col = col * (0.75 + 0.5 * fbm(n, 6, 4, 28)[..., None])
    T['fur'] = image('k_fur', col)
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
    tz = [(1.60, 0.10, 0.10), (1.53, 0.20, 0.15), (1.46, 0.29, 0.18), (1.34, 0.28, 0.19), (1.20, 0.25, 0.17), (1.06, 0.23, 0.155), (0.96, 0.235, 0.16)]
    rings = [ell(0, 0.01, z, rx, ry, 20) for z, rx, ry in tz]
    B.loft(rings, 'k_brigandine', uscale=3, vscale=1.4)
    # gorget
    B.loft([ell(0, 0.0, 1.66, 0.095, 0.10, 14), ell(0, 0.0, 1.57, 0.13, 0.13, 14)], 'k_steel')
    # crossed straps over the chest (X), as on the concept
    def chest_y(x, z):
        rx, ry = 0.28, 0.19
        return 0.01 - ry * math.sqrt(max(0.02, 1 - (x / rx) ** 2)) - 0.012
    for s in (-1, 1):
        pts = [V(s * 0.20, 0, 1.50), V(s * 0.08, 0, 1.30), V(-s * 0.08, 0, 1.12), V(-s * 0.20, 0, 0.99)]
        for p in pts: p.y = chest_y(p.x, p.z)
        tube(B, pts, [(0.035, 0.008)] * 4, 'k_leather', n=4, caps=False, hint=V(0, -1, 0))
        for p in pts[1:3]:
            B.box(p + V(0, -0.008, 0), (0.025, 0.012, 0.025), 'k_brass')
    # brooch: round medallion on the chest
    bc = V(0.0, -0.21, 1.24)
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
    B.loft([ell(0, 0.01, 1.02, 0.25, 0.175, 20), ell(0, 0.01, 0.93, 0.255, 0.18, 20)], 'k_leather_dk', uscale=4)
    B.box(V(0, -0.19, 0.975), (0.11, 0.025, 0.09), 'k_brass')
    B.box(V(0, -0.20, 0.975), (0.07, 0.02, 0.055), 'k_leather_dk')
    for s, w in ((-1, 0.10), (1, 0.08)):
        B.box(V(s * 0.20, -0.12, 0.90), (w, 0.06, 0.12), 'k_leather', Matrix.Rotation(s * -0.5, 3, 'Z'))
        B.box(V(s * 0.20, -0.15, 0.95), (w + 0.01, 0.02, 0.04), 'k_leather_dk', Matrix.Rotation(s * -0.5, 3, 'Z'))
    for x in (-0.1, 0.1):
        B.box(V(x, -0.185, 0.975), (0.018, 0.012, 0.018), 'k_brass')
    # thighs (mail) and legs
    for s in (-1, 1):
        hip, knee, ankle = V(s * 0.11, 0.0, 0.94), V(s * 0.17, -0.02, 0.52), V(s * 0.21, 0.0, 0.13)
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
        for k in range(3):
            c = shoulder + V(s * (0.02 + 0.025 * k), 0, -0.035 * k)
            B.loft([ell(c.x, c.y, c.z + 0.07, 0.07 - k * 0.01, 0.09, 12, rfn=lambda a: 1.0),
                    ell(c.x, c.y, c.z - 0.03, 0.13 - k * 0.012, 0.14 - k * 0.01, 12)], 'k_steel', cap_top=True)
        tube(B, [shoulder, elbow], [0.07, 0.062], 'k_mail', n=10)
        tube(B, [shoulder + (elbow - shoulder) * 0.35, elbow + (shoulder - elbow) * 0.12], [0.078, 0.07], 'k_steel', n=10, caps=False)
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
    shR, elR, haR = V(-0.30, 0.01, 1.47), V(-0.42, 0.03, 1.14), V(-0.47, -0.25, 1.30)
    shL, elL, haL = V(0.30, 0.01, 1.47), V(0.38, 0.02, 1.12), V(0.40, -0.13, 0.88)
    arm(-1, shR, elR, haR, V(1, 0, 0))
    arm(1, shL, elL, haL, V(0, 0, 1))

    # helm: tall great helm, front keel, eye slit and breaths
    hz = [(1.885, 0.075, 0.085), (1.87, 0.105, 0.115), (1.82, 0.115, 0.128), (1.72, 0.115, 0.132), (1.64, 0.11, 0.127), (1.60, 0.105, 0.12)]
    def keel(a):
        return 1.0 + 0.16 * max(0.0, -math.sin(a)) ** 6
    rings = [ell(0, 0.015, z, rx, ry, 24, rfn=keel, p=2.4) for z, rx, ry in hz]
    B.loft(rings, 'k_steel', cap_top=True, uscale=2)
    # visor slit and breathing holes (dark insets on the face)
    fy = 0.015 - 0.132 * 1.16
    B.box(V(0, fy + 0.012, 1.775), (0.17, 0.03, 0.016), 'k_visor')
    B.box(V(0, fy + 0.006, 1.775), (0.016, 0.04, 0.09), 'k_steel')   # nasal ridge
    B.box(V(0, fy + 0.002, 1.70), (0.02, 0.04, 0.20), 'k_steel')   # vertical keel ridge
    for k in range(5):
        for s in (-1, 1):
            B.box(V(s * (0.035 + 0.008 * k), fy + 0.016 + 0.006 * k, 1.69 - k * 0.018), (0.011, 0.03, 0.006), 'k_visor')
    B.loft([ell(0, 0.015, 1.84, 0.118, 0.131, 24, rfn=keel, p=2.4), ell(0, 0.015, 1.825, 0.12, 0.133, 24, rfn=keel, p=2.4)], 'k_iron')  # brow band
    armor = B.build(root, smooth_angle=40)

    # ---- fur mantle: ragged pale fur over the shoulders, pointed clumps hanging down
    F = Part(K + 'fur', ['k_fur'])
    # base collar following the shoulder slope
    prof = [(1.68, 0.10, 0.10), (1.65, 0.20, 0.18), (1.60, 0.31, 0.26), (1.53, 0.38, 0.30), (1.46, 0.39, 0.31)]
    F.loft([ell(0, 0.03, z, rx, ry, 24) for z, rx, ry in prof], 'k_fur', uscale=4)
    # tufts: flattened pointed clumps on the collar, lying down-and-out, longest at the rim
    def collar(a, t):  # t 0 = neck, 1 = rim
        z = 1.685 - 0.21 * t ** 1.3
        rx, ry = 0.10 + 0.30 * t ** 0.7, 0.10 + 0.22 * t ** 0.7
        return V(rx * math.cos(a), 0.03 + ry * math.sin(a), z)
    for row, (t, cnt, ln) in enumerate(((0.2, 12, 0.09), (0.45, 18, 0.12), (0.7, 22, 0.15), (0.9, 26, 0.18), (1.0, 22, 0.22))):
        for k in range(cnt):
            a = TAU * (k + rnd.uniform(-0.45, 0.45) + row * 0.37) / cnt
            p0 = collar(a, t)
            p1 = collar(a, min(1.0, t + 0.05))
            out = (p1 - p0).normalized()
            front = max(0.0, -math.sin(a))
            d = (out + V(0, 0, -0.25 - 0.75 * t)).normalized()
            L = ln * rnd.uniform(0.5, 1.5) * (1 + 0.3 * front * t)
            w = rnd.uniform(0.06, 0.095) * (0.8 + 0.4 * t)
            side = V(-math.sin(a), math.cos(a), 0)
            nrm = d.cross(side).normalized()
            base = p0 + nrm * 0.02
            bend = nrm * rnd.uniform(0.0, 0.03)
            rings = []
            for s, ww in ((0.0, w), (0.45, w * 0.8), (1.0, 0.0)):
                c = base + d * L * s + bend * s * s
                rings.append([c + side * ww * math.cos(TAU * j / 5) + nrm * ww * 0.6 * math.sin(TAU * j / 5) for j in range(5)])
            F.loft(rings, 'k_fur')
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
        a0 = math.radians(18 - 48 * s ** 0.8); a1 = math.pi - a0
        rx = 0.40 + 0.32 * s ** 1.0
        ry = 0.27 + 0.23 * s ** 1.0
        cy = 0.06 + 0.10 * s
        ring = []
        for k in range(n):
            t = k / (n - 1)
            a = a0 + (a1 - a0) * t
            fold = 1 + (0.13 * s ** 1.1) * math.sin(a * 9 + 0.7) + 0.04 * s * math.sin(a * 21 + 2.0)
            ztop = 1.53  # attached under the fur
            zbot = 0.02 + hem[k] + 0.18 * (1 - math.sin(math.pi * t)) ** 2
            z = ztop + (zbot - ztop) * s
            ring.append(V(rx * math.cos(a) * fold, cy + ry * math.sin(a) * fold, z))
        rings.append(ring)
    # ring order goes +X (his left) -> back -> -X: counter-clockwise from above, outward normals
    C.loft(rings, cape_m, closed=False)
    C.v_from_height(1.53, 0.02)
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
            rx, ry = 0.26 + 0.09 * s, 0.185 + 0.10 * s
            z = 0.97 + ((0.30 + hem[k]) - 0.97) * s
            ring.append(V(rx * math.cos(a) * fold, 0.01 + ry * math.sin(a) * fold, z))
        rings.append(ring)
    S.loft(rings, skirt_m)
    S.v_from_height(0.97, 0.30)
    skirt = S.build(root, smooth_angle=89, solidify=(0.01, -1.0))

    # ---- sword, held low in his left hand, point toward the ground by the right foot
    W = Part(K + 'sword', ['k_blade', 'k_steel', 'k_brass', 'k_leather_dk'])
    tip = V(0.66, -0.50, 0.06)
    d = (tip - haL).normalized()
    x, y, z = frame(d, V(0, -1, 0))
    def at(t, ox=0, oy=0): return haL + d * t + x * ox + y * oy
    # blade: diamond section, fullered
    L = 0.98
    bl = []
    for t, w in ((0.09, 0.034), (0.5, 0.030), (0.85, 0.022), (L - 0.02, 0.006)):
        bl.append([at(t, w, 0), at(t, 0, 0.007), at(t, -w, 0), at(t, 0, -0.007)])
    W.loft(bl, 'k_blade', cap_top=True)
    v_tip = W.bm.verts.new(at(L))
    # crossguard and grip, pommel
    W.box(at(0.07), (0.20, 0.025, 0.025), 'k_steel', Matrix((x, y, d)).transposed())
    tube(W, [at(0.065), at(-0.12)], [0.016, 0.018], 'k_leather_dk', n=6, hint=V(0, -1, 0))
    tube(W, [at(-0.12), at(-0.15)], [0.03, 0.025], 'k_brass', n=8, hint=V(0, -1, 0))
    sword = W.build(root, smooth_angle=30)

    # ---- torch in his raised right hand
    TP = Part(K + 'torch', ['k_wood', 'k_iron'])
    tb, tt = haR + V(0.0, 0.0, -0.24), haR + V(0.02, -0.03, 0.40)
    tube(TP, [tb, haR, tt], [0.022, 0.025, 0.03], 'k_wood', n=8, hint=V(1, 0, 0))
    cup = tt + V(0, 0, 0.0)
    TP.loft([ell(cup.x, cup.y, cup.z + 0.13, 0.08, 0.08, 10), ell(cup.x, cup.y, cup.z + 0.06, 0.065, 0.065, 10), ell(cup.x, cup.y, cup.z, 0.035, 0.035, 10)], 'k_iron', cap_bot=True)
    for k in range(5):
        a = TAU * k / 5
        TP.box(cup + V(math.cos(a) * 0.075, math.sin(a) * 0.075, 0.15), (0.012, 0.012, 0.07), 'k_iron')
    torch = TP.build(root, smooth_angle=40)
    FL = Part(K + 'torch_flame', ['glow_fire', 'glow_firehot'])
    fc = cup + V(0, 0, 0.10)
    # a ragged fire: a body plus several curling tongues of different heights, hot core inside
    tongues = [(0.0, 0.0, 0.40, 0.085, 0.0), (0.045, 0.02, 0.28, 0.05, 0.5), (-0.05, 0.01, 0.31, 0.05, -0.4),
               (0.02, -0.045, 0.24, 0.045, 0.2), (-0.02, 0.05, 0.22, 0.045, -0.2), (0.06, -0.02, 0.17, 0.035, 0.9)]
    for ox, oy, h, r, lean in tongues:
        rings = []
        for t in (1.0, 0.75, 0.5, 0.25, 0.0):
            w = r * math.sin(math.pi * (0.15 + 0.85 * (1 - t))) ** 0.8 * (1 - t) ** 0.25 if t < 1 else 0.0
            cx = fc.x + ox + lean * 0.12 * t * t
            cy = fc.y + oy + 0.03 * math.sin(t * 5 + ox * 40)
            rings.append(ell(cx, cy, fc.z + h * t, max(w, 0.001), max(w, 0.001) * 0.8, 8,
                             rfn=lambda a, t=t: 1 + 0.25 * math.sin(a * 3 + t * 7)))
        FL.loft(rings, 'glow_fire', cap_bot=True)
    rings = []
    for t, w in ((1.0, 0.0), (0.7, 0.03), (0.35, 0.05), (0.0, 0.045)):
        rings.append(ell(fc.x, fc.y - 0.01, fc.z + 0.22 * t, max(w, 0.001), max(w, 0.001), 8))
    FL.loft(rings, 'glow_firehot', cap_bot=True)
    flame = FL.build(root, smooth_angle=89, location=fc)  # pivot at the base: the game flickers flames by scaling y

    # ---- crown: tall battered gold crown with alternating spikes, sitting on the helm
    crown = bpy.data.objects.new(K + 'crown', None); bpy.context.scene.collection.objects.link(crown)
    crown.parent = root; crown.location = (0, 0.015, 1.84)
    CR = Part(K + 'crown_mesh', ['k_gold', gem_m])
    n = 80
    def spike(a):
        t = (a / TAU * 10) % 1.0          # 10 spikes, alternating tall and short
        k = int(a / TAU * 10) % 10
        h = (0.25 if k % 2 == 0 else 0.12) * (1.15 if k == 7 else 1.0)  # k=7 faces front (-Y)
        tri = 1 - abs(t - 0.5) * 2
        return h * (tri ** 1.1) + 0.012 * math.sin(a * 37)
    def lean(a):
        return 1.0 + 0.04 * math.sin(a * 3 + 1)
    rings = [
        ell(0, 0, 0.0 + 0.0, 0.0, 0.0, n, zfn=lambda a: 0.0, rfn=lambda a: 0),
    ]
    top = ell(0, 0, 0.09, 0.165, 0.18, n, zfn=spike, rfn=lean)
    mid = ell(0, 0, 0.07, 0.14, 0.155, n)
    low = ell(0, 0, -0.02, 0.125, 0.14, n)
    CR.loft([top, mid, low], 'k_gold', uscale=3)
    # spike knobs
    for k in range(0, 10, 2):
        a = TAU * (k + 0.5) / 10
        h = 0.09 + spike(a)
        c = V(0.165 * math.cos(a) * lean(a), 0.18 * math.sin(a) * lean(a), h)
        CR.loft([ell(c.x, c.y, c.z + 0.05, 0.001, 0.001, 6), ell(c.x, c.y, c.z + 0.018, 0.02, 0.02, 6), ell(c.x, c.y, c.z - 0.01, 0.006, 0.006, 6)], 'k_gold')
    for k in range(5):
        a = TAU * (k + 0.5) / 5 + TAU / 20 * 0
        a = TAU * (2 * k + 0.5) / 10
        CR.box(V(0.145 * math.cos(a), 0.16 * math.sin(a), 0.03), (0.024, 0.016, 0.024), gem_m, Matrix.Rotation(a + math.pi / 2, 3, 'Z'))
    crown_mesh = CR.build(crown, smooth_angle=35, solidify=(0.012, 0.0))
    return root

# ---------------------------------------------------------------- main
def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene; scn.name = 'Ringhold_Kings'
    global MATS
    T = paint_textures(random.Random(1))
    MATS = {}
    def add(name, tex=None, color=(1, 1, 1)): MATS[name] = material(name, tex, color)
    for k in ('steel', 'blade', 'mail', 'leather', 'leather_dk', 'brass', 'gold', 'fur', 'wood', 'iron', 'visor'):
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

main()
