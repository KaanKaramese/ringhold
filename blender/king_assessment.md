# King models vs concept sheet: assessment

Each item is worked on in order and re-assessed after every iteration until it passes.
Proportions are measured by `blender/assess_kings.py` as fractions of **H** (feet to top of helm),
against the same proportions measured on the concept sheet. Visual items are judged from in-game
screenshots next to the matching crop of the concept.

| # | Item | Criterion | How it is checked | Status |
|---|---|---|---|---|
| 1 | Proportions & silhouette | All 8 measured proportions within tolerance (crown height 0.11 H, crown top 1.08 H, fur width 0.34 H, cape hem width from behind 0.44 H, cape top/hem 0.62, belt 0.56 H, sword 0.47 H, flame top 1.02 H) | `assess_kings.py` | **PASS** (8 / 8) |
| 2 | Fur mantle | Reads as matted fur, not feathers: 12–20 large clumps of uneven length, dark gaps between clumps, bright tips; covers the shoulders and the top of the upper arms | Close-up + `assess_kings.py` (lowest side fur 0.70 ± 0.04 H) | **PASS** |
| 3 | Rim light & painted edges | Warm highlight along silhouette edges on the torch side at night; pale worn edges painted on armour plates; shadowed creases | Night screenshot, armour close-up | **PASS** |
| 4 | Helm & crown | Helm: vertical centre ridge sticking out ≥ 3 cm, angled beak-like front, sunk into the fur. Crown: 5 main prongs, flaring outward (top ≥ 1.2× band width), leaf-shaped tips, no ball finials | `assess_kings.py` (ridge ≥ 3 cm, fur above helm bottom, crown flare, 5 prongs) + close-up | **PASS** |
| 5 | Blue stays blue | Blue King's cape reads as navy blue in the final image by day and at night: ≥ 55% blue pixels, mean saturation ≥ 0.30, cloth hue 210–250° | Pixels read back from the game canvas over the projected cape area | **PASS** |
| 6 | Torch flame | Soft glow around the flame, bright hot core, rising sparks; lights the king's torch side | Night screenshot | **PASS** |
| 7 | Limb armour & sword | Pauldrons, elbow and knee plates visible outside the fur and cape; sword blade at least 3.5% of H wide | Screenshot + `assess_kings.py` (blade width) | **PASS** |

## Iteration log

### Item 1: proportions & silhouette
- Baseline: 0 / 8. Crown twice too tall (0.23 H), fur 0.53 H wide, cape a bell 0.86 H wide from behind, belt low, sword too long, flame too high.
- Iteration 1: slimmer torso and shoulders, belt raised to 0.56 H (longer legs), crown halved, fur narrowed, back of cape hangs straight with the front corners sweeping forward below the hands, shorter sword in a lower hand, shorter torch. Result 8 / 8, silhouette checked against the concept front and back views. **PASS**

### Item 2: fur mantle
- Iteration 1: 16 broad locks at the rim + 2 rows. Side fur too low (0.63 H); from behind the cape covered the fur. FAIL
- Iteration 2: cape tucked under the fur, side locks shortened (0.68 H). Close-up: locks in tidy tiers like a petal skirt, thin. FAIL
- Iteration 3: bulky pelt rising round the helm, locks scattered at random. Pelt showed as a smooth black dome, locks too sparse. FAIL
- Iteration 4: mid-grey pelt, 16 large rim locks + 110 scattered locks of varied size and direction, ragged shapes with split tips. Reads as shaggy fur. Too pale, gaps too light. FAIL (close)
- Iteration 5: grey-brown tone, dark roots and gaps, near-white tips, fur width margin restored (0.37 H). In-game day and night: shaggy pointed locks, pale tips, dark gaps. **PASS**

### Item 3: rim light & painted edges
- Iteration 1: rim light added to the shared paint shader (warm, brush-edged highlight on edges facing a point light). Too weak to see. FAIL
- Iteration 2: stronger and crisper. Fur edges glow on the torch side; dark armour still barely shows it. Painted wear bake added (Cycles: occlusion into creases, pale chipped wear on edges, one unique texture per part). Wear landed on every facet seam of the rounded armour. FAIL
- Iteration 3: wear limited to sharp plate edges (20–40°+). Rim made independent of how dark the surface is. Side effect: ground seen at a shallow angle lit up as a pale wedge. FAIL
- Iteration 4: rim restricted to king materials (shader define on `k_*`). Night: warm brushed edges on fur, cape and arm plates on the torch side, pale worn plate edges, dark creases in the fur and at the belt; ground normal again. **PASS** (the helm's own edge glow is subtle at play distance)
- Re-check: the browser had been serving a cached copy of the model, so some earlier in-game views showed an older build. Re-verified with the current build (forced re-download): rim light on the fur, crown and torch side at night, worn plate edges, dark creases. Still **PASS**.

### Item 4: helm & crown
- Iteration 1: helm moved to its own part with a beak-shaped front and a 3.8 cm centre ridge; crown rebuilt as a studded band with 5 spearhead prongs and 5 small points. All measurements pass. Side effect found: the fur occlusion bake turned the fur charcoal; fixed with a gentler, shorter occlusion for the fur only (item 2 restored).
- Iteration 2: crown band too wide compared with the helm; helm read as a smooth cylinder. Band narrowed to hug the helm, beak stopped below the crown, helm built from 12 flat plates. FAIL in-game: crown tips read as knobs, helm a flat black block.
- Iteration 3: prongs widened into broad triangles with a spear tip; painted shading no longer drops the darkest tone to black (darks keep 60% of their light), steel brightened. Verified on the current build: faceted grey helm with a visible ridge, sunk in the fur; broad crown prongs flaring outward on a studded band. **PASS**

### Item 5: blue stays blue
- Baseline (pixels read from the game canvas over the cape): day 25% blue pixels, saturation 0.20, hue 259°; night 0% blue, hue 349° (torchlight + the screen filter turned navy brown). FAIL
- Iteration 1: the paint shader flags navy-painted surfaces in the spare alpha channel; the screen filter repaints flagged pixels navy, keeping their light and some warm rim (gold emblems and the worn hem are not navy, so they keep their colour). First measurement hit a rock because the king stood elsewhere; switched to projecting the cape into the camera. Night: 65% blue, saturation 0.31, hue 256° (too violet). FAIL
- Iteration 2: navy shifted to the concept's hue, less warm blend. Day 75% blue, saturation 0.38, cloth hue 230°; night 70%, 0.37, 232°. **PASS**

### Item 6: torch flame
- Iteration 1: lower bands of every flame tongue burn yellow-white (the hot core was hidden inside the orange tongues); additive glow sprite and a stream of rising sparks added in the game. Glow was a 0.9 m hard-edged brown disc, sparks were big squares rising a metre. FAIL
- Iteration 2: glow halved, sparks made tiny embers with a short rise. Hot base and embers right; glow still showed a round edge after the oil-paint filter. FAIL (close)
- Iteration 3: glow given a smooth gaussian falloff. Night: soft orange halo round the flame, yellow-white base, embers rising, fur and armour on the torch side lit warm. **PASS**

### Item 7: limb armour & sword
- Iteration 1: pauldron lames extended below the fur, three overlapping flared plates down each upper arm, skirt hem raised at the sides so the knee plates show, blade width check added. Blade 0.035 H, exactly on the limit. FAIL (borderline)
- Iteration 2: blade widened to 0.040 H. In-game: plated arms below the fur, knees visible beside the tabard, broad blade. **PASS**

## Result
All 7 items pass; all 14 measurements pass for both kings (`assess_kings.py`).
