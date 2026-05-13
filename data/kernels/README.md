# SPICE kernels

Tracked in this repo so a fresh clone has everything needed for ephemerides
and time-scale conversions without a network fetch.

| File | Size | Purpose | Source |
|---|---|---|---|
| `naif0012.tls` | 5 KB | Leap-second kernel (LSK). Required for UTC ↔ ET. | `naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/` |
| `de440s.bsp` | 32 MB | DE440 short SPK (1849–2150). Sun, planets, Moon barycenters. | `naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/` |
| `pck00011.tpc` | 131 KB | Generic planetary constants (radii, GM hints). | `naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/` |

All three are NASA-produced and public domain.

## Regenerating

Run `pixi run kernels` from the repo root. The script reads
`backend/oamp/spice/manifest.toml`, verifies any committed file against its
pinned sha256, and re-downloads if absent or mismatched.

Bumping a kernel version: update the URL in `manifest.toml`, delete the old
local file, run `pixi run kernels` (it prints the new sha256), paste that
back into `manifest.toml`, and commit both the new file and the manifest.
