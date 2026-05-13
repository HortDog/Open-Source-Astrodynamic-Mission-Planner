# Static web data

Files in `web/public/` are served by Vite at the URL root in both dev and prod
(copied verbatim to `dist/` at build time).

| File | Size | Purpose | Source |
|---|---|---|---|
| `ne_110m_coastline.json` | 237 KB | Natural Earth 110m coastline (GeoJSON). Renderer overlays it on the Earth wireframe. | `github.com/martynafford/natural-earth-geojson` (which mirrors `naturalearthdata.com`) |

## Regenerating

```sh
curl -L -o web/public/data/ne_110m_coastline.json \
  https://cdn.jsdelivr.net/gh/martynafford/natural-earth-geojson@master/110m/physical/ne_110m_coastline.json
```

Bumping to a higher resolution: change `110m` to `50m` (~430 KB, ~10× more
vertices) or `10m` (~3 MB, ~100× more) in the URL above. The renderer will
parse any GeoJSON `FeatureCollection` containing `LineString` /
`MultiLineString` geometry — no code changes needed for a resolution bump.

Natural Earth is public-domain cartographic data; no attribution required.
