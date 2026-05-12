// Earth surface-feature data for the wireframe overlay.
//
// - `GRATICULE` is built procedurally (equator + prime/anti-meridian) and is
//   always available without a network round-trip.
// - `fetchEarthCoastline()` pulls the Natural Earth 110m coastline at boot
//   from jsDelivr's mirror of the martynafford/natural-earth-geojson repo.
//   ~96 KB GeoJSON, public domain (Natural Earth licence terms).
//
// All polylines are returned as ordered `[lat, lon]` pairs in degrees with
// open semantics — segments are drawn `i → i + 1` only, no automatic closure.
// GeoJSON natively stores coordinates as `[lon, lat]`; we swap on ingest.
// Equator: 48-segment full circle (closed by repeating the first vertex).
const equator = [];
for (let i = 0; i <= 48; i++)
    equator.push([0, -180 + (360 * i) / 48]);
// Prime meridian: open arc north → south pole through 0° longitude.
const primeMeridian = [];
for (let i = 0; i <= 36; i++)
    primeMeridian.push([90 - (180 * i) / 36, 0]);
// Antimeridian: same shape on the opposite side of the globe.
const antiMeridian = [];
for (let i = 0; i <= 36; i++)
    antiMeridian.push([90 - (180 * i) / 36, 180]);
export const GRATICULE = [
    equator,
    primeMeridian,
    antiMeridian,
];
// Natural Earth 110m coastline. ~2 000 vertices total, 96 KB JSON.
// jsDelivr mirrors GitHub releases with permissive CORS.
const COASTLINE_URL = "https://cdn.jsdelivr.net/gh/martynafford/natural-earth-geojson@master/110m/physical/ne_110m_coastline.json";
export async function fetchEarthCoastline() {
    const r = await fetch(COASTLINE_URL);
    if (!r.ok)
        throw new Error(`coastline ${r.status}`);
    const gj = (await r.json());
    const out = [];
    for (const f of gj.features ?? []) {
        const g = f.geometry;
        if (!g)
            continue;
        if (g.type === "LineString") {
            out.push(g.coordinates
                .map(([lon, lat]) => [lat, lon]));
        }
        else if (g.type === "MultiLineString") {
            for (const sub of g.coordinates) {
                out.push(sub.map(([lon, lat]) => [lat, lon]));
            }
        }
    }
    return out;
}
