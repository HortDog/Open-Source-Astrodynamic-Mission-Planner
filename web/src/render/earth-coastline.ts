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

type LL = readonly [number, number];

// Equator: 48-segment full circle (closed by repeating the first vertex).
const equator: LL[] = [];
for (let i = 0; i <= 48; i++) equator.push([0, -180 + (360 * i) / 48] as const);

// Prime meridian: open arc north → south pole through 0° longitude.
const primeMeridian: LL[] = [];
for (let i = 0; i <= 36; i++) primeMeridian.push([90 - (180 * i) / 36, 0] as const);

// Antimeridian: same shape on the opposite side of the globe.
const antiMeridian: LL[] = [];
for (let i = 0; i <= 36; i++) antiMeridian.push([90 - (180 * i) / 36, 180] as const);

export const GRATICULE: ReadonlyArray<ReadonlyArray<LL>> = [
  equator,
  primeMeridian,
  antiMeridian,
];

// Natural Earth 110m coastline. ~2 000 vertices total, 96 KB JSON.
// jsDelivr mirrors GitHub releases with permissive CORS.
const COASTLINE_URL =
  "https://cdn.jsdelivr.net/gh/martynafford/natural-earth-geojson@master/110m/physical/ne_110m_coastline.json";

type GeoJsonGeom =
  | { type: "LineString"; coordinates: Array<[number, number]> }
  | { type: "MultiLineString"; coordinates: Array<Array<[number, number]>> }
  | { type: string };

type GeoJsonFC = {
  features?: Array<{ geometry?: GeoJsonGeom }>;
};

export async function fetchEarthCoastline(): Promise<LL[][]> {
  const r = await fetch(COASTLINE_URL);
  if (!r.ok) throw new Error(`coastline ${r.status}`);
  const gj = (await r.json()) as GeoJsonFC;
  const out: LL[][] = [];
  for (const f of gj.features ?? []) {
    const g = f.geometry;
    if (!g) continue;
    if (g.type === "LineString") {
      out.push((g as { coordinates: Array<[number, number]> }).coordinates
        .map(([lon, lat]) => [lat, lon] as const));
    } else if (g.type === "MultiLineString") {
      for (const sub of (g as { coordinates: Array<Array<[number, number]>> }).coordinates) {
        out.push(sub.map(([lon, lat]) => [lat, lon] as const));
      }
    }
  }
  return out;
}
