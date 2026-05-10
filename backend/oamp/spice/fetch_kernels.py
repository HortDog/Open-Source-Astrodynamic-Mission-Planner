"""Download SPICE kernels listed in manifest.toml into data/kernels/.

Run via `pixi run kernels`.

Each entry in the manifest may pin `sha256` and `size`. When pinned, this
script verifies and exits non-zero on mismatch. When unpinned, it downloads
the file and prints the computed hash so you can paste it into the manifest
(making CI verification possible).
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tomllib
from pathlib import Path
from typing import TypedDict

import httpx

from oamp.spice import kernels_dir


class KernelSpec(TypedDict, total=False):
    name: str
    url: str
    sha256: str
    size: int
    purpose: str


def _manifest_path() -> Path:
    return Path(__file__).with_name("manifest.toml")


def _load_manifest() -> list[KernelSpec]:
    with _manifest_path().open("rb") as f:
        data = tomllib.load(f)
    return list(data.get("kernels", []))


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(
                        f"\r  {dest.name}: {downloaded / 1e6:6.1f} /"
                        f" {total / 1e6:6.1f} MB ({pct:5.1f}%)",
                        end="",
                    )
        print()
    shutil.move(str(tmp), str(dest))


def main() -> int:
    out_dir = kernels_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"target dir: {out_dir}")

    manifest = _load_manifest()
    if not manifest:
        print("manifest.toml has no [[kernels]] entries", file=sys.stderr)
        return 1

    failures: list[str] = []
    for k in manifest:
        name, url = k["name"], k["url"]
        dest = out_dir / name
        expected = k.get("sha256")

        if dest.exists() and expected:
            actual = _sha256_of(dest)
            if actual == expected:
                print(f"[ok ] {name} (verified)")
                continue
            print(
                f"[!! ] {name}: sha256 mismatch "
                f"(have {actual[:16]}…, want {expected[:16]}…) — redownloading"
            )
            dest.unlink()

        if not dest.exists():
            print(f"[get] {name} <- {url}")
            try:
                _download(url, dest)
            except Exception as e:
                print(f"[err] {name}: {e}", file=sys.stderr)
                failures.append(name)
                continue

        actual = _sha256_of(dest)
        size = dest.stat().st_size
        if expected:
            if actual != expected:
                print(f"[err] {name}: sha256 mismatch after download "
                      f"(got {actual}, want {expected})", file=sys.stderr)
                failures.append(name)
            else:
                print(f"[ok ] {name} (verified, {size:,} bytes)")
        else:
            print(f"[new] {name}: sha256 = {actual}  size = {size}")
            print(f"      (paste into manifest.toml under the {name!r} entry to lock)")

    if failures:
        print(f"\n{len(failures)} kernel(s) failed: {failures}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
