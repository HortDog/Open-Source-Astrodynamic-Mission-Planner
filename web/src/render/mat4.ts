// Tiny mat4 helpers for the WebGPU pipeline (WebGPU NDC: Z in [0, 1]).

export type Mat4 = Float32Array<ArrayBuffer>;
export type Vec3 = readonly [number, number, number];

export function mat4(): Mat4 {
  return new Float32Array(16);
}

export function identity(out: Mat4): Mat4 {
  out.fill(0);
  out[0] = 1; out[5] = 1; out[10] = 1; out[15] = 1;
  return out;
}

// Right-handed perspective for WebGPU clip space (Z in [0, 1]).
export function perspective(out: Mat4, fovYRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovYRad / 2);
  const nf = 1 / (near - far);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far * nf;
  out[11] = -1;
  out[14] = far * near * nf;
  return out;
}

// Right-handed look-at; column-major.
export function lookAt(out: Mat4, eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
  const zl = 1 / Math.hypot(zx, zy, zz);
  const fx = zx * zl, fy = zy * zl, fz = zz * zl;

  const xx0 = up[1] * fz - up[2] * fy;
  const xy0 = up[2] * fx - up[0] * fz;
  const xz0 = up[0] * fy - up[1] * fx;
  const xl = 1 / Math.hypot(xx0, xy0, xz0);
  const xx = xx0 * xl, xy = xy0 * xl, xz = xz0 * xl;

  const yx = fy * xz - fz * xy;
  const yy = fz * xx - fx * xz;
  const yz = fx * xy - fy * xx;

  out[0] = xx;  out[1] = yx;  out[2]  = fx;  out[3]  = 0;
  out[4] = xy;  out[5] = yy;  out[6]  = fy;  out[7]  = 0;
  out[8] = xz;  out[9] = yz;  out[10] = fz;  out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(fx * eye[0] + fy * eye[1] + fz * eye[2]);
  out[15] = 1;
  return out;
}

export function multiply(out: Mat4, a: Mat4, b: Mat4): Mat4 {
  const a00 = a[0]!,  a01 = a[1]!,  a02 = a[2]!,  a03 = a[3]!;
  const a10 = a[4]!,  a11 = a[5]!,  a12 = a[6]!,  a13 = a[7]!;
  const a20 = a[8]!,  a21 = a[9]!,  a22 = a[10]!, a23 = a[11]!;
  const a30 = a[12]!, a31 = a[13]!, a32 = a[14]!, a33 = a[15]!;
  for (let i = 0; i < 4; i++) {
    const b0 = b[i * 4 + 0]!, b1 = b[i * 4 + 1]!, b2 = b[i * 4 + 2]!, b3 = b[i * 4 + 3]!;
    out[i * 4 + 0] = a00 * b0 + a10 * b1 + a20 * b2 + a30 * b3;
    out[i * 4 + 1] = a01 * b0 + a11 * b1 + a21 * b2 + a31 * b3;
    out[i * 4 + 2] = a02 * b0 + a12 * b1 + a22 * b2 + a32 * b3;
    out[i * 4 + 3] = a03 * b0 + a13 * b1 + a23 * b2 + a33 * b3;
  }
  return out;
}
