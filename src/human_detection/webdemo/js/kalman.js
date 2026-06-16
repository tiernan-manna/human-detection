// Port of supervision/tracker/byte_tracker/kalman_filter.py.
//
// 8-dim state (x, y, a, h, vx, vy, va, vh) with constant-velocity motion and
// a linear observation model over (x, y, a, h). Matrices are tiny and the
// motion matrix is fixed (identity + dt on the velocity off-diagonal), so
// everything is written with explicit loops over flat Float64Arrays — no
// matrix library needed, and the arithmetic matches numpy float64 closely
// enough for box-level parity.

const NDIM = 4;
const STATE = 8;
const STD_WEIGHT_POSITION = 1 / 20;
const STD_WEIGHT_VELOCITY = 1 / 160;

// mean: Float64Array(8); covariance: Float64Array(64) row-major.

export class KalmanFilter {
  /** Create track state from measurement [x, y, a, h]. */
  initiate(measurement) {
    const mean = new Float64Array(STATE);
    for (let i = 0; i < NDIM; i++) mean[i] = measurement[i];
    const h = measurement[3];
    const std = [
      2 * STD_WEIGHT_POSITION * h,
      2 * STD_WEIGHT_POSITION * h,
      1e-2,
      2 * STD_WEIGHT_POSITION * h,
      10 * STD_WEIGHT_VELOCITY * h,
      10 * STD_WEIGHT_VELOCITY * h,
      1e-5,
      10 * STD_WEIGHT_VELOCITY * h,
    ];
    const cov = new Float64Array(STATE * STATE);
    for (let i = 0; i < STATE; i++) cov[i * STATE + i] = std[i] * std[i];
    return { mean, covariance: cov };
  }

  /** In-place prediction step. */
  predict(mean, covariance) {
    const h = mean[3];
    const std = [
      STD_WEIGHT_POSITION * h,
      STD_WEIGHT_POSITION * h,
      1e-2,
      STD_WEIGHT_POSITION * h,
      STD_WEIGHT_VELOCITY * h,
      STD_WEIGHT_VELOCITY * h,
      1e-5,
      STD_WEIGHT_VELOCITY * h,
    ];

    // mean' = F mean with F = I + E (E[i][i+4] = 1 for i < 4)
    const newMean = new Float64Array(STATE);
    for (let i = 0; i < STATE; i++) {
      newMean[i] = mean[i] + (i < NDIM ? mean[i + NDIM] : 0);
    }

    // cov' = F C F^T + Q. With F = I + E:
    // F C F^T = C + C E^T + E C + E C E^T
    const c = covariance;
    const out = new Float64Array(STATE * STATE);
    for (let i = 0; i < STATE; i++) {
      for (let j = 0; j < STATE; j++) {
        let val = c[i * STATE + j];
        if (j < NDIM) val += c[i * STATE + (j + NDIM)]; // C E^T
        if (i < NDIM) val += c[(i + NDIM) * STATE + j]; // E C
        if (i < NDIM && j < NDIM) val += c[(i + NDIM) * STATE + (j + NDIM)];
        out[i * STATE + j] = val;
      }
    }
    for (let i = 0; i < STATE; i++) out[i * STATE + i] += std[i] * std[i];
    return { mean: newMean, covariance: out };
  }

  /** Project state to measurement space → {mean: Float64Array(4), cov: Float64Array(16)}. */
  project(mean, covariance) {
    const h = mean[3];
    const std = [
      STD_WEIGHT_POSITION * h,
      STD_WEIGHT_POSITION * h,
      1e-1,
      STD_WEIGHT_POSITION * h,
    ];
    const pMean = new Float64Array(NDIM);
    for (let i = 0; i < NDIM; i++) pMean[i] = mean[i];
    const pCov = new Float64Array(NDIM * NDIM);
    for (let i = 0; i < NDIM; i++) {
      for (let j = 0; j < NDIM; j++) {
        pCov[i * NDIM + j] = covariance[i * STATE + j];
      }
      pCov[i * NDIM + i] += std[i] * std[i];
    }
    return { mean: pMean, cov: pCov };
  }

  /** Correction step. measurement: [x, y, a, h]. Returns new {mean, covariance}. */
  update(mean, covariance, measurement) {
    const { mean: pMean, cov: pCov } = this.project(mean, covariance);

    // B = C H^T = first 4 columns of C (8x4).
    const B = new Float64Array(STATE * NDIM);
    for (let i = 0; i < STATE; i++) {
      for (let j = 0; j < NDIM; j++) B[i * NDIM + j] = covariance[i * STATE + j];
    }
    // Solve pCov X = B^T for X (4x8), K = X^T (8x4), via Cholesky.
    const L = cholesky4(pCov);
    const K = new Float64Array(STATE * NDIM);
    const col = new Float64Array(NDIM);
    for (let r = 0; r < STATE; r++) {
      for (let j = 0; j < NDIM; j++) col[j] = B[r * NDIM + j];
      choleskySolve4(L, col); // in-place solve pCov x = col
      for (let j = 0; j < NDIM; j++) K[r * NDIM + j] = col[j];
    }

    const innovation = new Float64Array(NDIM);
    for (let j = 0; j < NDIM; j++) innovation[j] = measurement[j] - pMean[j];

    const newMean = new Float64Array(STATE);
    for (let i = 0; i < STATE; i++) {
      let acc = mean[i];
      for (let j = 0; j < NDIM; j++) acc += K[i * NDIM + j] * innovation[j];
      newMean[i] = acc;
    }

    // newCov = C - K pCov K^T
    const KP = new Float64Array(STATE * NDIM); // K · pCov
    for (let i = 0; i < STATE; i++) {
      for (let j = 0; j < NDIM; j++) {
        let acc = 0;
        for (let k = 0; k < NDIM; k++) acc += K[i * NDIM + k] * pCov[k * NDIM + j];
        KP[i * NDIM + j] = acc;
      }
    }
    const newCov = new Float64Array(STATE * STATE);
    for (let i = 0; i < STATE; i++) {
      for (let j = 0; j < STATE; j++) {
        let acc = 0;
        for (let k = 0; k < NDIM; k++) acc += KP[i * NDIM + k] * K[j * NDIM + k];
        newCov[i * STATE + j] = covariance[i * STATE + j] - acc;
      }
    }
    return { mean: newMean, covariance: newCov };
  }
}

/** Cholesky decomposition of a 4x4 SPD matrix (lower triangular L, LL^T = A). */
function cholesky4(a) {
  const L = new Float64Array(NDIM * NDIM);
  for (let i = 0; i < NDIM; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = a[i * NDIM + j];
      for (let k = 0; k < j; k++) sum -= L[i * NDIM + k] * L[j * NDIM + k];
      if (i === j) {
        L[i * NDIM + j] = Math.sqrt(Math.max(sum, 1e-12));
      } else {
        L[i * NDIM + j] = sum / L[j * NDIM + j];
      }
    }
  }
  return L;
}

/** Solve A x = b in-place given L from cholesky4 (forward + back substitution). */
function choleskySolve4(L, b) {
  for (let i = 0; i < NDIM; i++) {
    let sum = b[i];
    for (let k = 0; k < i; k++) sum -= L[i * NDIM + k] * b[k];
    b[i] = sum / L[i * NDIM + i];
  }
  for (let i = NDIM - 1; i >= 0; i--) {
    let sum = b[i];
    for (let k = i + 1; k < NDIM; k++) sum -= L[k * NDIM + i] * b[k];
    b[i] = sum / L[i * NDIM + i];
  }
}
