// Port of supervision/tracker/byte_tracker/matching.py.
//
// `linearSumAssignment` replicates scipy.optimize.linear_sum_assignment via
// the classic potentials + augmenting-path Hungarian algorithm (O(n^3)).
// Cost matrices here are tiny (tracks x detections, both < 50), so clarity
// beats cleverness. Optimal total cost matches scipy exactly; in degenerate
// equal-cost ties the chosen pairing can differ, which downstream code is
// insensitive to (any optimal assignment is equally valid).

/** IoU between two xyxy boxes (arrays/objects of 4 numbers). */
export function boxIou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) {
  const ix1 = Math.max(ax1, bx1);
  const iy1 = Math.max(ay1, by1);
  const ix2 = Math.min(ax2, bx2);
  const iy2 = Math.min(ay2, by2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  if (inter <= 0) return 0;
  const areaA = Math.max(0, ax2 - ax1) * Math.max(0, ay2 - ay1);
  const areaB = Math.max(0, bx2 - bx1) * Math.max(0, by2 - by1);
  const denom = areaA + areaB - inter;
  return denom > 0 ? inter / denom : 0;
}

/**
 * IoU matrix between two lists of xyxy boxes ([[x1,y1,x2,y2], ...]).
 * Mirrors supervision's box_iou_batch.
 */
export function boxIouBatch(boxesA, boxesB) {
  const out = [];
  for (let i = 0; i < boxesA.length; i++) {
    const a = boxesA[i];
    const row = new Float64Array(boxesB.length);
    for (let j = 0; j < boxesB.length; j++) {
      const b = boxesB[j];
      row[j] = boxIou(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
    }
    out.push(row);
  }
  return out;
}

/** cost = 1 - IoU between track tlbrs and detection tlbrs. */
export function iouDistance(aBoxes, bBoxes) {
  const ious = boxIouBatch(aBoxes, bBoxes);
  return ious.map((row) => {
    const r = new Float64Array(row.length);
    for (let j = 0; j < row.length; j++) r[j] = 1 - row[j];
    return r;
  });
}

/** fuse_cost = 1 - (iou * detScore). Mirrors matching.fuse_score. */
export function fuseScore(costMatrix, detScores) {
  return costMatrix.map((row) => {
    const r = new Float64Array(row.length);
    for (let j = 0; j < row.length; j++) {
      const iouSim = 1 - row[j];
      r[j] = 1 - iouSim * detScores[j];
    }
    return r;
  });
}

/**
 * Solve min-cost assignment on a rectangular cost matrix.
 * Returns col4row: for every row index (0..nRows-1) the assigned column, or
 * -1 when nRows > nCols and the row is unassigned. Mirrors scipy in that
 * exactly min(nRows, nCols) pairs are produced with minimal total cost.
 */
export function linearSumAssignment(cost, nRows, nCols) {
  if (nRows === 0 || nCols === 0) return new Int32Array(nRows).fill(-1);
  // The augmenting-path algorithm below requires nRows <= nCols; transpose
  // when needed and map back at the end.
  const transpose = nRows > nCols;
  const n = transpose ? nCols : nRows;
  const m = transpose ? nRows : nCols;
  const at = (i, j) => (transpose ? cost[j][i] : cost[i][j]);

  // u/v potentials, p[j] = row matched to column j (1-based internally).
  const INF = Number.POSITIVE_INFINITY;
  const u = new Float64Array(n + 1);
  const v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1).fill(0);
  const way = new Int32Array(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(INF);
    const used = new Uint8Array(m + 1);
    do {
      used[j0] = 1;
      const i0 = p[j0];
      let delta = INF;
      let j1 = -1;
      for (let j = 1; j <= m; j++) {
        if (used[j]) continue;
        const cur = at(i0 - 1, j - 1) - u[i0] - v[j];
        if (cur < minv[j]) {
          minv[j] = cur;
          way[j] = j0;
        }
        if (minv[j] < delta) {
          delta = minv[j];
          j1 = j;
        }
      }
      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else {
          minv[j] -= delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }

  const result = new Int32Array(nRows).fill(-1);
  for (let j = 1; j <= m; j++) {
    if (p[j] === 0) continue;
    const row = p[j] - 1;
    const col = j - 1;
    if (transpose) result[col] = row;
    else result[row] = col;
  }
  return result;
}

/**
 * Port of matching.linear_assignment: clamp costs above `thresh`, solve,
 * then keep only matches whose (clamped) cost <= thresh.
 *
 * Returns { matches: [[a, b], ...], unmatchedA: number[], unmatchedB: number[] }.
 */
export function linearAssignment(costMatrix, nRows, nCols, thresh) {
  if (nRows === 0 || nCols === 0) {
    return {
      matches: [],
      unmatchedA: Array.from({ length: nRows }, (_, i) => i),
      unmatchedB: Array.from({ length: nCols }, (_, j) => j),
    };
  }
  // Clamp mirrors `cost_matrix[cost_matrix > thresh] = thresh + 1e-4`.
  const clamped = [];
  for (let i = 0; i < nRows; i++) {
    const row = new Float64Array(nCols);
    for (let j = 0; j < nCols; j++) {
      const c = costMatrix[i][j];
      row[j] = c > thresh ? thresh + 1e-4 : c;
    }
    clamped.push(row);
  }
  const col4row = linearSumAssignment(clamped, nRows, nCols);
  const matches = [];
  const matchedA = new Uint8Array(nRows);
  const matchedB = new Uint8Array(nCols);
  for (let i = 0; i < nRows; i++) {
    const j = col4row[i];
    if (j < 0) continue;
    if (clamped[i][j] <= thresh) {
      matches.push([i, j]);
      matchedA[i] = 1;
      matchedB[j] = 1;
    }
  }
  const unmatchedA = [];
  for (let i = 0; i < nRows; i++) if (!matchedA[i]) unmatchedA.push(i);
  const unmatchedB = [];
  for (let j = 0; j < nCols; j++) if (!matchedB[j]) unmatchedB.push(j);
  return { matches, unmatchedA, unmatchedB };
}
