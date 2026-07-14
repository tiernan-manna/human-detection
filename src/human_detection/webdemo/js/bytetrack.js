// Port of supervision/tracker/byte_tracker/{core,single_object_track,utils}.py
// INCLUDING the sidecar's single-match activation patch from
// inference_worker.py: `STrack.activate` marks the track is_activated
// immediately, so a brand-new track surfaces on its first match rather than
// supervision's stock two-match requirement. See the long comment in
// inference_worker.py for why.

import { KalmanFilter } from "./kalman.js";
import { boxIouBatch, fuseScore, iouDistance, linearAssignment } from "./matching.js";

export const TrackState = Object.freeze({
  New: 0,
  Tracked: 1,
  Lost: 2,
  Removed: 3,
});

class IdCounter {
  constructor(startId = 0) {
    this.startId = startId;
    this.NO_ID = -1;
    this.reset();
  }
  reset() {
    this._id = this.startId;
  }
  newId() {
    return this._id++;
  }
}

export class STrack {
  /**
   * @param {number[]} tlwh   [top-left x, top-left y, width, height]
   * @param {number} score
   */
  constructor(tlwh, score, minimumConsecutiveFrames, sharedKalman, internalIdCounter, externalIdCounter) {
    this.state = TrackState.New;
    this.isActivated = false;
    this.startFrame = 0;
    this.frameId = 0;

    this._tlwh = tlwh.slice();
    this.kalmanFilter = null;
    this.sharedKalman = sharedKalman;
    this.mean = null;
    this.covariance = null;

    this.score = score;
    this.trackletLen = 0;
    this.minimumConsecutiveFrames = minimumConsecutiveFrames;

    this.internalIdCounter = internalIdCounter;
    this.externalIdCounter = externalIdCounter;
    this.internalTrackId = internalIdCounter.NO_ID;
    this.externalTrackId = externalIdCounter.NO_ID;
  }

  static multiPredict(stracks, sharedKalman) {
    for (const st of stracks) {
      const meanState = st.mean.slice();
      if (st.state !== TrackState.Tracked) meanState[7] = 0;
      const { mean, covariance } = sharedKalman.predict(meanState, st.covariance);
      st.mean = mean;
      st.covariance = covariance;
    }
  }

  activate(kalmanFilter, frameId) {
    this.kalmanFilter = kalmanFilter;
    this.internalTrackId = this.internalIdCounter.newId();
    const { mean, covariance } = this.kalmanFilter.initiate(
      STrack.tlwhToXyah(this._tlwh)
    );
    this.mean = mean;
    this.covariance = covariance;

    this.trackletLen = 0;
    this.state = TrackState.Tracked;
    // Sidecar patch: always activate on first match (supervision stock only
    // does this when frameId === 1).
    this.isActivated = true;

    if (this.minimumConsecutiveFrames === 1) {
      this.externalTrackId = this.externalIdCounter.newId();
    }

    this.frameId = frameId;
    this.startFrame = frameId;
  }

  reActivate(newTrack, frameId) {
    const { mean, covariance } = this.kalmanFilter.update(
      this.mean,
      this.covariance,
      STrack.tlwhToXyah(newTrack.tlwh)
    );
    this.mean = mean;
    this.covariance = covariance;
    this.trackletLen = 0;
    this.state = TrackState.Tracked;
    this.frameId = frameId;
    this.score = newTrack.score;
  }

  update(newTrack, frameId) {
    this.frameId = frameId;
    this.trackletLen += 1;

    const { mean, covariance } = this.kalmanFilter.update(
      this.mean,
      this.covariance,
      STrack.tlwhToXyah(newTrack.tlwh)
    );
    this.mean = mean;
    this.covariance = covariance;
    this.state = TrackState.Tracked;
    if (this.trackletLen === this.minimumConsecutiveFrames) {
      this.isActivated = true;
      if (this.externalTrackId === this.externalIdCounter.NO_ID) {
        this.externalTrackId = this.externalIdCounter.newId();
      }
    }
    this.score = newTrack.score;
  }

  get tlwh() {
    if (this.mean === null) return this._tlwh.slice();
    const ret = [this.mean[0], this.mean[1], this.mean[2], this.mean[3]];
    ret[2] *= ret[3];
    ret[0] -= ret[2] / 2;
    ret[1] -= ret[3] / 2;
    return ret;
  }

  get tlbr() {
    const ret = this.tlwh;
    ret[2] += ret[0];
    ret[3] += ret[1];
    return ret;
  }

  static tlwhToXyah(tlwh) {
    const ret = tlwh.slice();
    ret[0] += ret[2] / 2;
    ret[1] += ret[3] / 2;
    ret[2] /= ret[3];
    return ret;
  }

  static tlbrToTlwh(tlbr) {
    const ret = tlbr.slice();
    ret[2] -= ret[0];
    ret[3] -= ret[1];
    return ret;
  }
}

export class ByteTrack {
  constructor({
    trackActivationThreshold = 0.25,
    lostTrackBuffer = 30,
    minimumMatchingThreshold = 0.8,
    frameRate = 30,
    minimumConsecutiveFrames = 1,
  } = {}) {
    this.trackActivationThreshold = trackActivationThreshold;
    this.minimumMatchingThreshold = minimumMatchingThreshold;

    this.frameId = 0;
    this.detThresh = this.trackActivationThreshold + 0.1;
    this.maxTimeLost = Math.trunc((frameRate / 30.0) * lostTrackBuffer);
    this.minimumConsecutiveFrames = minimumConsecutiveFrames;
    this.kalmanFilter = new KalmanFilter();
    this.sharedKalman = new KalmanFilter();

    this.trackedTracks = [];
    this.lostTracks = [];
    this.removedTracks = [];

    this.internalIdCounter = new IdCounter(0);
    this.externalIdCounter = new IdCounter(1);
  }

  /**
   * Mirror of update_with_detections. `detections` is an array of plain
   * objects with x1/y1/x2/y2/conf. Returns the SUBSET of detections that
   * matched an output track, in original order, with `trackerId` assigned.
   */
  updateWithDetections(detections) {
    const tensors = detections.map((d) => [d.x1, d.y1, d.x2, d.y2, d.conf]);
    const tracks = this.updateWithTensors(tensors);

    if (tracks.length === 0) return [];

    const detBoxes = tensors.map((t) => t.slice(0, 4));
    const trackBoxes = tracks.map((t) => t.tlbr);
    const ious = boxIouBatch(detBoxes, trackBoxes);
    const iouCosts = ious.map((row) => {
      const r = new Float64Array(row.length);
      for (let j = 0; j < row.length; j++) r[j] = 1 - row[j];
      return r;
    });
    const { matches } = linearAssignment(
      iouCosts,
      detBoxes.length,
      trackBoxes.length,
      0.5
    );
    const trackerIds = new Array(detections.length).fill(-1);
    for (const [iDet, iTrack] of matches) {
      trackerIds[iDet] = tracks[iTrack].externalTrackId;
    }
    const out = [];
    for (let i = 0; i < detections.length; i++) {
      if (trackerIds[i] === -1) continue;
      out.push({ ...detections[i], trackerId: trackerIds[i] });
    }
    return out;
  }

  /** tensors: array of [x1, y1, x2, y2, score]. Returns activated tracked tracks. */
  updateWithTensors(tensors) {
    this.frameId += 1;
    const activatedStracks = [];
    const refindStracks = [];
    const lostStracks = [];
    const removedStracks = [];

    const detsHigh = [];
    const detsSecond = [];
    for (const t of tensors) {
      const score = t[4];
      if (score > this.trackActivationThreshold) {
        detsHigh.push(t);
      } else if (score > 0.1) {
        detsSecond.push(t);
      }
    }

    let detections = detsHigh.map(
      (t) =>
        new STrack(
          STrack.tlbrToTlwh(t.slice(0, 4)),
          t[4],
          this.minimumConsecutiveFrames,
          this.sharedKalman,
          this.internalIdCounter,
          this.externalIdCounter
        )
    );

    const unconfirmed = [];
    const trackedStracks = [];
    for (const track of this.trackedTracks) {
      if (!track.isActivated) unconfirmed.push(track);
      else trackedStracks.push(track);
    }

    // Step 2: first association with high-score detections.
    const strackPool = jointTracks(trackedStracks, this.lostTracks);
    STrack.multiPredict(strackPool, this.sharedKalman);
    let dists = iouDistance(
      strackPool.map((t) => t.tlbr),
      detections.map((t) => t.tlbr)
    );
    dists = fuseScore(dists, detections.map((d) => d.score));
    const assoc1 = linearAssignment(
      dists,
      strackPool.length,
      detections.length,
      this.minimumMatchingThreshold
    );

    for (const [iTracked, iDet] of assoc1.matches) {
      const track = strackPool[iTracked];
      const det = detections[iDet];
      if (track.state === TrackState.Tracked) {
        track.update(det, this.frameId);
        activatedStracks.push(track);
      } else {
        track.reActivate(det, this.frameId);
        refindStracks.push(track);
      }
    }

    // Step 3: second association with low-score detections (pure IoU).
    const detectionsSecond = detsSecond.map(
      (t) =>
        new STrack(
          STrack.tlbrToTlwh(t.slice(0, 4)),
          t[4],
          this.minimumConsecutiveFrames,
          this.sharedKalman,
          this.internalIdCounter,
          this.externalIdCounter
        )
    );
    const rTrackedStracks = assoc1.unmatchedA
      .map((i) => strackPool[i])
      .filter((t) => t.state === TrackState.Tracked);
    const dists2 = iouDistance(
      rTrackedStracks.map((t) => t.tlbr),
      detectionsSecond.map((t) => t.tlbr)
    );
    const assoc2 = linearAssignment(
      dists2,
      rTrackedStracks.length,
      detectionsSecond.length,
      0.5
    );
    for (const [iTracked, iDet] of assoc2.matches) {
      const track = rTrackedStracks[iTracked];
      const det = detectionsSecond[iDet];
      if (track.state === TrackState.Tracked) {
        track.update(det, this.frameId);
        activatedStracks.push(track);
      } else {
        track.reActivate(det, this.frameId);
        refindStracks.push(track);
      }
    }
    for (const it of assoc2.unmatchedA) {
      const track = rTrackedStracks[it];
      if (track.state !== TrackState.Lost) {
        track.state = TrackState.Lost;
        lostStracks.push(track);
      }
    }

    // Deal with unconfirmed tracks (single-beginning-frame tracks).
    detections = assoc1.unmatchedB.map((i) => detections[i]);
    let dists3 = iouDistance(
      unconfirmed.map((t) => t.tlbr),
      detections.map((t) => t.tlbr)
    );
    dists3 = fuseScore(dists3, detections.map((d) => d.score));
    const assoc3 = linearAssignment(
      dists3,
      unconfirmed.length,
      detections.length,
      0.7
    );
    for (const [iTracked, iDet] of assoc3.matches) {
      unconfirmed[iTracked].update(detections[iDet], this.frameId);
      activatedStracks.push(unconfirmed[iTracked]);
    }
    for (const it of assoc3.unmatchedA) {
      const track = unconfirmed[it];
      track.state = TrackState.Removed;
      removedStracks.push(track);
    }

    // Step 4: init new stracks.
    for (const iNew of assoc3.unmatchedB) {
      const track = detections[iNew];
      if (track.score < this.detThresh) continue;
      track.activate(this.kalmanFilter, this.frameId);
      activatedStracks.push(track);
    }

    // Step 5: update state.
    for (const track of this.lostTracks) {
      if (this.frameId - track.frameId > this.maxTimeLost) {
        track.state = TrackState.Removed;
        removedStracks.push(track);
      }
    }

    this.trackedTracks = this.trackedTracks.filter(
      (t) => t.state === TrackState.Tracked
    );
    this.trackedTracks = jointTracks(this.trackedTracks, activatedStracks);
    this.trackedTracks = jointTracks(this.trackedTracks, refindStracks);
    this.lostTracks = subTracks(this.lostTracks, this.trackedTracks);
    this.lostTracks.push(...lostStracks);
    this.lostTracks = subTracks(this.lostTracks, this.removedTracks);
    this.removedTracks = removedStracks;
    const dedup = removeDuplicateTracks(this.trackedTracks, this.lostTracks);
    this.trackedTracks = dedup[0];
    this.lostTracks = dedup[1];

    return this.trackedTracks.filter((t) => t.isActivated);
  }
}

function jointTracks(listA, listB) {
  const seen = new Set();
  const result = [];
  for (const track of [...listA, ...listB]) {
    if (!seen.has(track.internalTrackId)) {
      seen.add(track.internalTrackId);
      result.push(track);
    }
  }
  return result;
}

function subTracks(listA, listB) {
  const idsB = new Set(listB.map((t) => t.internalTrackId));
  // Mirror the dict-based dedup in supervision: last occurrence of a
  // duplicate internal id wins, insertion order otherwise preserved.
  const map = new Map();
  for (const track of listA) map.set(track.internalTrackId, track);
  for (const id of idsB) map.delete(id);
  return [...map.values()];
}

function removeDuplicateTracks(tracksA, tracksB) {
  const dist = iouDistance(
    tracksA.map((t) => t.tlbr),
    tracksB.map((t) => t.tlbr)
  );
  const dupA = new Set();
  const dupB = new Set();
  for (let i = 0; i < tracksA.length; i++) {
    for (let j = 0; j < tracksB.length; j++) {
      if (dist[i][j] < 0.15) {
        const timeA = tracksA[i].frameId - tracksA[i].startFrame;
        const timeB = tracksB[j].frameId - tracksB[j].startFrame;
        if (timeA > timeB) dupB.add(j);
        else dupA.add(i);
      }
    }
  }
  return [
    tracksA.filter((_, idx) => !dupA.has(idx)),
    tracksB.filter((_, idx) => !dupB.has(idx)),
  ];
}
