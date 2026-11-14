"""Extract body + both-hand landmarks from every clip in the manifest.

For each video this saves keypoints/<clip_id>.npy with shape (T, 67, 3):
    T  = every frame of the clip (no cutting or padding, so frame sampling
         can be changed later at training time without re-extracting)
    67 = 25 upper-body pose points + 21 left-hand + 21 right-hand points
    3  = x, y, z in MediaPipe's image-relative coordinates
Missing parts (e.g. a hand out of frame) are stored as NaN.
Body-relative normalization is done later in the data loader, from these raw values.

Runs in parallel and can be split across Slurm array tasks:
    python data/feature_extraction/extract_keypoints.py --shard $SLURM_ARRAY_TASK_ID --num-shards 5 --workers 12
Clips that already have an output file are skipped, so a re-run resumes.
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

POSE_POINTS = 25          # MediaPipe pose 0-24: face, shoulders, arms, hands, hips
HAND_POINTS = 21
NUM_POINTS = POSE_POINTS + 2 * HAND_POINTS   # 67
MAX_SIDE = 640            # downscale big frames first; MediaPipe resizes internally anyway

_holistic_mod = None


def _init_worker():
    global _holistic_mod
    import cv2
    cv2.setNumThreads(1)
    import mediapipe as mp
    _holistic_mod = mp.solutions.holistic


def _landmarks(lms, n):
    if lms is None:
        return np.full((n, 3), np.nan, dtype=np.float32)
    return np.array([[p.x, p.y, p.z] for p in lms.landmark[:n]], dtype=np.float32)


def extract(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    # New model per video so tracking state never carries over between clips
    with _holistic_mod.Holistic(static_image_mode=False, model_complexity=1,
                                refine_face_landmarks=False) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if max(h, w) > MAX_SIDE:
                s = MAX_SIDE / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)))
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(np.concatenate([
                _landmarks(res.pose_landmarks, POSE_POINTS),
                _landmarks(res.left_hand_landmarks, HAND_POINTS),
                _landmarks(res.right_hand_landmarks, HAND_POINTS),
            ]))
    cap.release()
    if not frames:
        raise RuntimeError("no frames decoded")
    return np.stack(frames)


def process(job):
    clip_id, video_path, out_path = job
    t = time.time()
    try:
        seq = extract(video_path)
        tmp = out_path + ".tmp.npy"
        np.save(tmp, seq)
        os.replace(tmp, out_path)       # atomic: no half-written files if a job is killed
        hand_frames = int((~np.isnan(seq[:, POSE_POINTS:, 0])).any(axis=1).sum())
        return clip_id, "ok", len(seq), hand_frames, time.time() - t, ""
    except Exception as e:
        return clip_id, "error", 0, 0, time.time() - t, str(e)[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/hpc/gkobea/asl-data")
    ap.add_argument("--manifest", default=None, help="default: <data-root>/manifest.csv")
    ap.add_argument("--out-dir", default=None, help="default: <data-root>/keypoints")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=0, help="only process the first N clips (for testing)")
    args = ap.parse_args()

    manifest = args.manifest or os.path.join(args.data_root, "manifest.csv")
    out_dir = args.out_dir or os.path.join(args.data_root, "keypoints")
    os.makedirs(out_dir, exist_ok=True)

    with open(manifest) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    rows = rows[args.shard::args.num_shards]

    jobs = []
    for r in rows:
        out_path = os.path.join(out_dir, r["clip_id"] + ".npy")
        if not os.path.exists(out_path):
            jobs.append((r["clip_id"], os.path.join(args.data_root, r["path"]), out_path))
    print(f"Shard {args.shard}/{args.num_shards}: {len(rows)} clips, {len(rows) - len(jobs)} already done, "
          f"{len(jobs)} to process with {args.workers} workers", flush=True)
    if not jobs:
        return

    log_path = os.path.join(out_dir, f"_log_shard{args.shard}.csv")
    new_log = not os.path.exists(log_path)
    start, done, errors = time.time(), 0, 0
    with open(log_path, "a", newline="") as logf, Pool(args.workers, initializer=_init_worker) as pool:
        log = csv.writer(logf)
        if new_log:
            log.writerow(["clip_id", "status", "frames", "frames_with_hand", "seconds", "error"])
        for result in pool.imap_unordered(process, jobs, chunksize=4):
            log.writerow(result)
            done += 1
            errors += result[1] != "ok"
            if done % 200 == 0 or done == len(jobs):
                logf.flush()
                rate = done / (time.time() - start)
                eta = (len(jobs) - done) / rate / 60
                print(f"  {done}/{len(jobs)}  {rate:.1f} clips/s  errors={errors}  ~{eta:.0f} min left", flush=True)

    print(f"Done in {(time.time() - start) / 60:.1f} min, {errors} errors (see {log_path})")
    sys.exit(1 if errors == len(jobs) else 0)


if __name__ == "__main__":
    main()
