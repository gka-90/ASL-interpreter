"""Build the list of clips (manifest) for the chosen vocabulary.

Reads vocab/vocab_240.json plus the WLASL and ASL Citizen label files and writes
manifest.csv with one row per clip:
    clip_id, path, word, label, source, split, signer

Splits:
  - ASL Citizen keeps its official train/val/test split (test signers are never
    seen in training).
  - WLASL has no split and different signers, so all of it goes to train.

Usage (on the HPC):
    python data/feature_extraction/make_manifest.py --data-root /hpc/gkobea/asl-data --vocab vocab/vocab_240.json
"""
import argparse
import csv
import json
import os
import re
from collections import Counter


def norm(gloss):
    """Match names across datasets: 'SAIL1' -> 'sail', 'NOT MIND' -> 'notmind'."""
    return re.sub(r"[^a-z]", "", re.sub(r"\d+$", "", gloss.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/hpc/gkobea/asl-data")
    ap.add_argument("--vocab", default="vocab/vocab_240.json")
    ap.add_argument("--out", default=None, help="default: <data-root>/manifest.csv")
    args = ap.parse_args()

    words = json.load(open(args.vocab))["words"]
    label_of = {w: i for i, w in enumerate(words)}
    rows, missing = [], 0

    # ASL Citizen
    cit_root = os.path.join(args.data_root, "asl_citizen", "ASL_Citizen")
    for split in ["train", "val", "test"]:
        with open(os.path.join(cit_root, "splits", f"{split}.csv")) as f:
            for r in csv.DictReader(f):
                w = norm(r["Gloss"])
                if w not in label_of:
                    continue
                rel = os.path.join("asl_citizen", "ASL_Citizen", "videos", r["Video file"])
                if not os.path.exists(os.path.join(args.data_root, rel)):
                    missing += 1
                    continue
                clip_id = "cit_" + os.path.splitext(r["Video file"])[0].split("-")[0]
                rows.append([clip_id, rel, w, label_of[w], "asl_citizen", split, r["Participant ID"]])

    # WLASL (Hugging Face / FiftyOne copy)
    samples = json.load(open(os.path.join(args.data_root, "wlasl", "samples.json")))["samples"]
    for s in samples:
        w = norm(s["gloss"]["label"])
        if w not in label_of:
            continue
        rel = os.path.join("wlasl", s["filepath"])
        if not os.path.exists(os.path.join(args.data_root, rel)):
            missing += 1
            continue
        clip_id = "wl_" + s["filepath"].replace("/", "_").replace(".mp4", "")
        rows.append([clip_id, rel, w, label_of[w], "wlasl", "train", "wlasl"])

    # Guard against duplicate ids (would overwrite each other's keypoint files)
    ids = Counter(r[0] for r in rows)
    dupes = [i for i, c in ids.items() if c > 1]
    assert not dupes, f"duplicate clip ids: {dupes[:5]}"

    out = args.out or os.path.join(args.data_root, "manifest.csv")
    with open(out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["clip_id", "path", "word", "label", "source", "split", "signer"])
        wr.writerows(rows)

    by_split = Counter((r[4], r[5]) for r in rows)
    per_word = Counter(r[2] for r in rows)
    print(f"Wrote {len(rows)} clips to {out} ({missing} listed but missing on disk)")
    for k, v in sorted(by_split.items()):
        print(f"  {k[0]:12s} {k[1]:5s} {v}")
    print(f"Words with clips: {len(per_word)}/{len(words)}; "
          f"clips per word min {min(per_word.values())}, max {max(per_word.values())}")
    absent = [w for w in words if w not in per_word]
    if absent:
        print("WARNING: no clips for", absent)


if __name__ == "__main__":
    main()
