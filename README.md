# ASL Interpreter 🤟

Real-time American Sign Language (ASL) recognition from webcam video. MediaPipe extracts body and hand landmarks from each frame, and a PyTorch bidirectional LSTM classifies the motion as a sign. The project started as a 6-phrase recognizer and is being scaled to **240 signs** using two public datasets and a Slurm GPU cluster.

<!-- Add a demo GIF here: ![Demo](docs/demo.gif) -->

## Results

**6-phrase model** (hello, how are you, nice to meet you, please, sorry, thank you), trained on ~1,050 self-recorded videos:

| Metric | Score |
|---|---|
| Test accuracy | **89.4%** |
| Validation accuracy | 89.8% |
| Best classes | hello, please, sorry (F1 ≥ 0.95) |
| Hardest class | nice to meet you (F1 0.71), often confused with *thank you* |

**240-sign model:** in progress. It will be evaluated on ASL Citizen's held-out test signers (people never seen in training). [Add top-1 / top-5 accuracy when done.]

## How it works

```
Webcam (2 s clip) → MediaPipe landmarks → BiLSTM → predicted sign + confidence
```

1. **Recording:** the browser records a 2-second clip and sends it to the backend over a WebSocket.
2. **Feature extraction:** MediaPipe finds landmarks in every frame.
   - 6-phrase model: 21 hand points × (x, y, z) = 63 features per frame, 60 frames per clip.
   - 240-sign model: 25 upper-body + 2 × 21 hand points (67 landmarks), every frame kept, normalized to the signer's body.
3. **Classification:** a 2-layer bidirectional LSTM (256 hidden units) reads the frame sequence forwards and backwards and outputs a probability for each sign.
4. **Response:** the top prediction and its confidence are shown in the browser.

**Training details:** class-balanced sampling, 6 time-series augmentations (time warping, jitter, scaling, translation, frame dropout, landmark occlusion), learning-rate scheduling on validation loss, and early stopping.

## Scaling to 240 signs

| Dataset | Videos | Signs | Used for |
|---|---|---|---|
| [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (Microsoft) | 83,399 | 2,731 | train / val / test (signer-independent split) |
| [WLASL](https://github.com/dxli94/WLASL) | 11,880 | 2,000 | extra training data |

- **Vocabulary:** 240 words found in both datasets, each with at least 30 clips (10,871 clips total), chosen so they combine into everyday sentences. Full list in [`vocab/VOCABULARY.md`](vocab/VOCABULARY.md).
- **Sentence test:** 35 test sentences written in ASL gloss order, e.g. `ME HUNGRY WANT EAT PIZZA`.
- **HPC pipeline:** keypoint extraction runs as a Slurm job array across 5 CPU nodes, then training runs on an NVIDIA RTX 5070 Ti.

## Tech stack

Python · PyTorch · MediaPipe · OpenCV · NumPy · scikit-learn · FastAPI · WebSockets · Slurm

## Project structure

```
backend/                  FastAPI server: WebSocket endpoint, live inference
frontend/                 Web page: webcam capture and results
src/                      Model, data loader, training and evaluation
data/feature_extraction/  Clip manifest, keypoint extraction, Slurm job script
data/wlasl/               WLASL download scripts
scripts/                  Dataset download, recording tool, GPU checks
vocab/                    240-word vocabulary and test sentences
```

## Run the demo locally

Requires **Python 3.11** (MediaPipe 0.10.21 doesn't support 3.13 or later).

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install torch numpy==1.26.4 opencv-python==4.11.0.86 mediapipe==0.10.21 \
            fastapi "uvicorn[standard]" scikit-learn matplotlib

cd backend && python server.py      # starts on localhost:8000
# then open frontend/index.html in your browser and click "Sign to Translate"
```

## Run the 240-sign pipeline (Slurm cluster)

```bash
bash scripts/download_datasets.sh                     # WLASL + ASL Citizen (~48 GB)
python data/feature_extraction/make_manifest.py       # list of clips for the 240 words
sbatch data/feature_extraction/extract.sbatch         # parallel keypoint extraction
```

## Limitations

- Recognizes **isolated signs**, not continuous signing. Sentences are tested sign by sign.
- The 6-phrase model tracks one hand only, which hurts two-handed signs. The 240-sign pipeline tracks both hands and the upper body.
- Facial expressions, which carry grammar in ASL, aren't used yet.

## Team

Built as a team project for [course name / Gatehacks 2025] by:
- [Your name]: [your parts, e.g. HPC pipeline, dataset scale-up, vocabulary design]
- Ritchie Registre

## Data and licenses

Datasets are **not included** in this repository.
- **ASL Citizen** is licensed for non-commercial research only and may not be redistributed.
- **WLASL** is under the Computational Use of Data Agreement (C-UDA).

Don't commit dataset videos or extracted keypoints.
