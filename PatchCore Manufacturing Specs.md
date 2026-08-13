# PatchCore Manufacturing Installation Anomaly Detection — Technical Specification

## 1. Objective

Build a vision-based anomaly detection system for manufacturing installation verification.

The initial scope focuses on **one installation step**.

The system receives an image of the completed installation step and determines whether the visual state is consistent with previously observed correct installations.

The first version only needs to answer:

- **PASS** — installation appears normal
- **ANOMALY** — installation differs significantly from the learned normal state

The primary target anomaly is:

- missing installed item

Future versions may additionally detect:

- wrong component
- incorrect orientation
- incomplete insertion
- incorrect position
- damaged component
- unexpected component
- foreign object

No bounding-box, segmentation-mask, or per-component annotations are required for training.

---

# 2. Core Design Principle

PatchCore is a **one-class / unsupervised industrial anomaly detection** approach.

The model is trained only on images representing correct installation states.

```text
Correct installation images
          │
          ▼
Pretrained feature extractor
          │
          ▼
Local patch embeddings
          │
          ▼
Representative feature memory bank
          │
          ▼
     PatchCore model
```

During inference:

```text
New installation image
          │
          ▼
Same feature extractor
          │
          ▼
Local patch embeddings
          │
          ▼
Compare against normal memory bank
          │
          ▼
   anomaly distance
          │
      ┌───┴────┐
      │        │
    PASS     ANOMALY
```

PatchCore can also generate an anomaly heatmap indicating the image regions most different from the learned normal distribution.

---

# 3. High-Level System Architecture

```text
┌─────────────────────────────────────┐
│            Operator / UI            │
│                                     │
│   Start inspection for Step 01      │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│          Image Acquisition          │
│                                     │
│ Camera / uploaded production image  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│       Image Quality Validation      │
│                                     │
│ - resolution                        │
│ - blur                              │
│ - exposure                          │
│ - optional framing validation       │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│        Image Preprocessing          │
│                                     │
│ - crop inspection ROI               │
│ - resize                            │
│ - normalization                     │
│ - optional image registration       │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│          PatchCore Model            │
│                                     │
│ pretrained CNN backbone             │
│              │                      │
│              ▼                      │
│ feature extraction                  │
│              │                      │
│              ▼                      │
│ nearest-neighbor comparison         │
│ against normal memory bank          │
└──────────────────┬──────────────────┘
                   │
          ┌────────┴─────────┐
          ▼                  ▼
┌─────────────────┐   ┌──────────────────┐
│ Anomaly Score   │   │ Anomaly Heatmap  │
└────────┬────────┘   └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    ▼
┌─────────────────────────────────────┐
│          Decision Engine            │
│                                     │
│ score < threshold      → PASS       │
│ score >= threshold     → ANOMALY    │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│              UI Result              │
│                                     │
│ PASS                                │
│ or                                  │
│ ANOMALY + highlighted region        │
└─────────────────────────────────────┘
```

---

# 4. Initial Scope

## 4.1 Included

The first proof of concept will support:

- one equipment type
- one installation step
- one fixed or mostly fixed camera view
- one inspection ROI
- PatchCore anomaly detection
- normal-only model training
- anomaly score generation
- anomaly heatmap generation
- configurable threshold
- PASS / ANOMALY result
- offline image inference
- optional live-camera capture

---

## 4.2 Not Included Initially

The first version will not attempt to:

- identify the missing component by name
- classify anomaly types
- inspect multiple assembly steps simultaneously
- reason about installation sequence
- perform object detection
- perform segmentation
- perform OCR
- perform 3D reconstruction
- automatically correct large camera-angle differences
- automatically learn from operator feedback

These can be added later.

---

# 5. Deployment Concept

The production system should eventually support multiple installation steps.

Recommended structure:

```text
                     Shared Application
                           │
                           ▼
                    Current Step ID
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      Step 01 Model   Step 02 Model   Step 03 Model
           │               │               │
           ▼               ▼               ▼
       PatchCore        PatchCore        PatchCore
     memory bank      memory bank      memory bank
```

Each installation step maintains its own definition of "normal".

Example:

```text
models/
├── equipment_A/
│   ├── step_01/
│   │   ├── model.pt
│   │   ├── metadata.json
│   │   └── threshold.json
│   │
│   ├── step_02/
│   │   ├── model.pt
│   │   ├── metadata.json
│   │   └── threshold.json
│   │
│   └── step_03/
│       ├── model.pt
│       ├── metadata.json
│       └── threshold.json
```

For the first POC, only:

```text
equipment_A/step_01
```

needs to be implemented.

---

# 6. Dataset Strategy

## 6.1 Training Principle

Only **correctly completed installations** should be used for PatchCore training.

Training images should represent the acceptable production variation of the step.

Example:

```text
dataset/
└── step_01/
    ├── train/
    │   └── good/
    │       ├── img_0001.jpg
    │       ├── img_0002.jpg
    │       ├── img_0003.jpg
    │       └── ...
    │
    ├── validation/
    │   ├── good/
    │   └── bad/
    │
    └── test/
        ├── good/
        └── bad/
```

No bounding boxes or masks are required.

---

# 7. Recommended Dataset Size

For the first POC:

## Training

Target:

```text
100–300 correct images
```

Minimum experimental starting point:

```text
50 correct images
```

Preferred:

```text
150–300 correct images
```

The diversity of the training data is more important than raw image count.

---

## Validation

Recommended:

```text
20–50 correct images
30–100 deliberately abnormal images
```

Abnormal validation examples should preferably include the failure conditions that matter most.

For example:

```text
validation/bad/
├── missing_screw/
├── missing_bracket/
├── missing_connector/
└── multiple_missing/
```

These subfolder names are only for engineering analysis.

PatchCore does not use them during training.

---

## Test

Keep a completely independent final test set.

Recommended:

```text
50+ correct images
50+ abnormal images
```

Test images should preferably come from different:

- manufacturing units
- capture sessions
- operators
- dates

than the training images.

---

# 8. Data Collection Requirements

The training data should capture realistic acceptable variation.

Examples include:

- different physical machines or assemblies
- minor part appearance variations
- acceptable cable movement
- acceptable reflections
- small lighting variation
- slight camera-position variation
- different operators
- different production lots

Do not collect 200 almost-identical consecutive frames and treat them as 200 independent training samples.

A better dataset is:

```text
Unit 001 → 3 images
Unit 002 → 3 images
Unit 003 → 3 images
...
```

rather than:

```text
Unit 001 → 200 consecutive frames
```

---

# 9. Dataset Leakage Prevention

Do not randomly split individual video frames across train, validation, and test.

For example, avoid:

```text
Frame 001 → train
Frame 002 → train
Frame 003 → test
Frame 004 → validation
```

because these images may be nearly identical.

Prefer splitting at the physical-unit or capture-session level.

Example:

```text
Units 001–070 → train

Units 071–085 → validation

Units 086–100 → test
```

---

# 10. Camera Requirements

For the first POC, the camera setup should be as controlled as practical.

Preferred:

```text
fixed camera
+
fixed mounting position
+
fixed focal length
+
fixed exposure
+
fixed illumination
+
fixed inspection area
```

This significantly reduces false anomaly detections.

---

# 11. Recommended Camera Resolution

Starting recommendation:

```text
1920 × 1080
```

or higher if very small components must be detected.

The full image should not necessarily be passed directly into PatchCore.

Instead:

```text
Camera image
     │
     ▼
Inspection ROI
     │
     ▼
Model input
```

Example:

```text
1920 × 1080 camera image

          ↓ crop

800 × 800 inspection region

          ↓ resize

512 × 512 model input
```

Actual resolution should be validated experimentally.

---

# 12. Region of Interest

The first version should use a fixed ROI.

Example:

```text
Full machine image

┌────────────────────────────────────┐
│                                    │
│                                    │
│       ┌───────────────────┐        │
│       │                   │        │
│       │   STEP 01 ROI     │        │
│       │                   │        │
│       └───────────────────┘        │
│                                    │
└────────────────────────────────────┘
```

Only the relevant installation area should be inspected.

Benefits:

- removes irrelevant visual variation
- increases effective object resolution
- decreases memory usage
- decreases inference time
- reduces false anomalies

---

# 13. ROI Configuration

Store ROI configuration separately from the model.

Example:

```json
{
  "equipment": "equipment_A",
  "step": "step_01",
  "roi": {
    "x": 420,
    "y": 180,
    "width": 850,
    "height": 720
  }
}
```

This allows the camera/ROI setup to be changed without changing application code.

---

# 14. Preprocessing Pipeline

Recommended initial pipeline:

```text
Raw camera frame
        │
        ▼
ROI crop
        │
        ▼
Resize
        │
        ▼
RGB conversion
        │
        ▼
PatchCore preprocessing
        │
        ▼
Inference
```

Avoid aggressive augmentation during inference.

---

# 15. Image Registration

Image registration should **not be implemented initially** if the camera is fixed.

Version 1:

```text
fixed camera
   ↓
fixed ROI
   ↓
PatchCore
```

If false anomalies caused by camera movement become significant, add:

```text
Input image
     │
     ▼
feature matching
     │
     ▼
homography / registration
     │
     ▼
aligned image
     │
     ▼
PatchCore
```

Possible future registration methods:

- OpenCV ORB
- SIFT
- LightGlue
- LoFTR
- EfficientLoFTR

---

# 16. Image Quality Gate

PatchCore should not inspect images that are obviously unusable.

Recommended checks:

## Blur

Use a simple focus metric such as:

```text
variance of Laplacian
```

Example:

```text
blur score < threshold

→ reject image
→ request recapture
```

---

## Exposure

Check whether too much of the image is:

- saturated white
- saturated black

---

## Resolution

Reject unexpectedly small input images.

---

## Optional Future Checks

- camera pose
- framing
- occlusion
- motion blur
- glare

---

# 17. PatchCore Model

Recommended implementation:

```text
Anomalib
+
PyTorch
+
PatchCore
```

Initial backbone recommendation:

```text
wide_resnet50_2
```

Typical PatchCore feature layers:

```text
layer2
layer3
```

Initial coreset sampling ratio:

```text
0.1
```

These should be treated as baseline values and benchmarked.

---

# 18. PatchCore Internal Architecture

Simplified:

```text
Input image
      │
      ▼
Pretrained WideResNet50
      │
      ├── layer2 features
      │
      └── layer3 features
      │
      ▼
Feature aggregation
      │
      ▼
Patch embeddings
      │
      ▼
Coreset sampling
      │
      ▼
Normal memory bank
```

Inference:

```text
Test image
     │
     ▼
Patch embeddings
     │
     ▼
Nearest-neighbor search
     │
     ▼
Distance to normal features
     │
     ├───────────────┐
     ▼               ▼
image score      patch scores
                     │
                     ▼
                anomaly map
```

---

# 19. Training Workflow

## Step 1 — Collect good images

```text
train/good/
```

Only correctly completed installations.

---

## Step 2 — Preprocess

Each image:

```text
crop ROI
↓
resize
↓
normalize
```

---

## Step 3 — Extract Features

The pretrained backbone processes every training image.

```text
image
  ↓
WideResNet
  ↓
layer2 + layer3 features
```

---

## Step 4 — Generate Patch Embeddings

Spatial features are converted into local patch descriptors.

Each descriptor represents a portion of the installation image.

---

## Step 5 — Build Memory Bank

Collect patch descriptors from all normal training images.

Example conceptual size:

```text
150 images
×
~1,000 patches/image

≈ 150,000 patch features
```

Actual number depends on feature-map resolution.

---

## Step 6 — Coreset Sampling

Keeping every patch can consume unnecessary memory.

PatchCore selects a representative subset.

Example:

```text
150,000 patches

      ↓ 10% coreset

15,000 representative patches
```

This becomes the normal feature memory bank.

---

## Step 7 — Save Model Artifact

Store:

```text
backbone configuration
feature configuration
memory bank
preprocessing parameters
model metadata
```

---

# 20. Threshold Calibration

PatchCore outputs a continuous anomaly score.

Example:

```text
Good image       → 0.11
Good image       → 0.17
Good image       → 0.23

Missing screw    → 0.68
Missing bracket  → 0.74
```

A decision threshold must be selected.

Example:

```text
threshold = 0.42
```

Then:

```text
score < 0.42
→ PASS

score >= 0.42
→ ANOMALY
```

The exact threshold must be determined using validation data.

---

# 21. Threshold Selection Priority

In manufacturing, the most dangerous error is:

```text
bad installation
      ↓
model predicts PASS
```

Therefore threshold optimization should prioritize **low false-accept rate**.

The tradeoff is potentially higher false rejects.

Example target:

```text
False Accept Rate < 1%
```

while keeping:

```text
False Reject Rate
```

at an operationally acceptable level.

---

# 22. Optional Three-Level Decision

After the initial binary POC, consider:

```text
score < T1
→ PASS

T1 <= score < T2
→ REVIEW

score >= T2
→ ANOMALY
```

Example:

```text
0.00–0.30
PASS

0.30–0.45
MANUAL REVIEW

> 0.45
ANOMALY
```

This is safer around borderline images.

---

# 23. Inference Workflow

Production inference:

```text
Operator completes Step 01
        │
        ▼
Capture image
        │
        ▼
Image quality gate
        │
      invalid
        │
        └────► request recapture
        │
       valid
        ▼
Crop ROI
        │
        ▼
Resize / preprocess
        │
        ▼
Load Step 01 PatchCore model
        │
        ▼
Predict
        │
        ├──────────────┐
        ▼              ▼
anomaly score      anomaly map
        │              │
        └──────┬───────┘
               ▼
        Compare threshold
               │
        ┌──────┴──────┐
        ▼             ▼
      PASS         ANOMALY
```

---

# 24. API Response

Recommended inference response:

```json
{
  "equipment_id": "equipment_A",
  "step_id": "step_01",
  "result": "ANOMALY",
  "anomaly_score": 0.672,
  "threshold": 0.420,
  "model_version": "1.0.0",
  "processing_time_ms": 84
}
```

Optional:

```json
{
  "heatmap_path": "/results/abc123_heatmap.png"
}
```

---

# 25. Heatmap Output

When an anomaly is detected, generate a visualization:

```text
Original image
       +
anomaly heatmap
       ↓
overlay result
```

Example UI:

```text
┌─────────────────────────────┐
│                             │
│       installation          │
│                             │
│             ███             │
│            █RED█            │
│             ███             │
│                             │
└─────────────────────────────┘

ANOMALY DETECTED
Score: 0.67
Threshold: 0.42
```

The heatmap is diagnostic only.

It should not initially be interpreted as a component identity.

---

# 26. Evaluation Metrics

Overall classification metrics:

- accuracy
- precision
- recall
- F1
- AUROC
- AUPR

Manufacturing-specific metrics should additionally include:

## False Accept Rate

```text
actual bad
+
predicted good
```

This should be treated as the most critical metric.

---

## False Reject Rate

```text
actual good
+
predicted anomaly
```

This determines nuisance alarms and operator acceptance.

---

## Detection Rate by Failure Type

Even though failure types are not used during model training, they should be recorded during validation.

Example:

```text
Missing screw          100%
Missing connector       98%
Missing bracket        100%
Two items missing      100%
```

This provides much more useful engineering information than overall accuracy alone.

---

# 27. Experiment Tracking

Every model run should record:

```text
model version
dataset version
training date
backbone
input resolution
ROI
coreset ratio
threshold
training image count
validation results
test results
```

Example:

```json
{
  "model_version": "step01_patchcore_v1.2",
  "backbone": "wide_resnet50_2",
  "input_size": [512, 512],
  "coreset_ratio": 0.1,
  "train_images": 184,
  "threshold": 0.417,
  "false_accept_rate": 0.005,
  "false_reject_rate": 0.021
}
```

---

# 28. Model Versioning

Recommended naming:

```text
equipmentA_step01_patchcore_v1.0
equipmentA_step01_patchcore_v1.1
equipmentA_step01_patchcore_v2.0
```

Each model should be associated with:

```text
equipment revision
step revision
camera revision
dataset revision
```

Changing any of these may require revalidation.

---

# 29. Suggested Project Structure

```text
patchcore-installation-inspection/
│
├── README.md
│
├── requirements.txt
│
├── configs/
│   ├── equipment_A/
│   │   └── step_01.yaml
│   │
│   └── model/
│       └── patchcore.yaml
│
├── data/
│   └── equipment_A/
│       └── step_01/
│           ├── train/
│           │   └── good/
│           │
│           ├── validation/
│           │   ├── good/
│           │   └── bad/
│           │
│           └── test/
│               ├── good/
│               └── bad/
│
├── models/
│   └── equipment_A/
│       └── step_01/
│           ├── model.pt
│           ├── metadata.json
│           └── threshold.json
│
├── src/
│   ├── training/
│   │   ├── train_patchcore.py
│   │   └── evaluate.py
│   │
│   ├── inference/
│   │   ├── predictor.py
│   │   └── preprocessing.py
│   │
│   ├── vision/
│   │   ├── roi.py
│   │   ├── quality.py
│   │   └── heatmap.py
│   │
│   ├── api/
│   │   └── app.py
│   │
│   └── utils/
│       └── config.py
│
├── results/
│   ├── experiments/
│   └── inference/
│
└── tests/
```

---

# 30. Step Configuration Example

```yaml
equipment_id: equipment_A

step_id: step_01

step_name: Install bracket assembly

camera:
  input_width: 1920
  input_height: 1080

roi:
  x: 420
  y: 180
  width: 850
  height: 720

preprocessing:
  input_size:
    - 512
    - 512

model:
  type: patchcore
  backbone: wide_resnet50_2

  layers:
    - layer2
    - layer3

  coreset_sampling_ratio: 0.1

decision:
  threshold: 0.42

quality:
  blur_check: true
  minimum_blur_score: 100
```

---

# 31. Software Stack

Recommended initial environment:

```text
Python 3.11 or 3.12

PyTorch

Anomalib

OpenCV

NumPy

Pillow

scikit-learn
```

Optional:

```text
FastAPI
Uvicorn
```

for deployment as an inspection service.

---

# 32. Hardware — Training

PatchCore does not require a high-end training GPU.

Recommended development workstation:

```text
CPU:
8+ modern cores

RAM:
16 GB minimum
32 GB preferred

GPU:
NVIDIA CUDA-capable GPU

VRAM:
8 GB minimum
12 GB preferred

Storage:
20–50 GB available
```

A modern RTX workstation is more than sufficient for the first POC.

---

# 33. Hardware — Deployment

Possible deployment targets:

```text
Industrial PC + NVIDIA GPU
```

or potentially:

```text
CPU-only industrial PC
```

depending on required latency.

Future export formats can include:

```text
ONNX
OpenVINO
```

if required.

---

# 34. Performance Target

For the initial POC:

```text
single image inspection
< 1 second
```

is sufficient.

Later production target:

```text
100–300 ms
```

would provide near-immediate operator feedback.

The exact requirement depends on workflow.

---

# 35. POC Development Phases

## Phase 1 — Dataset Creation

Collect:

```text
150–300 good images

50+ anomaly images
```

for one step.

---

## Phase 2 — Baseline Training

Train PatchCore using:

```text
wide_resnet50_2

layer2
layer3

coreset = 0.1
```

---

## Phase 3 — Evaluation

Measure:

```text
false accept rate
false reject rate
detection rate by missing-item type
```

---

## Phase 4 — Threshold Calibration

Select production-oriented threshold.

Priority:

```text
minimize false PASS
```

---

## Phase 5 — Offline Inference Application

Input:

```text
image file
```

Output:

```text
PASS / ANOMALY
anomaly score
heatmap
```

---

## Phase 6 — Camera Integration

Replace uploaded image with:

```text
camera capture
```

---

## Phase 7 — Production UI

Show:

```text
reference / golden image

live capture

PASS / ANOMALY

anomaly overlay
```

---

# 36. Initial Success Criteria

The POC should be considered successful if it demonstrates that:

1. No component annotations are required for training.

2. PatchCore learns the normal state from good installation images.

3. Missing-component conditions produce significantly larger anomaly scores than correct installations.

4. The anomaly map highlights the approximate missing-component region.

5. An operating threshold can be selected with an acceptably low false-accept rate.

Suggested initial target:

```text
Missing-item detection recall:
>= 95%

False accept rate:
<= 1%

False reject rate:
<= 5%
```

These targets should be adjusted after observing real production variability.

---

# 37. Main Technical Risks

## Camera movement

Large viewpoint changes may appear anomalous.

Mitigation:

```text
fixed camera
or
future image registration
```

---

## Lighting changes

Strong reflections or shadows can create false anomalies.

Mitigation:

```text
controlled illumination
+
training data diversity
```

---

## Acceptable installation variation

Flexible cables or movable parts may vary legitimately.

Mitigation:

```text
include normal variation in training set
```

---

## Very small missing components

A tiny missing screw may occupy too few feature pixels.

Mitigation:

```text
smaller ROI
higher camera resolution
higher model input resolution
```

---

## Background variation

Irrelevant objects can create anomalies.

Mitigation:

```text
tight ROI
controlled workstation
```

---

# 38. Future Extensions

After the initial PatchCore POC, possible improvements include:

```text
PatchCore
   │
   ├── image registration
   │
   ├── multi-ROI inspection
   │
   ├── multiple step models
   │
   ├── EfficientAD comparison
   │
   ├── DINO feature anomaly detection
   │
   ├── anomaly type classification
   │
   └── component identification
```

---

# 39. Recommended First Experiment

Start with exactly one installation step where a missing component is visually obvious.

Collect approximately:

```text
Training:
150 good images

Validation:
30 good
30 missing-component images

Test:
50 good
50 missing-component images
```

Use:

```text
Fixed camera

Tight ROI

512 × 512 input

PatchCore

WideResNet50-2

layer2 + layer3

coreset ratio = 0.1
```

Evaluate whether the anomaly-score distributions separate clearly:

```text
Normal

0.08
0.12
0.17
0.20
0.23


Missing component

0.58
0.64
0.69
0.75
0.82
```

If the two distributions separate strongly, the core concept is validated.

---

# 40. Final POC Architecture

```text
                CORRECT INSTALLATIONS
                         │
                         │
                150–300 images
                         │
                         ▼
                    ROI CROP
                         │
                         ▼
                    PATCHCORE
                         │
                         ▼
                  MEMORY BANK
                         │
                         │
                         │
                 PRODUCTION IMAGE
                         │
                         ▼
                IMAGE QUALITY CHECK
                         │
                         ▼
                    ROI CROP
                         │
                         ▼
                    PATCHCORE
                         │
                ┌────────┴────────┐
                │                 │
                ▼                 ▼
          ANOMALY SCORE       ANOMALY MAP
                │                 │
                └────────┬────────┘
                         ▼
                     THRESHOLD
                         │
                 ┌───────┴────────┐
                 ▼                ▼
               PASS            ANOMALY
                                  │
                                  ▼
                       HIGHLIGHT SUSPECT AREA
```

This architecture keeps the first implementation deliberately small while providing a direct path toward a multi-step manufacturing installation verification system.