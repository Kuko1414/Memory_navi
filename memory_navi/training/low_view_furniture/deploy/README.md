# Low-view furniture detector deployment

`low_view_furniture_v1.json` is the publication profile for the locally deployed
YOLO26n segmentation model. The model file is intentionally ignored by Git and lives at:

```text
/home/kuko/Kuko1414/models/low_view_furniture/low_view_furniture_v1/best.pt
```

The expected SHA256 is recorded in the profile. `candidate_only` labels and detections
below their per-class publish threshold are returned in `candidate_objects`; they are
excluded from the confirmed `objects` list that can enter navigation memory. In shadow
mode, candidate-inclusive output is written separately as
`area_yolo_with_candidates.json` and is never used for navigation.

## Shadow mode

Keep Qwen as the driving perception backend while writing the YOLO comparison to a
separate report directory:

```bash
export PERCEPTION_BACKEND=qwen
export DUAL_REVIEW_DIR=/home/kuko/Kuko1414/Report/low_view_furniture/shadow_route_01
export YOLOE_PYTHON=/home/kuko/miniconda3/envs/yolo/bin/python
export YOLOE_MODEL=/home/kuko/Kuko1414/models/low_view_furniture/low_view_furniture_v1/best.pt
export YOLOE_PROFILE=/home/kuko/Kuko1414/memory_navi/training/low_view_furniture/deploy/low_view_furniture_v1.json
export YOLOE_CONF=0.05
```

## Make YOLO the driving backend

Only do this after the three shadow routes pass review:

```bash
export PERCEPTION_BACKEND=yoloe
```

## Roll back

```bash
export PERCEPTION_BACKEND=qwen
unset YOLOE_PROFILE YOLOE_MODEL DUAL_REVIEW_DIR
```
