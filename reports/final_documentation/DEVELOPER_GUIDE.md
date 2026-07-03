# Developer Guide

## Modifying the Model
To replace the default MobileNetV2 with a different architecture (e.g., ConvNeXt):
1. Place the new `.keras` model in `models/`.
2. Update `MODEL_NAME` in `api/services/inference.py`.

## Modifying Validation Heuristics
Update `block_keywords` in `api/services/validation.py` to add or remove ImageNet classes that trigger a rejection.
