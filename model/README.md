# Model Files

This folder should contain the required model files. They are **not included** in the repository due to size and licensing restrictions.

## Required Files

### 1. FLAME Model (`generic_model.pkl`)

**Download:**
1. Visit [FLAME Model Website](https://flame.is.tue.mpg.de/)
2. Register for an account
3. Download FLAME 2020 (`generic_model.pkl`)
4. Place in this `models/` folder

**Size:** ~100 MB

**License:** Requires registration. For research use only.

**Important:** Make sure you download the **FLAME 2020** version, as the pre-trained mappings in this repository are designed for this specific version.

### 2. MediaPipe Face Landmarker (`face_landmarker.task`)

**Download:**
```bash
# From the root of the repository
curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Or download manually from:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

**Size:** ~30 MB

**License:** Apache 2.0

**Note:** Move this file to the root directory or update the `MODEL_PATH` in `main.py`.

## File Structure

After downloading, your `models/` folder should look like:

```
models/
├── README.md (this file)
└── generic_model.pkl
```

And your root directory should have:
```
flame-avatar-driver/
├── face_landmarker.task
├── models/
│   └── generic_model.pkl
└── ...
```

## Troubleshooting

### Cannot load FLAME model
- Ensure you downloaded **FLAME 2020**, not FLAME 2023 or other versions
- Check file is named exactly `generic_model.pkl`
- Verify file is in `models/` folder

### MediaPipe model not found
- Check `face_landmarker.task` is in the root directory
- Or update `MODEL_PATH` in `main.py` to point to correct location

## Alternative: Use Your Own FLAME Model

If you have a different FLAME model variant:
1. Place it in `models/` folder
2. Update `FLAME_PATH` in `main.py`
3. You may need to retrain the mappings using the tools in `tools/` folder