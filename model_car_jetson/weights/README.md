# Model Weights

Model binaries are not tracked in Git. Download or attach them through a GitHub
Release and place them here:

```text
weights/
├── detection/
│   └── yolov8s_model_car_best.pt
└── segmentation/
    └── segformer_b0_yellow_line_best_model/
        ├── config.json
        └── model.safetensors
```

The runtime defaults expect these paths. The backup detector checkpoint is not
needed for the public repository.
