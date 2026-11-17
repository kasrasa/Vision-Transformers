# Vision-Transformers

Experiments and implementation walkthroughs for the Vision Transformer article series.
Run the notebooks from the repository root so local imports and data paths resolve.

| Location | Responsibility |
| --- | --- |
| `models/part1_vit/` | Part 1 ViT backbone and original MLP-head builder |
| `models/part2_deit/` | Part 2 linear-head ViT and two-token distillation student |
| `models/part3_swin/window_utils.py` | Part 3 spatial window partitioning and reconstruction |
| `models/common/` | Patch embeddings, encoder blocks, and heads reused across parts |
| `models/cnn/` | Shared half-width ResNet-18 and ResNet-50 baselines |
| `datasets/imagenette.py` | Imagenette loading and deterministic preprocessing |
| `datasets/augmentation.py` | Training image transforms, Mixup/CutMix, and target smoothing |
| `pipeline/part1_train.py` | Part 1 training, checkpoint selection, and early stopping |
| `pipeline/train.py` | Fixed-budget training and validation used in Part 2 |
| `pipeline/scheduler.py` | Part 1 plateau schedule and Part 2 warmup/decay schedules |
| `pipeline/losses.py` | Supervision, distillation, and two-head probability fusion |
| `pipeline/evaluate.py` | Classification and calibration metrics |
| `pipeline/inference.py` | Device synchronization and inference timing |
| `pipeline/checkpoints.py` | Checkpoint loading/saving and result tables |
| `pipeline/experiment_utils.py` | Shared seed resetting and JSON experiment metadata |
| `visualization/plots.py` | Patch displays and training curves |
| `visualization/windows.py` | Window partition and reconstruction displays |
| `part1.ipynb` | Part 1 component walkthrough and CNN/ViT baseline experiments |
| `part2.ipynb` | Part 2 training recipe, distillation, and checkpoint evaluation |
| `part3.ipynb` | Part 3 window mechanics, score-count scaling, and dataset setup |
| `article-results/part3/draft.md` | Revised Part 3 article opening |

Model imports identify the article part explicitly:

```python
from models.part1_vit import build_vision_transformer
from models.part2_deit import (
    build_vision_transformer_single_classifier,
    build_distilled_vision_transformer,
)
from models.part3_swin import window_partition, window_reverse
from models.cnn import ResNet18
```

Training and experiment orchestration remain explicit in the notebooks. Part 1
retains its original plateau schedule and early stopping. Part 2 retains its
existing model configurations, training budget, checkpoint selection, and saved
results; only imports and shared setup helpers have moved.

Part 3 currently implements window mechanics. Its small grid and synthetic token
examples run before Imagenette is loaded. The optional RGB example uses the local
image when available. The final dataset section retains the Part 2 split policy
for later training comparisons. Window attention, shifted windows, patch merging,
and a full Swin classifier are subsequent implementation milestones.
