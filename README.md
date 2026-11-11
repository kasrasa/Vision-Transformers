# Vision-Transformers
A repo to compare and experiment with different key ViTs.

Run the notebooks from the repository root so local imports and data paths resolve.

| Location | Responsibility |
| --- | --- |
| `models/` | ViT and CNN definitions, patch/token embeddings, encoder blocks, and prediction heads |
| `datasets/imagenette.py` | Imagenette loading and image preprocessing |
| `pipeline/part1_train.py` | Part 1's original training, checkpoint selection, and early stopping |
| `pipeline/part1_scheduler.py` | Part 1's original validation-driven learning-rate reductions |
| `pipeline/train.py` | Part 2 training and validation, including the fixed-budget A0/A1 runs |
| `pipeline/scheduler.py` | Part 2 warmup with linear or cosine decay, updated with optimizer steps |
| `pipeline/evaluate.py` | Classification metrics on selected model weights |
| `pipeline/inference.py` | Device synchronization and inference timing |
| `pipeline/predictions.py` | Shared prediction and two-head probability-fusion rules |
| `pipeline/checkpoints.py` | Checkpoint loading/saving and JSON experiment results |
| `visualization/plots.py` | Reusable plotting helpers |
| `part1.ipynb` | Part 1 component walkthrough and baseline experiments |
| `patch_embedding.ipynb` | Earlier component/experiment notebook, updated to the same package imports |
| `part2.ipynb` | Part 2 experiment orchestration, starting with the A0/A1 scheduler comparison |
| `article-results/` | Article drafts, experiment plans, and existing figures/results |

Training remains explicit in the notebook cells. Part 1 uses its original `linear_scheduler` function, which reduces the learning rate based on validation loss, with early stopping enabled and the existing maximum epoch limits preserved. Its training path is independent of the new Part 2 warmup/decay schedules. The published Part 1 article measurements have not been replaced by new training runs.
