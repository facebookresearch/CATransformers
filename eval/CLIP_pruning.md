## Pruning For CLIP models using MoPE
We prune the CLIP models using importance based techniques from MoPE-CLIP. We provide pre-calculated importance ranking for the ViT-B-16 datacomp model in [eval/mope/ViT-B-16_datacomp_xl_s13b_b90k](eval/mope/ViT-B-16_datacomp_xl_s13b_b90k). To use other models, modify `MODEL_ARCH` and `PRETRAINED` in [configurations.py](configurations.py), and run:

```bash
python eval/init_importance.py
```
This only needs to be done once, when running the model for the first time. NOTE: this may take up to a few hours to run.
