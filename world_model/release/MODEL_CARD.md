---
license: other
license_name: nvidia-open-model-license
base_model: nvidia/Cosmos-Predict2-2B-Video2World
library_name: cosmos
pipeline_tag: image-to-video
tags:
  - robotics
  - world-model
  - multi-view-video
  - action-conditioned-video
  - cosmos
---

# A2World World Model

A2World is a multi-view action-conditioned diffusion world model trained to predict future robot observations from initial camera observations and a 20-step action chunk. A2World-sim adds pose-guided visual history and autoregressive rollout support.

## Variants

- [`a2world-pretrained.pt`](https://huggingface.co/Fleurrr/A2World-World-Model/resolve/main/a2world-pretrained.pt): heterogeneous robot-data pretrained A2World checkpoint.
- [`a2world-libero.pt`](https://huggingface.co/Fleurrr/A2World-World-Model/resolve/main/a2world-libero.pt): history-aware A2World-sim checkpoint adapted on LIBERO.

Both released files use EMA weights, the standard `net.*` inference prefix, and BF16 storage compatible with the public rollout pipeline.

## Base Model and License

The checkpoints are derivative models of `nvidia/Cosmos-Predict2-2B-Video2World` and are governed by the NVIDIA Open Model License included in this repository. Source code is separately available under Apache-2.0.

## Intended Use

- Multi-view action-conditioned robot video prediction.
- Open-loop and chunk-wise autoregressive world-model rollouts.
- LIBERO learned-simulator research and downstream adaptation.
- Research on transferable robot dynamics priors.

## Limitations

- Generated rollouts may hallucinate geometry, contact, object state, or task success.
- Autoregressive errors accumulate over long horizons.
- Camera identity and action semantics must match the selected embeddings and preprocessing.
- The model is not a certified simulator and must not replace real-robot safety validation.

## Citation

```bibtex
@inproceedings{huang2026a2world,
    title={Learning Transferable Dynamics Priors from Action to World Modeling},
    author={Huang, Ze and Zhang, Jiahui and Liu, Hairuo and Zhang, Chenxi and Cheng, Ran and Zhang, Li},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2026},
}
```
