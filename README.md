<div align="center" style="font-family: charter;">

<h1>Learning Transferable Dynamics Priors from Action to World Modeling</h1>

<p>
    <a href="https://logosroboticsgroup.github.io/A2World/" target="_blank">
        <img alt="Project Page" src="https://img.shields.io/badge/Project_Page-online-blue.svg" height="20" />
    </a>
</p>

<div>
    <span>Ze Huang</span><sup>1 *</sup>,
    <span>Jiahui Zhang</span><sup>1 *</sup>,
    <span>Hairuo Liu</span><sup>2 3 *</sup>,
    <span>Chenxi Zhang</span><sup>2</sup>,
    <span>Ran Cheng</span><sup>4</sup>,
    <a href="https://lzrobots.github.io/" target="_blank">Li Zhang</a><sup>1 2 &dagger;</sup>
</div>

<div>
    <sup>1</sup>Fudan University&emsp;
    <sup>2</sup>Shanghai Innovation Institute&emsp;
    <sup>3</sup>Shanghai Jiao Tong University&emsp;
    <sup>4</sup>McGill University
</div>

<div>
    <sup>*</sup>Equal contribution&emsp;
    <sup>&dagger;</sup>Corresponding author
</div>

<br>

<img src="https://logosroboticsgroup.github.io/A2World/docs/resources/teaser.png" width="100%"/>

<p align="justify" style="font-style: italic;">
  <b>We view action-conditioned world modeling as a transferable dynamics prior for robot learning.</b>
  A2World is pretrained on 2.1M+ robot manipulation trajectories spanning 20+ embodiments to predict future multi-view manipulation videos from an initial observation and future action chunks.
  The same pretrained dynamics prior can be adapted into A2World-sim, a long-horizon autoregressive simulator for policy evaluation, and A2World-policy, a video-action joint prediction model for instruction-conditioned robot control.
</p>

</div>

## Abstract

Project page: [https://logosroboticsgroup.github.io/A2World/](https://logosroboticsgroup.github.io/A2World/)

We study action-conditioned world modeling as a scalable way to learn transferable dynamics priors for robot learning. By pretraining a model to predict how actions drive visual scene evolution, the resulting world model captures reusable interaction dynamics beyond appearance-level video generation.

Concretely, we pretrain a multi-view interactive base diffusion world model, **A2World**, on large-scale robot manipulation data with real action annotations. We validate the learned dynamics priors from two complementary perspectives. First, we adapt A2World into a task- or scene-specialized real-world simulator, **A2World-sim**, whose long-horizon rollouts support simulator-based policy evaluation and scalable what-if analysis by replacing real-robot rollouts with world model rollouts. Second, starting from the same pretrained weights, we adapt A2World into a video-action joint prediction model, **A2World-policy**, that predicts actions under visual and instruction conditioning.

Experiments across simulation benchmarks and real-robot settings demonstrate that action-conditioned world model pretraining yields transferable dynamics priors that benefit both simulator-centric and policy-centric robot learning.

## Method

<img src="https://logosroboticsgroup.github.io/A2World/docs/resources/extensions.png" width="100%"/>

**A2World** learns action-to-video dynamics from robot data, then transfers the pretrained prior into two downstream variants:

- **A2World-sim** injects pose-guided history and rolls out future observations autoregressively for long-horizon simulator-based evaluation.
- **A2World-policy** performs joint video-action diffusion with shared attention and action-specific denoising branches for instruction-conditioned control.

## Code Release

The current release focuses on the **A2World world-model component**:

- multi-view action-conditioned A2World inference;
- history-aware A2World-sim autoregressive rollout generation;
- LIBERO HDF5 conversion, full-parameter fine-tuning, and inference;
- checkpoint validation, model-card metadata, and NVIDIA weight licensing.

```bash
cd world_model
python -m pip install -e . --no-deps
python scripts/download_base_models.py

a2world-demo \
  --variant libero \
  --checkpoint /path/to/a2world-libero.pt \
  --input /path/to/agentview.mp4 /path/to/eye_in_hand.mp4 \
  --actions /path/to/actions.npz \
  --base-checkpoints checkpoints \
  --output outputs/libero_rollout.mp4 \
  --autoregressive
```

See [`world_model/README.md`](world_model/README.md) for environment setup, data conversion, training, and rollout options.

The released checkpoints are hosted at [`Fleurrr/A2World-World-Model`](https://huggingface.co/Fleurrr/A2World-World-Model):

- [`a2world-pretrained.pt`](https://huggingface.co/Fleurrr/A2World-World-Model/resolve/main/a2world-pretrained.pt)
- [`a2world-libero.pt`](https://huggingface.co/Fleurrr/A2World-World-Model/resolve/main/a2world-libero.pt)

## World Model Rollouts

These videos show A2World world model rollouts on real-robot manipulation scenarios. They are predictions of future interaction dynamics, not direct camera recordings of policy execution. Click a preview to watch the full rollout.

<table>
  <tr>
    <td align="center" width="20%">
      <a href="https://logosroboticsgroup.github.io/A2World/docs/resources/world-model-rollouts/flip_small_box.mp4">
        <img src="https://logosroboticsgroup.github.io/A2World/docs/resources/posters/flip_small_box.jpg" width="100%" alt="A2World world model rollout: flip small box"/>
      </a>
      <br>
      <sub><b>Flip small box</b><br>Reorientation under contact-rich manipulation.</sub>
    </td>
    <td align="center" width="20%">
      <a href="https://logosroboticsgroup.github.io/A2World/docs/resources/world-model-rollouts/insert_memory_module.mp4">
        <img src="https://logosroboticsgroup.github.io/A2World/docs/resources/posters/insert_memory_module.jpg" width="100%" alt="A2World world model rollout: insert RAM module"/>
      </a>
      <br>
      <sub><b>Insert RAM module</b><br>Precision alignment and insertion dynamics.</sub>
    </td>
    <td align="center" width="20%">
      <a href="https://logosroboticsgroup.github.io/A2World/docs/resources/world-model-rollouts/lift_box_high.mp4">
        <img src="https://logosroboticsgroup.github.io/A2World/docs/resources/posters/lift_box_high.jpg" width="100%" alt="A2World world model rollout: lift box high"/>
      </a>
      <br>
      <sub><b>Lift box high</b><br>Longer-horizon object lifting and transport.</sub>
    </td>
    <td align="center" width="20%">
      <a href="https://logosroboticsgroup.github.io/A2World/docs/resources/world-model-rollouts/put_chain_in_the_box.mp4">
        <img src="https://logosroboticsgroup.github.io/A2World/docs/resources/posters/put_chain_in_the_box.jpg" width="100%" alt="A2World world model rollout: put chain in the box"/>
      </a>
      <br>
      <sub><b>Put chain in the box</b><br>Deformable-object handling with container interaction.</sub>
    </td>
    <td align="center" width="20%">
      <a href="https://logosroboticsgroup.github.io/A2World/docs/resources/world-model-rollouts/electric_valve_switch.mp4">
        <img src="https://logosroboticsgroup.github.io/A2World/docs/resources/posters/electric_valve_switch.jpg" width="100%" alt="A2World world model rollout: toggle power switch"/>
      </a>
      <br>
      <sub><b>Toggle power switch</b><br>Interactive switch manipulation with small contact changes.</sub>
    </td>
  </tr>
</table>

## Real-Robot Execution Videos

A2World-policy is evaluated on a Flexiv dual-arm real-robot suite covering precision insertion, reorientation, switch interaction, lifting, and deformable-object handling. Click a preview to watch the full video.

<table>
  <tr>
    <td colspan="2" align="center" width="100%">
      <a href="https://logosroboticsgroup.github.io/A2World/docs/resources/real-world-execution/put-chain-in-box.mp4">
        <img src="https://logosroboticsgroup.github.io/A2World/docs/resources/posters/put-chain-in-box.jpg" width="70%" alt="A2World-policy real-robot execution: put chain in the box"/>
      </a>
      <br>
      <sub><b>Put chain in the box</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://logosroboticsgroup.github.io/A2World/docs/resources/real-world-execution/toggle-power-switch.mp4">
        <img src="https://logosroboticsgroup.github.io/A2World/docs/resources/posters/toggle-power-switch.jpg" width="100%" alt="A2World-policy real-robot execution: toggle power switch"/>
      </a>
      <br>
      <sub><b>Toggle power switch</b></sub>
    </td>
    <td align="center" width="50%">
      <a href="https://logosroboticsgroup.github.io/A2World/docs/resources/real-world-execution/flip-small-box.mp4">
        <img src="https://logosroboticsgroup.github.io/A2World/docs/resources/posters/flip-small-box.jpg" width="100%" alt="A2World-policy real-robot execution: flip small box"/>
      </a>
      <br>
      <sub><b>Flip small box</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://logosroboticsgroup.github.io/A2World/docs/resources/real-world-execution/insert-ram-module.mp4">
        <img src="https://logosroboticsgroup.github.io/A2World/docs/resources/posters/insert-ram-module.jpg" width="100%" alt="A2World-policy real-robot execution: insert RAM module"/>
      </a>
      <br>
      <sub><b>Insert RAM module</b></sub>
    </td>
    <td align="center" width="50%">
      <a href="https://logosroboticsgroup.github.io/A2World/docs/resources/real-world-execution/lift-box-high.mp4">
        <img src="https://logosroboticsgroup.github.io/A2World/docs/resources/posters/lift-box-high.jpg" width="100%" alt="A2World-policy real-robot execution: lift box high"/>
      </a>
      <br>
      <sub><b>Lift box high</b></sub>
    </td>
  </tr>
</table>

## BibTeX

If you find this project helpful, please consider citing our paper, in Proceedings of ECCV 2026:

```bibtex
@inproceedings{huang2026a2world,
    title={Learning Transferable Dynamics Priors from Action to World Modeling},
    author={Huang, Ze and Zhang, Jiahui and Liu, Hairuo and Zhang, Chenxi and Cheng, Ran and Zhang, Li},
    booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
    year={2026},
}
```
