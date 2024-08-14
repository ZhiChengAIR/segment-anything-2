# SAM 2: Segment Anything in Images and Videos

**[AI at Meta, FAIR](https://ai.meta.com/research/)**

[Nikhila Ravi](https://nikhilaravi.com/), [Valentin Gabeur](https://gabeur.github.io/), [Yuan-Ting Hu](https://scholar.google.com/citations?user=E8DVVYQAAAAJ&hl=en), [Ronghang Hu](https://ronghanghu.com/), [Chaitanya Ryali](https://scholar.google.com/citations?user=4LWx24UAAAAJ&hl=en), [Tengyu Ma](https://scholar.google.com/citations?user=VeTSl0wAAAAJ&hl=en), [Haitham Khedr](https://hkhedr.com/), [Roman Rädle](https://scholar.google.de/citations?user=Tpt57v0AAAAJ&hl=en), [Chloe Rolland](https://scholar.google.com/citations?hl=fr&user=n-SnMhoAAAAJ), [Laura Gustafson](https://scholar.google.com/citations?user=c8IpF9gAAAAJ&hl=en), [Eric Mintun](https://ericmintun.github.io/), [Junting Pan](https://junting.github.io/), [Kalyan Vasudev Alwala](https://scholar.google.co.in/citations?user=m34oaWEAAAAJ&hl=en), [Nicolas Carion](https://www.nicolascarion.com/), [Chao-Yuan Wu](https://chaoyuan.org/), [Ross Girshick](https://www.rossgirshick.info/), [Piotr Dollár](https://pdollar.github.io/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/)

[[`Paper`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)] [[`Project`](https://ai.meta.com/sam2)] [[`Demo`](https://sam2.metademolab.com/)] [[`Dataset`](https://ai.meta.com/datasets/segment-anything-video)] [[`Blog`](https://ai.meta.com/blog/segment-anything-2)] [[`BibTeX`](#citing-sam-2)]

![SAM 2 architecture](assets/model_diagram.png?raw=true)

**Segment Anything Model 2 (SAM 2)** is a foundation model towards solving promptable visual segmentation in images and videos. We extend SAM to video by considering images as a video with a single frame. The model design is a simple transformer architecture with streaming memory for real-time video processing. We build a model-in-the-loop data engine, which improves model and data via user interaction, to collect [**our SA-V dataset**](https://ai.meta.com/datasets/segment-anything-video), the largest video segmentation dataset to date. SAM 2 trained on our data provides strong performance across a wide range of tasks and visual domains.

![SA-V dataset](assets/sa_v_dataset.jpg?raw=true)

## Table of Content
1. [Installation](#Installation)
2. [Getting Started](#getting-started)
3. [Image Prediction](#image-prediction)
4. [Video Prediction](#video-prediction)
5. [Model Description](#model-description)
6. [Segment Anything Video Dataset](#segment-anything-video-dataset)
7. [USING THE SAM2 ENCODER](#using-the-sam2-encoder) (FOR ZHICHENG-AI)

## Installation

SAM 2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. You can install SAM 2 on a GPU machine using:

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git

cd segment-anything-2 & pip install -e .
```
If you are installing on Windows, it's strongly recommended to use [Windows Subsystem for Linux (WSL)](https://learn.microsoft.com/en-us/windows/wsl/install) with Ubuntu.

To use the SAM 2 predictor and run the example notebooks, `jupyter` and `matplotlib` are required and can be installed by:

```bash
pip install -e ".[demo]"
```

Note:
1. It's recommended to create a new Python environment via [Anaconda](https://www.anaconda.com/) for this installation and install PyTorch 2.3.1 (or higher) via `pip` following https://pytorch.org/. If you have a PyTorch version lower than 2.3.1 in your current environment, the installation command above will try to upgrade it to the latest PyTorch version using `pip`.
2. The step above requires compiling a custom CUDA kernel with the `nvcc` compiler. If it isn't already available on your machine, please install the [CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with a version that matches your PyTorch CUDA version.
3. If you see a message like `Failed to build the SAM 2 CUDA extension` during installation, you can ignore it and still use SAM 2 (some post-processing functionality may be limited, but it doesn't affect the results in most cases).

Please see [`INSTALL.md`](./INSTALL.md) for FAQs on potential issues and solutions.

## Getting Started

### Download Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```bash
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

or individually from:

- [sam2_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt)
- [sam2_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt)
- [sam2_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt)
- [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt)

Then SAM 2 can be used in a few lines as follows for image and video prediction.

### Image prediction

SAM 2 has all the capabilities of [SAM](https://github.com/facebookresearch/segment-anything) on static images, and we provide image prediction APIs that closely resemble SAM for image use cases. The `SAM2ImagePredictor` class has an easy interface for image prompting.

```python
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image(<your_image>)
    masks, _, _ = predictor.predict(<input_prompts>)
```

Please refer to the examples in [image_predictor_example.ipynb](./notebooks/image_predictor_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/segment-anything-2/blob/main/notebooks/image_predictor_example.ipynb)) for static image use cases.

SAM 2 also supports automatic mask generation on images just like SAM. Please see [automatic_mask_generator_example.ipynb](./notebooks/automatic_mask_generator_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/segment-anything-2/blob/main/notebooks/automatic_mask_generator_example.ipynb)) for automatic mask generation in images.

### Video prediction

For promptable segmentation and tracking in videos, we provide a video predictor with APIs for example to add prompts and propagate masklets throughout a video. SAM 2 supports video inference on multiple objects and uses an inference state to keep track of the interactions in each video.

```python
import torch
from sam2.build_sam import build_sam2_video_predictor

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(<your_video>)

    # add new prompts and instantly get the output on the same frame
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, <your_prompts>):

    # propagate the prompts to get masklets throughout the video
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        ...
```

Please refer to the examples in [video_predictor_example.ipynb](./notebooks/video_predictor_example.ipynb) (also in Colab [here](https://colab.research.google.com/github/facebookresearch/segment-anything-2/blob/main/notebooks/video_predictor_example.ipynb)) for details on how to add click or box prompts, make refinements, and track multiple objects in videos.


## Model Description

|      **Model**       | **Size (M)** |    **Speed (FPS)**     | **SA-V test (J&F)** | **MOSE val (J&F)** | **LVOS v2 (J&F)** |
| :------------------: | :----------: | :--------------------: | :-----------------: | :----------------: | :---------------: |
|   sam2_hiera_tiny    |     38.9     |          47.2          |        75.0         |        70.9        |       75.3        |
|   sam2_hiera_small   |      46      | 43.3 (53.0 compiled\*) |        74.9         |        71.5        |       76.4        |
| sam2_hiera_base_plus |     80.8     | 34.8 (43.8 compiled\*) |        74.7         |        72.8        |       75.8        |
|   sam2_hiera_large   |    224.4     | 24.2 (30.2 compiled\*) |        76.0         |        74.6        |       79.8        |

\* Compile the model by setting `compile_image_encoder: True` in the config.

## Segment Anything Video Dataset

See [sav_dataset/README.md](sav_dataset/README.md) for details.

## USING THE SAM2 ENCODER
Since this repo is customised for Diffusion Policy, below shows a guide of how to use
the encoder within the project.

### Relative Directory Structure
Since both SAM2 and Diffusion Policy both use Hydra, the directory structure is a little bit
confusing. There are two sets of files that need to be placed carefully in the Diffusion Policy
repo, these are as follows:

#### Config Files

Put SAM2 config yaml files found [here](https://github.com/ZhiChengAIR/segment-anything-2/tree/main/sam2_configs) into [diffusion_policy/config](https://github.com/ZhiChengAIR/DiffusionPolicy/tree/main/diffusion_policy/config).

#### Model Weights

The address for where the weights are to be found should start at the beginning of the
diffusion policy directory. This will be shown in the code block below.

### Creating the Model
To create the model, as shown in the code above and according to the file structure mentioned,
we create the model like so, with the below example showing how to call the SAM2-large model.

```python
from sam2.build_sam import build_sam2


sam_weights_addr = "diffusion_policy/model/vision/segment-anything-2" \
                + "/checkpoints/sam2_hiera_large.pt"
sam2_config = f"sam2_hiera_l.yaml"
sam2 = build_sam2(sam2_config, sam_weights_addr, apply_postprocessing=False)
```

### Encoding Images
To access the encoder it's as simple as the following code:
```python
sam2_encoder = sam2.image_encoder
```

However to encode images requires a little bit of adjustments, the below code
shows how to properly extract the encoded images from SAM2.

```python
import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Resize


class SAM2Transforms(nn.Module):
    def __init__(self, resolution):
        """
        Transforms for SAM2.
        """
        super().__init__()
        self.resolution = resolution
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transforms = torch.jit.script(
            nn.Sequential(
                Resize((self.resolution, self.resolution)),
                Normalize(self.mean, self.std),
            )
        )

    def __call__(self, x):
        return self.transforms(x)


bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

def set_image_batch(image_list: torch.Tensor) -> None:
    transforms = SAM2Transforms()
    img_batch = transforms(image_list)
    batch_size = img_batch.shape[0]
    backbone_out = sam2.forward_image(img_batch)
    _, vision_feats, _, _ = sam2._prepare_backbone_features(backbone_out)
    feats = [
        feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
        for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
    ][::-1]
    return feats[-1]
```
