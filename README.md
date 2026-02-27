<h2 align="center">MultiAnimate: Pose-Guided Image Animation Made Extensible</h2>

<div align="center">
  Yingcheng Hu<sup>1,2,3*</sup>&nbsp;&nbsp;&nbsp;
  Haowen Gong<sup>3</sup>&nbsp;&nbsp;&nbsp;
  <a href="https://winycg.github.io/" target="_blank">Chuanguang Yang</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
  Zhulin An<sup>1,&dagger;</sup>&nbsp;&nbsp;&nbsp;
  Yongjun Xu<sup>1</sup>&nbsp;&nbsp;&nbsp;
  <a href="https://huage001.github.io/" target="_blank">Songhua Liu</a><sup>3,&dagger;</sup>
</div>

<div align="center" style="margin-top: 10px;">
  <sup>1</sup>State Key Laboratory of AI Safety, Institute of Computing Technology, CAS<br>
  <sup>2</sup>ShanghaiTech University<br>
  <sup>3</sup>Shanghai Jiao Tong University<br>
  <small><sup>&dagger;</sup>Corresponding Authors</small>
</div>

<br>

<div align="center">
  <a href="https://arxiv.org/abs/2602.21581"><img src="https://img.shields.io/badge/arXiv-2602.21581-b31b1b.svg?logo=arXiv" alt="arXiv"></a>
  <a href="https://hyc001.github.io/MultiAnimate/"><img src="https://img.shields.io/badge/Project%20Page-Website-1E90FF.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/N00B0DY/MultiAnimate"><img src="https://img.shields.io/badge/🤗_Weights-HuggingFace-ffc107.svg" alt="Hugging Face"></a>
</div>

## 📖 Introduction
We present **MultiAnimate** for multi-character image animation, which is the first extensible framework built upon modern DiT-based video generators, to the best of our knowledge.

## 📰 News
* **[Feb 2026]** 🎉 MultiAnimate has been accepted by **CVPR 2026**! 

## 🎥 Demos

Our framework, trained only on two-character data, is capable of producing identity-consistent three-person videos and can, in principle, be extended to scenarios with even more participants.

<div align="center">
  <video src="https://github.com/user-attachments/assets/0ac48ba9-02b9-4d98-ad24-3f8486ca7d5a" autoplay muted loop playsinline width="80%"></video>
  <strong>Figure 1:</strong> Three-character animation.
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/5f19b816-7231-41d7-9761-1631099d2fcb" autoplay muted loop playsinline width="80%"></video>
  <strong>Figure 2:</strong> Four-character animation.
</div>

<!-- ## 💻 Code
🚧 **Coming Soon!** We are currently cleaning up the codebase and will release the inference scripts shortly. Please stay tuned! -->
## ⚙️ Environment Setup

We recommend using Anaconda to manage your environment. 

```bash
conda create -n multianimate python=3.10.16
conda activate multianimate

# CUDA 12.1
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/hyc001/MultiAnimate.git
cd MultiAnimate
# DiffSynth-Studio from source code
pip install -e .

pip install -r requirements.txt
```

## 🚀 Quick Demo

Before running the demos, please ensure you have the Hugging Face CLI installed:
```bash
pip install -U "huggingface_hub[cli]"
```

### 1. Download Base Model
MultiAnimate is built upon the Wan2.1 framework. First, download the base Wan2.1-I2V-14B-720P model into your `checkpoints` directory:
```bash
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir checkpoints/Wan2.1-I2V-14B-720P
```

### 2. Download Demo Dataset
We provide pre-annotated data (including reference images and poses) for 3-person and 4-person generation scenarios. Download the `demo` folder from our repository:
```bash
huggingface-cli download N00B0DY/MultiAnimate --include "demo/*" --local-dir .
```

### 3. Demo 1: Standard Model (Up to 3 Characters)
Download our standard checkpoint optimized for complex human interactions, and run the inference script:
```bash
# Download MultiAnimate weights
huggingface-cli download N00B0DY/MultiAnimate --include "epoch=39-step=7000.ckpt/*" --local-dir checkpoints/

# Run inference
CUDA_VISIBLE_DEVICES="0" python examples/inference_multi3.py
```

### 4. Demo 2: Extended Model (Up to 7 Characters)
We provide the extended model. Here, we use a 4-person scenario as an example:
```bash
# Download extended MultiAnimate weights
huggingface-cli download N00B0DY/MultiAnimate --include "epoch=23-step=4200-multi.ckpt/*" --local-dir checkpoints/

# Run inference for 4 characters
CUDA_VISIBLE_DEVICES="0" python examples/inference_multi4.py
```

## 🏃‍♂️ Training & Data Preparation

### 1. Data Processing
To train or fine-tune the model, you need to extract human poses and character masks from your video data. We recommend the following pipeline, though you are completely free to use other equivalent methods:

* **Pose Extraction:** We utilize [DWPose](https://github.com/IDEA-Research/DWPose) to extract human skeletal keypoints.
* **Mask Extraction:** We use [Sa2VA-8B](https://github.com/bytedance/Sa2VA) to segment the characters. A typical prompt used for extraction looks like: `"<image>Please segment the left person"`.

After processing, your `processed_data` directory should be organized as follows:

```text
processed_data/
├── video1/
│   ├── frames.pkl
│   ├── mask_female.pkl
│   ├── mask_male.pkl
│   └── pose.pkl
├── video2/
│   ├── frames.pkl
│   ├── ...
└── ...
```

### 2. Training
**💡 Memory Saving Tip:** Based on our experience, if you are training on an A100 GPU (or GPUs with similar VRAM constraints), we highly recommend using **prompt feature caching** to save memory. 
Please refer to `examples/prompt_emb.py` for feature extraction and `examples/train_multi.py` for the core training logic.

Once your data and environment are ready, you can start the training process by running:

```bash
sh train.sh
```

## 🙏 Acknowledgements
Our codebase is built upon the wonderful [UniAnimate-DiT](https://github.com/ali-vilab/UniAnimate-DiT). We sincerely thank the authors for their fantastic open-source contribution to the community!

## 📝 BibTeX
If you find our work helpful, please consider citing:

```bibtex
@article{hu2026multianimateposeguidedimageanimation,
      title={MultiAnimate: Pose-Guided Image Animation Made Extensible}, 
      author={Yingcheng Hu and Haowen Gong and Chuanguang Yang and Zhulin An and Yongjun Xu and Songhua Liu},
      year={2026},
      eprint={2602.21581},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.21581}, 
}
