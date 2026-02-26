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

## 💻 Code
🚧 **Coming Soon!** We are currently cleaning up the codebase and will release the inference scripts shortly. Please stay tuned!

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
