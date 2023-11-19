## Demo code for:


[[Paper]](https://ieeexplore.ieee.org/abstract/document/10304214)
[[Demo code]](https://github.com/XuZitai/S2S-WTV/blob/main/S2S_WTV.py)
# Deep Nonlocal Regularizer: A Self-Supervised Learning Method for 3-D Seismic Denoising  


### Citation

This article is accepted by IEEE TGRS. If you use this model in your research, please cite:

    @ARTICLE{dnlr,
  author={Xu, Zitai and Luo, Yisi and Wu, Bangyu and Meng, Deyu and Chen, Yangkang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Deep Nonlocal Regularizer: A Self-Supervised Learning Method for 3-D Seismic Denoising}, 
  year={2023},
  volume={61},
  number={},
  pages={1-17},
  doi={10.1109/TGRS.2023.3329303}}
     

### Abstract

Noise suppression for seismic data can meliorate the quality of many subsequent geophysical tasks. In this work, we propose a novel self-supervised learning method, the deep nonlocal regularizer (DNLR), for 3-D seismic denoising. Our DNLR fully exploits the nonlocal self-similarity (NSS) of seismic data under a self-supervised learning framework for noise attenuation. It can be flexibly combined with different hand-crafted regularizers, e.g., total variation (TV), nuclear norm (NN), and correlated TV (CTV), by performing the regularizer on nonlocal self-similar patches, which more effectively characterizes the intrinsic structures underlying seismic data. Our DNLR can be easily plugged into existing self-supervised denoising methods, e.g., deep image prior (DIP) and Self2Self (S2S), and consistently improve their performance. To make the optimization model tractable, an algorithm based on the alternating direction multiplier method (ADMM) is introduced to solve the DNLR-based seismic denoising problem. Extensive seismic denoising experiments on synthetic and field data validate the superior performances of our DNLR as compared with state-of-the-art model-based and deep learning seismic denoising methods. Code is available at https://github.com/XuZitai/DNLR .
 

**Note**

Accepted by IEEE TGRS
