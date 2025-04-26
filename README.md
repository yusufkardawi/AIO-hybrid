# Efficient Hybrid Network with Prompt Learning for Multi-Degradation Image Restoration


[Muhammad Yusuf Kardawi](https://scholar.google.com/citations?user=SsezH7EAAAAJ&hl=id&authuser=2), [Laksmita Rahadianti](https://scholar.google.com/citations?hl=id&authuser=2&user=zXG3mDwAAAAJ)



### News
- February 8, 2025: Submitted to [Jurnal RESTI](https://jurnal.iaii.or.id/index.php/RESTI)
- April 12, 2025: Accepted
- April 20, 2025: [Published Online](https://jurnal.iaii.or.id/index.php/RESTI/article/view/6381)

---

> Abstract: Image restoration aims to repair degraded images. Traditional image restoration methods have limited generalization capabilities due to the difficulty in dealing with different types and levels of degradation. On the other hand, contemporary research has focused on multi-degradation image restoration by developing unified networks capable of handling various types of degradation. One promising approach is using prompts to provide additional information on the type of input images and the extent of degradation. Nonetheless, all-in-one image restoration requires a high computational cost, making it challenging to implement on resource-constrained devices. This research proposes a multi-degradation image restoration model based on PromptIR with lower computational cost and complexity. The proposed model is trained and tested on various datasets yet it is still practical for deraining, dehazing, and denoising tasks. By unification convolution, transformer, and dynamic prompt operations, the proposed model successfully reduces FLOPs by 32.07% and the number of parameters by 27.87%, with a comparable restoration result and an SSIM of 34.15 compared to 34.33 achieved by the original architecture for the denoising task.

---

## Network Architecture

## Getting Started
```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tqdm scikit-image matplotlib
pip install einops lightning scikit-video thop wandb
pip install thop wandb
```

## Real-world Dataset
The real-world dataset used in this project can be found at [Google Drive](https://drive.google.com/file/d/1PV2mQSPGQAWSTLSI96mPQD3VlwVmR4zY/view?usp=sharing)


## Citation

Please cite our work if you find it useful:
```
@article{Muhammad Yusuf Kardawi_Laksmita Rahadianti_2025, 
  title={Efficient Hybrid Network with Prompt Learning for Multi-Degradation Image Restoration }, 
  volume={9}, 
  url={https://jurnal.iaii.or.id/index.php/RESTI/article/view/6381}, 
  DOI={10.29207/resti.v9i2.6381}, 
  number={2}, 
  journal={Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi)}, 
  author={Muhammad Yusuf Kardawi and Laksmita Rahadianti}, 
  year={2025}, 
  month={Apr.}, 
  pages={404 - 415} 
}
```

## Contact
Should you have any question, please contact **<ins>my.kardawi@gmail.com</ins>**

## Acknowledgment
This project is based on the [PromptIR](https://github.com/va1shn9v/PromptIR.git) and [CAPTNet](https://github.com/Tombs98/CAPTNet.git)

