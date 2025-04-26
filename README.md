# Efficient Hybrid Network with Prompt Learning for Multi-Degradation Image Restoration

Official implementation of **"Efficient Hybrid Network with Prompt Learning for Multi-Degradation Image Restoration"**  
[[Paper]](https://jurnal.iaii.or.id/index.php/RESTI/article/view/6381)

---

## Abstract
Image restoration aims to repair degraded images. Traditional image restoration methods have limited generalization capabilities due to the difficulty in dealing with different types and levels of degradation. On the other hand, contemporary research has focused on multi-degradation image restoration by developing unified networks capable of handling various types of degradation. One promising approach is using prompts to provide additional information on the type of input images and the extent of degradation. Nonetheless, all-in-one image restoration requires a high computational cost, making it challenging to implement on resource-constrained devices. This research proposes a multi-degradation image restoration model based on PromptIR with lower computational cost and complexity. The proposed model is trained and tested on various datasets yet it is still practical for deraining, dehazing, and denoising tasks. By unification convolution, transformer, and dynamic prompt operations, the proposed model successfully reduces FLOPs by 32.07% and the number of parameters by 27.87%, with a comparable restoration result and an SSIM of 34.15 compared to 34.33 achieved by the original architecture for the denoising task..

---

## Dataset
The dataset used in this project can be found in the [`dataset/`](./dataset/) directory.

---

## Getting Started

### Installation
Clone the repository:

git clone https://github.com/your-username/your-repo-name.git

Run inference on sample images:
python main.py --input_dir ./dataset/sample_images/ --output_dir ./results/


Please cite our work if you find it useful:
@article{your_citation_key,
  title={Paper Title},
  author={Your Name and Collaborator Name},
  journal={Conference/Journal},
  year={2025}
}
