# OS-Oracle: A Comprehensive Framework for Cross-Platform GUI Critic Models
<div align="center">

 [\[💻Code\]](https://github.com/numbmelon/OS-Oracle) [\[📝Paper\]]() [\[🤗Models\]]()[\[🤗Data\]]() [\[🤗OS-Critic Bench\]](https://huggingface.co/datasets/OS-Copilot/OS-Critic-Bench) 

</div>


## Overview
OS-Oracle is a comprehensive framework designed for developing cross-platform GUI critic models that span mobile, desktop, and web environments. The framework integrates three key components — data synthesis, model training, and evaluation — to enable consistent and scalable critic model development across diverse GUI platforms.

To facilitate systematic evaluation, we introduce OS-Critic Bench, a unified benchmark for assessing GUI critic models across all platforms. Models trained under the OS-Oracle framework demonstrate strong generalization and reasoning ability, with OS-Oracle-7B achieving state-of-the-art performance among open-sourced VLMs on OS-Critic Bench.

![os-orcale-overview](https://github.com/user-attachments/assets/f4ca58c1-740a-488a-9ebf-25b8bb4a8f3f)

## 📝TODO List
- [ ] Release data synthesis pipeline  
- [ ] Release training datasets  
- [ ] Release model checkpoints    

## OS-Critic Bench
Follow the steps below to use **OS-Critic Bench**.
### 1. Download the Benchmark

Clone the dataset from Hugging Face and rename it:

```bash
cd os-critic-bench
git clone https://huggingface.co/datasets/OS-Copilot/OS-Critic-Bench


mv OS-Critic-Bench test_jsonl
```

### 2. Run the Inference Script
Execute the following command to run inference across all three platforms (Mobile, Desktop, and Web). 

Before running the evaluation, make sure that all dependencies for the target model are properly installed and that the script has been correctly configured.
```
bash run_eval.sh
```

### 3. Get the results
After inference is completed, compute the final metrics
```
python cal_acc.py --jsonl <your_output_file_path>
```

## Citation
If you find this repository helpful, feel free to cite our paper:
```bibtex
@article{wu2025osoracle,
        title={OS-Oracle: A Comprehensive Framework for Cross-Platform GUI Critic Models},
        author={Zhenyu Wu and Jingjing Xie and Zehao Li and Bowen Yang and Qiushi Sun and Zhaoyang Liu and Zhoumianze Liu and Yu Qiao and Xiangyu Yue and Zun Wang and Zichen Ding},
        journal={arXiv preprint arXiv:2512.16295},
        year={2025}
      }
```
