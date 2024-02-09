### Guiding Mathematical Reasoning via Mastering Commonsense Formula Knowledge

This is the offcial repo for the KDD-2023 paper "[Guiding Mathematical Reasoning via Mastering Commonsense Formula Knowledge](https://dl.acm.org/doi/10.1145/3580305.3599375)".

### Requirements
* python==3.6
* torch==1.7.1
* other pakages


### Environment
* OS: CentOS Linux release 7.7.1908
* CPU: 64 Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
* GPU: Four Tesla V100-SXM2 32GB
* CUDA: 10.1


### Datasets
* Math23K-F
```shell
formula.txt: collection of 51 commonsense math formulas
```
```shell
formula_variant.txt: collection of 131 commonsense math formulas and their variants
```
* MAWPS-F
```shell
formula.txt: collection of 18 commonsense math formulas
```
```shell
formula_variant.txt: collection of 46 commonsense math formulas and their variants
```

For any question, please contact us with email: jy251198@mail.ustc.edu.cn.

### Running

**Train the model :** 
```shell
python run_seq2tree.py
```

### Citation
If you find this work useful, please cite our paper:
```
@inproceedings{liu2023guiding,
  title={Guiding Mathematical Reasoning via Mastering Commonsense Formula Knowledge},
  author={Liu, Jiayu and Huang, Zhenya and Ma, Zhiyuan and Liu, Qi and Chen, Enhong and Su, Tianhuang and Liu, Haifeng},
  booktitle={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1477–1488},
  year={2023}
}
```






