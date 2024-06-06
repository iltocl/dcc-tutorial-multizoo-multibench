# dcc-tutorial-multizoo-multibench
This tutorial is aimed to provide a general description of the multizoo and multibench sources as well as practical examples of how to use it 

- [Multizoo and Multibench Tutorial](#dcc-tutorial-multizoo-multibench)
  - [Summary](#summary)
  - [Getting Started](#getting-started)
      - [Datasets](#datasets)
      - [Models](#models)
      - [Code Structure](#code-structure)
      - [Example of MultiBench usage](https://colab.research.google.com/github/iltocl/dcc-tutorial-multizoo-multibench/blob/main/Examples/Multibench_Example_Usage_Colab.ipynb)
  - [Media Coverage](#media-coverage)

# Summary
We share with you: 
- A brief description of the MultiZoo, MultiBench resources ([2406-TutorialMultibenchMultizoo.pdf](2406-TutorialMultibenchMultizoo.pdf))
- An overview tutorial on how to pre train and finetune a model considering multimodal datasets

# Getting Started
## Datasets
### ComicMischief and HateSpeech datasets
We share with you ComicMischief and HateSpeech datasets. 
- ComicMischief is a multimodal (audio, visual, and audio) video dataset aimed at comic mischief Binary and Multilabel (gory, slapstick, mature, sarcasm) tasks. 
- HateSpeech is a multimodal (audio, visual, and audio) video dataset aimed at hate speech Binary task.  

To access them go to this [drive folder](https://drive.google.com/drive/folders/1RrPJuVRm8kxqPey37YiuRxztmP-zEyxP?usp=sharing) and create a shortcut on your drive (see figure below).

![Alt text](img.png)

Once you have completed the previous step, you can run the tutorial: [Example of MultiBench usage](https://colab.research.google.com/github/iltocl/dcc-tutorial-multizoo-multibench/blob/main/Examples/Multibench_Example_Usage_Colab.ipynb)

### Multibench datasets
Multibench provides a compilation of multimodal datasets. In our case, Affective Computing datasets may be considered because of the availability of audio, visual, and text modalities. These Affective Computing datasets include:
- CMU-MOSEI
- CMU-MOSI
- UR-FUNNY
- MUSTARD

## Models
### HICCAP model
We can also use HICCAP model from the original paper ([Labelling Comic Mischief Content in Online Videos with a Multimodal Hierachical Cross-Attention Model](https://aclanthology.org/2024.lrec-main.874.pdf)). 
The original repository could be accessed at [here](https://github.com/RiTUAL-UH/Comic-Mischief-Prediction).

### MultiZoo models
MultiZoo toolkit provides a compilation of standard implementations of models. The catalog could be consulted at [multibench.readthedocs.io](https://multibench.readthedocs.io/en/latest/index.html)

## Code Structure
```
├── Examples
│   ├── 
├── HICCAP
│   ├── 
├── create_dataset.py
├── data_loader.py
├── testing.py
├── 
│   ├── 
│   ├── 
└── README.md
```

# Media Coverage
[MultiZoo & MultiBench: A Standardized Toolkit for Multimodal Deep Learning](https://www.jmlr.org/papers/volume24/22-1021/22-1021.pdf)
Paul Pu Liang, Yiwei Lyu, Xiang Fan, Arav Agarwal, Yun Cheng, Louis-Philippe Morency, Ruslan Salakhutdinov
JMLR 2022 Open Source Software.

[MultiBench: Multiscale Benchmarks for Multimodal Representation Learning](https://arxiv.org/abs/2107.07502)
Paul Pu Liang, Yiwei Lyu, Xiang Fan, Zetian Wu, Yun Cheng, Jason Wu, Leslie Chen, Peter Wu, Michelle A. Lee, Yuke Zhu, Ruslan Salakhutdinov, Louis-Philippe Morency
NeurIPS 2021 Datasets and Benchmarks Track.

[MultiBench GitHub repository](https://github.com/pliang279/MultiBench.git)

[Labeling Comic Mischief Content in Online Videos with a Multimodal Hierarchical-Cross-Attention Model](https://aclanthology.org/2024.lrec-main.874/)
Elaheh Baharlouei, Mahsa Shafaei, Yigeng Zhang, Hugo Jair Escalante, Thamar Solorio
Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)

[Comic-Mischief GitHub repository](https://github.com/RiTUAL-UH/Comic-Mischief-Prediction)
