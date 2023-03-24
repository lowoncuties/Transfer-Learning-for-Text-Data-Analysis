# Transfer-Learning-for-Text-Data-Analysis

This repository belongs to a Bachelor thesis on a topic of Transfer Learning for Text Data Analysis from Vysoká škola báňská - Technická univerzita Ostrava <br> 
https://dspace.vsb.cz/handle/10084/147359

## Abstract

The aim of this bachelor thesis was to test transfer learning methods on different datasets and then
compare results with simpler machine learning methods. Text analysis is a complex field, so I picked
a subfield called text classification. Experiments need data, therefore I included a part dedicated
to their preprocessing. There is more than one language used in the experiments. Used languages
are English, French and Czech, with more languages I could compare results of each method and
model within the language and without the doubt I could tell which method performs the best for
the language. I would like to mention a very good performance of the transformer models, they can
perform surprisingly well even with small training dataset, in most cases they even outperformed
deep learning methods trained on tens of thousands training samples.

## Summary

The thesis as itself is a summary of the different machine learning technique and architectures applied on a 4 different datasets:
1. IMDB 
2. English Twitter
3. French Twitter
4. CSFD (Similar to IMDB but Czech) <br>

The used techniques are: 
1. Support Vector Machine - Simpler ML model for benchmarking against the deep learning models
2. Recurrent neural networks (Own architecture)
3. Recurrent neural networks (Using pretrained FastText embeddings)
4. Transformers -  BERT, camemBERT, distilBERT, roBERTa, small-e-czech, XLM-roBERTa

Used framworks:
1. Scikit-learn - Support Vector Machine
2. Tensorflow, Keras - Recurrent neural networks
3. PyTorch - Only for Transformers

Directory: <br>
folder CSFD: <br>
- Jupyter notebook BERT_CSFD.ipynb containing all the experiments for the method using Transformers
- Jupyter notebook FastText_CSFD.ipynb containing all the experiments for the method using RNNs with FastText embeddings
- Jupyter notebook RNN_CSFD.ipynb containing all the experiments for the method using my own RNN architecture
- Jupyter notebook SVM_CSFD.ipynb containing all the experiments for the method using SVMs
- Jupyter notebook CSFD_Dataset.ipynb containing all the preprocessing steps and the description of the data

All directories for all datasets looks the same


## Results

The sizes for training set may vary, that is because the tables are showing the best results for each method and dataset, if you wanna find out which method was best in average or how does the architecture looked like, checkout jupyter notebooks and the thesis. <br>
Datasets can be found on my GDrive here: https://drive.google.com/drive/folders/1OmB3FlDFeXJKrOVyl0JPx_GIUN4RT0xB?usp=sharing

### IMDB dataset

|   Method  | Train set size | Training time (s) | Accuracy | F1-score |
|:---------:|:--------------:|:-----------------:|:--------:|:--------:|
| SVM       |         25 000 |              1575 |   84.29% |    0.842 |
| BERT      |         25 000 |             13549 |   93.38% |    0.934 |
| RNN model |         25 000 |              1363 |   86.92% |    0.874 |
| FastText  |         25 000 |               169 |   85.70% |    0.853 |

### English Twitter dataset

|   Method  | Train set size | Training time (s) | Accuracy | F1-score |
|:---------:|:--------------:|:-----------------:|:--------:|:--------:|
| SVM       |         25 000 |               266 |   72.27% |    0.726 |
| BERT      |         20 000 |               921 |   82.40% |    0.824 |
| RNN model |        900 000 |               766 |   82.58% |    0.827 |
| FastText  |        900 000 |               783 |   81.98% |    0.825 |

### French Twitter dataset

|   Method  | Train set size | Training time (s) | Accuracy | F1-score |
|:---------:|:--------------:|:-----------------:|:--------:|:--------:|
| SVM       |         25 000 |               343 |   73.76% |    0.742 |
| camemBERT |         40 000 |              3870 |   82.07% |    0.820 |
| RNN model |        900 000 |              1026 |   81.27% |    0.815 |
| FastText  |        900 000 |               734 |   80.42% |    0.798 |

### CSFD dataset

|     Method    | Train set size | Training time (s) | Accuracy | F1-score |
|:-------------:|:--------------:|:-----------------:|:--------:|:--------:|
| SVM           |         25 000 |               411 |   88.70% |    0.886 |
| small-e-czech |         50 613 |               855 |   89.55% |    0.898 |
| RNN model     |         50 613 |               561 |   90.04% |    0.902 |
| FastText      |         50 613 |               547 |   87.51% |    0.876 |




## Thesis
Since the thesis is written solely in Czech (Article is still TODO), only the results were here. Don't worry tho, the jupyter notebook comments are written in English. <br>

If you use any part from the Thesis, please cite <br>
```
@thesis{Jochymek2022,
  author = {Lukáš Jochymek},
  title = {Transfer learning pro analýzu textových dat},
  address = {Ostrava},
  year = {2022},
  school = {Vysoká škola báňská – Technická univerzita Ostrava},
  type = {Bakalářská práce},
  urldate = {2023-03-20},
  url = {http://hdl.handle.net/10084/147359},
}
```
