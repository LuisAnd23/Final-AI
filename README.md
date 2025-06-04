# Final-AI
This repository contains the implementation and comparative analysis of four deep learning models for skin lesion classification using the HAM10000 dataset. Our study demonstrates that model selection requires careful balancing of diagnostic accuracy with operational constraints, with multimodal integration achieving the best performance when complete clinical metadata is available. The highest melanoma recall was 49% (multimodal model), highlighting the persistent challenge in detecting malignant lesions.
The full data set is available on https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/code
The four models was run on google colab with GPTU 4 activated, in order to run the code on a local copilator it is necesary to adjust the paths 
the libraries necesaries are tensorflow, scikit-learn, pandas, numpy and matplotlib
