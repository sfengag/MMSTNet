# MMSTNet
This repo is the implementation of our manuscript entitled A macro-micro spatio-temporal neural network for traffic prediction. The code is based on Pytorch, and tested with NVIDIA GeForce RTX 3090.

In this study, we present a macro-micro spatio-temporal neural network model (MMSTNet) for traffic prediction. The model utilizes a graph convolution network and a spatial attention network to capture micro and macro spatial correlations, respectively. It also employs a temporal convolution network and a temporal attention network to learn temporal patterns. The model can also integrate hierarchically learned representations based on designed attention mechanisms.

## Model framework
<img src = "images/model framework1.png" width="50%">

## Dataset
The dataset is based on California Department of Transportation (Caltrans) Performance Measurement (PeMS, https://pems.dot.ca.gov/), where we select two of them for testing (PEMS03 and PEMS08). The raw data for the two dataset is summarized here. In the training and testing, we conduct some pre-processing such as Z-score normalization to make the raw data more appropriate for usage.

## Explanation of the code files

## Citation
