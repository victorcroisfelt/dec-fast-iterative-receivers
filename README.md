# Decentralized Design of Fast Iterative Receivers for Massive MIMO with Spatial Non-Stationarities
This is a research-oriented code package that is primarily intended to allow readers to replicate the results of the article mentioned below and also encourage and accelerate further research on this topic:

Victor Croisfelt, Taufik Abrão, Abolfazl Amiri, Elisabeth de Carvalho, and Petar Popovski, “Decentralized Design of Fast Iterative Receivers for Massive MIMO with Spatial Non-Stationarities. Available on: https://arxiv.org/abs/2107.11349.

I hope this content helps in your reaseach and contributes to building the precepts behind open science. Remarkably, in order to boost the idea of open science and further drive the evolution of science, I also motivate you to share your published results to the public.

If you have any questions and if you have encountered any inconsistency, please do not hesitate to contact me via victorcroisfelt@gmail.com.

## Abstract
Despite the extensive use of a centralized approach to design receivers for massive multiple-input multiple-output (M-MIMO) {base stations}, their actual implementation is a major challenge due to several bottlenecks imposed by the large number of antennas. One way to deal with this problem is by fully decentralizing the classical zero-forcing (ZF) receiver across multiple processing nodes based on the gradient descent (GD) method. In this paper, we first explicitly relate this decentralized receiver to a distributed version of the Kaczmarz algorithm and to the use of the successive interference cancellation (SIC) philosophy. In addition, we propose three methods to further accelerate the initial convergence of this decentralized receiver by exploring the connection with the Kaczmarz algorithm: 1) a new Bayesian distributed receiver, which iteratively estimates and eliminates noise; 2) a more practical method for choosing a relaxation parameter on an iteration basis; and 3) extension of it to tree-based architectures. The discussion also considers spatial non-stationarities that arise when the antenna arrays are extremely large (XL-MIMO). With our methods, we are able to improve the performance of the decentralized GD receiver for both spatially stationary and non-stationary channels, but mainly the non-stationary performance can still be improved. Future research directions are provided with the aim of further improving the effectiveness of receivers based on the so-called principle of successive residual cancellation (SRC).

## Content
The codes provided here can be used to simulate Figs. 6 and 7. The code is divided in the following way:
  - scripts starting with the keyword "plot_" actually plots the figures using matplotlib.
 
<!--
  - scripts starting with the keyword "data_" are used to generate data for curves that require a lot of processing. The data is saved in the /data folder and used by the respective "plot_" scripts.
  - scripts starting with the keyword "lookup_" are used to exhaustively find parameters, such as: number of nearby APs, Ccal_size, number of pilot-serving APs, Lmax, and effective DL transmit power for Estimator 3, delta. Considering the practical scenario, it also makes use of method proposed in Algorithm 1. 
-->

Further details about each file can be found inside them.

## Citing this Repository and License
This code is subject to the MIT license. If you use any part of this repository for research, please consider to cite our aforementioned work.
