# Decentralized Design of Fast Iterative Receivers for Massive MIMO with Spatial Non-Stationarities
This is a research-oriented code package that is primarily intended to allow readers to replicate the results of the article mentioned below and also encourage and accelerate further research on this topic:

V. Croisfelt, T. Abrão, A. Amiri, E. de Carvalho and P. Popovski, ["Decentralized Design of Fast Iterative Receivers for Massive MIMO with Spatial Non-Stationarities,"](https://ieeexplore.ieee.org/document/9723315) 2021 55th Asilomar Conference on Signals, Systems, and Computers, Pacific Grove, CA, USA, 2021, pp. 1242-1249, doi: 10.1109/IEEECONF53345.2021.9723315.

A preprint is available on:

Victor Croisfelt, Taufik Abrão, Abolfazl Amiri, Elisabeth de Carvalho, and Petar Popovski, “Decentralized Design of Fast Iterative Receivers for Massive MIMO with Spatial Non-Stationarities. Available on: https://arxiv.org/abs/2107.11349.

I hope this content helps in your research and contributes to building the precepts behind open science. Remarkably, in order to boost the idea of open science and further drive the evolution of science, I also motivate you to share your published results with the public.

If you have any questions or encounter any inconsistency, please do not hesitate to contact me via victorcroisfelt@gmail.com.

## Abstract
Despite the extensive use of a centralized approach to design receivers for massive multiple-input multiple-output (M-MIMO) {base stations}, their actual implementation is a major challenge due to several bottlenecks imposed by the large number of antennas. One way to deal with this problem is by fully decentralizing the classical zero-forcing (ZF) receiver across multiple processing nodes based on the gradient descent (GD) method. In this paper, we first explicitly relate this decentralized receiver to a distributed version of the Kaczmarz algorithm and to the use of the successive interference cancellation (SIC) philosophy. In addition, we propose three methods to further accelerate the initial convergence of this decentralized receiver by exploring the connection with the Kaczmarz algorithm: 1) a new Bayesian distributed receiver, which iteratively estimates and eliminates noise; 2) a more practical method for choosing a relaxation parameter on an iteration basis; and 3) extension of it to tree-based architectures. The discussion also considers spatial non-stationarities that arise when the antenna arrays are extremely large (XL-MIMO). With our methods, we are able to improve the performance of the decentralized GD receiver for both spatially stationary and non-stationary channels, but mainly the non-stationary performance can still be improved. Future research directions are provided with the aim of further improving the effectiveness of receivers based on the so-called principle of successive residual cancellation (SRC).

## Content
The codes provided here can be used to simulate Figs. 6 and 7. The code is divided in the following way:
  - scripts starting with sim_ generate data on the /data folder used to plot the figures.
  - scripts starting with the keyword "plot_" actually plot the figures using matplotlib.

The scripts comm.py, commsetup.py, receiver.py act as sources defining the main functions used in the simulation scripts.

Further details about each file can be found inside them.

## Citing this Repository and License
This code is subject to the MIT license. If you use any part of this repository for research, please consider citing our aforementioned work.

```bibtex
@INPROCEEDINGS{9723315,
  author={Croisfelt, Victor and Abrão, Taufik and Amiri, Abolfazl and de Carvalho, Elisabeth and Popovski, Petar},
  booktitle={2021 55th Asilomar Conference on Signals, Systems, and Computers}, 
  title={Decentralized Design of Fast Iterative Receivers for Massive MIMO with Spatial Non-Stationarities}, 
  year={2021},
  volume={},
  number={},
  pages={1242-1249},
  keywords={Interference cancellation;Philosophical considerations;Receiving antennas;Receivers;Massive MIMO;MIMO;Iterative algorithms;Massive MIMO (M-MIMO);extra-large scale MIMO (XL-MIMO);fully decentralized receivers;daisy-chain architecture;Kaczmarz algorithm;successive residual cancellation (SRC);Bayesian receiver;relaxation method},
  doi={10.1109/IEEECONF53345.2021.9723315}
}
