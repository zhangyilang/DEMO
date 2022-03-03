# DEMO

This repository provides codes for the paper: [DEMO: A Flexible Deartifacting Module for Compressed Sensing MRI](). 

Here we provide an example to showcase the implementation of the DEMO algorithm, with [ADMM-Net](https://github.com/yangyan92/Deep-ADMM-Net) as the backbone. Data can be downloaded from [this repository](https://github.com/yangyan92/ADMM-CSNet). As a flexible add-on module (architecture shown below), DEMO can be readily extended to other algorithms (e.g. [ADMM-CSNet](https://github.com/yangyan92/ADMM-CSNet), [TVAL3](https://www.caam.rice.edu/~optimization/L1/TVAL3/), [NLR-CS](http://see.xidian.edu.cn/faculty/wsdong/Code_release/NLR_codes.rar)) with some slight modifications. 

<img src="web_images/DEMO_for_NN.pdf">

# Visualization results

MRI data with synthetic artifacts:

<img src="web_images/synthetic_examples.pdf">

MRI data with real artifacts:

<img src="web_images/real_data.pdf">