# PISCO

This repository contains code for the paper <a href="https://arxiv.org/abs/2302.09795">Simple Disentanglement of Style and Content in Visual Representations</a> 

[Simple Disentanglement of Style and Content in Visual Representations](https://arxiv.org/abs/2302.09795)



# Experiments on MNIST, CIFAR-10, and ImageNet 

1. Start by installing the required packages listed in the ./requirements.txt file. <br><br>

2. Generate image features of datasets used in the experiments as follows: <br>
  - Generate MNIST features by running the `MNIST_feature_extractor.ipynb` Jupyter notebook which can be accessed by navigating to `experiments -> feature_extractors-> MNIST_feature_extractor.ipynb`  <br>
  
  - Generate CIFAR-10 ResNet-18 features by running the `CIFAR10_ResNet18_feature_extractor.ipynb` Jupyter notebook which can be accessed by navigating to `experiments -> feature_extractors-> CIFAR10_ResNet18_feature_extractor.ipynb`  <br>
  
  - Generate CIFAR-10 SimCLR features by running the `CIFAR10_SimCLR_feature_extractor.ipynb` Jupyter notebook which can be accessed by navigating to `experiments -> feature_extractors-> CIFAR10_SimCLR_feature_extractor.ipynb`  <br><br>
  
  - For ImageNet, first generate stylized ImageNet datasets using styles found in `experiments -> styles_for_imagenet` (one style at a time). To generate the stylized images use the code by Geirhos et al. https://github.com/rgeirhos/Stylized-ImageNet but first modify it so that one style is applied to all ImageNet images at time (create a stylized ImageNet dataset for each style).
  
  - To generate Resnet-50 features of original ImageNet dataset, first change the file paths in `ImageNet_ResNet50_feature_extractor_og.py` and then run this command `python ./experiments/feature_extractors/ImageNet_ResNet50_feature_extractor_og.py`
  
  - To generate Resnet-50 features of stylized ImageNet datasets, first change the file paths in `ImageNet_ResNet50_feature_extractor_stylized.py` and then run this command `python ./experiments/feature_extractors/ImageNet_ResNet50_feature_extractor_stylized.py` <br><br>
  
3. To generate results for MNIST, run the bash script `run_mnist.sh`. To generate results for CIFAR-10, run the bash script `run_cifar10.sh`. And to generate results for ImageNet, run the bash script `run_imagenet.sh` <br><br>

4. To play with the demo, to create plots for MNIST and CIFAR-10, and to generate results tables for ImageNet using results generated in step 3, open and run the Jupyter notebooks `results_mnist.ipynb`, `results_cifar10.ipynb`, and `results_imagenet.ipynb`, respectively - all found in `experiments -> results`. Plots are saved in the ./plots folder. <br>


# Synthetic Data Study

A demo and a visualization of the summary plot from saved results are provided in `demo_plot.ipynb`.  To reproduce the results and plots, run the bash script `jobs.sh`, and create \& visualize plots from `demo_plot.ipynp` file. All the files for the syntheic data study are in the `./synthetic_data_study` folder.

