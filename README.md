# PISCO

This repository contains code for the paper [Simple Disentanglement of Style and Content in Visual Representations](https://arxiv.org/abs/2302.09795)



# Experiments on MNIST, CIFAR-10, and ImageNet 

1. Start by installing the required packages listed in the `./requirements.txt` file. <br><br>

2. Generate image features of datasets used in the experiments as follows: <br><br>
  - Generate MNIST features by running the `MNIST_feature_extractor.ipynb` Jupyter notebook which can be accessed by navigating to `experiments -> feature_extractors-> MNIST_feature_extractor.ipynb` <br><br>
  
  - Generate CIFAR-10 ResNet-18 features by running the `CIFAR10_ResNet18_feature_extractor.ipynb` Jupyter notebook which can be accessed by navigating to `experiments -> feature_extractors-> CIFAR10_ResNet18_feature_extractor.ipynb` <br><br>
  
  - Generate CIFAR-10 SimCLR features by running the `CIFAR10_SimCLR_feature_extractor.ipynb` Jupyter notebook which can be accessed by navigating to `experiments -> feature_extractors-> CIFAR10_SimCLR_feature_extractor.ipynb`  <br><br>
  
  - For ImageNet, first [download the ImageNet dataset](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) and generate stylized ImageNet datasets using styles found in `experiments -> styles_for_imagenet` (one style at a time). To generate the stylized images use the [code by Geirhos et al.](https://github.com/rgeirhos/Stylized-ImageNet), but first modify it so that one style is applied to all ImageNet images at a time (create a stylized ImageNet dataset for each style). <br><br>
  
  - To generate Resnet-50 features of original ImageNet dataset, first change the file paths in `ImageNet_ResNet50_feature_extractor_og.py` and then run this command `python ./experiments/feature_extractors/ImageNet_ResNet50_feature_extractor_og.py` <br><br>
  
  - To generate Resnet-50 features of stylized ImageNet datasets, first change the file paths in `ImageNet_ResNet50_feature_extractor_stylized.py` and then run this command `python ./experiments/feature_extractors/ImageNet_ResNet50_feature_extractor_stylized.py` <br><br>
  
3. To generate results for MNIST, run the bash script `run_mnist.sh`. To generate results for CIFAR-10, run the bash script `run_cifar10.sh`. And to generate results for ImageNet, run the bash script `run_imagenet.sh`. Clear saved results for each dataset in `./experiments/results` before generating new results. <br><br>

4. To play with the demo, to create plots for MNIST and CIFAR-10, and to generate results tables for ImageNet using results generated in step 3, open and run the Jupyter notebooks `mnist_results_analysis.ipynb`, `cifar10_results_analysis.ipynb`, and `imagenet_results_analysis.ipynb`, respectively - all found in `./experiments` folder. Plots are saved in the `./plots` folder. <br><br>


# Synthetic Data Study

A demo and a visualization of the summary plot from saved results are provided in `demo_plot.ipynb`.  To reproduce the results and plots, run the bash script `jobs.sh`, and create \& visualize plots from `demo_plot.ipynp` file. All the files for the synthetic data study are in the `./synthetic_data_study` folder.

