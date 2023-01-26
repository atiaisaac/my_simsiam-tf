SimSiam is one of many self-supervised learning algorithms and thus shares a some similaritiies. However, what makes SimSiam different is the fact that it does not rely on negative image pairs, clustering or a memory bank to avoid a degenerating solution. Instead, it cleverly uses a *stop gradient* operation and predictor MLP (an autoencoder structure) to learn semantic representations present in image data to avoid a collapsing solution. 

In this notebook, I document what worked and didn't work for me, in my quest to build a SimSiam architecture in tensorflow following most of the training regimen specified in the original [paper](https://arxiv.org/pdf/2011.10566.pdf). 

The original architecture was built in Pytorch and as I have come to realize, there are some subtle differences between tensors/states of pytorch and tensorflow - thus my attempt tries to mimic as close as possible the pytorch implementation. Do note that, in the paper, training was done with the cifar10 dataset and was trained for 800 epochs on 8 gpu's. However, due to the lack of compute power, I trained for 400 epochs on a single gpu using the stanford dogs dataset. Pre-training and model validation together took a total of 6hrs to 8hrs depending on how the setup is tweaked.

## Methodology

### Augmentation scheme

1. RandomResizedCrop - tensorflow does not have this augmentation scheme out of the box. Hence a seperate function was created for this purpose. It has an applying probability of 1.0
2. Color Jitter - also tensorflow does not have color jitter out of the box, but can be implmeneted with seperate brightness, contrasts, hue and saturation levels with applying probability of 0.8
3. Color drop - this can be implemented out of the box with tensorflow using `tf.image.rgb_to_grayscale` with probability of 0.2
4. random flip - can be done out of the box in tensorflow. It has applying probability of 1.0.
5. Gaussian blur - implemented with `tfa.image.gaussian_filter2d` with probability of 0.5

### Model Architecture

In the paper, the SimSiam model architecture is comprised of two layers - the encoder which in itself is made up of the feature extraction backbone like a ResNet and a projection MLP comprised of dense (linear) layers and the prediction MLP layer also comprised of dense layers. In the cifar-10 experiment, ResNet18 was used as the feature extractor and the projection MLP had 2 dense layers instead of the 3 that is indicated in the baseline settings. 

The weights in Pytorch are initialized using LeCun initialization by default and have values ranging from (-k,k) where `k = sqrt(1/in-feature)` (ie uniform distribution). Tensorflow on the otherhand uses glorot initialization by default to initialize weights. To conform to the paper, my test uses the `LeCunUniform` for weight initialization which is very close to the pytorch implementation as it also has values ranging from (-k,k) where `k = sqrt(3/in-feature)`.

### Optimizer

The paper mentions the use of an SGD optimizer with a learning rate scheduler, weight decay and momentum of 0.9. 

Batch size however is not imperative and much attention was not given to it.

## Summary of my implementation and results

This section will be updated as I continue with my experiments and thus my notebook.

| architecture description| linear evaulation results |
| ----------------------- | ------------------------- |
| 

