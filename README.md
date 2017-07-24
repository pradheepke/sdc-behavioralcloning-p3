# Behaviorial Cloning Project

## Overview

The approach I took:
 - First get a simple model, train on 1K images, get end to end training and feeding data to simulator working.
 - Pick an architecture used for this problem (nVidia architecture: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). Make sure end-to-end works on small set of images.
 - Understand errors, improve model, iterate.

The file video.mp4 shows final video of model which is running perfectly for track 1.
I have also included two other videos with models from intermediate iterations to illustrate where things were failing.

## Model architecture

I used the nVidia architecture: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

Overfitting/underfitting detection.

## Sequence of steps to improve:
 - Flip LR.
 - 1000 images. 
 - Checkpointing for convenient early stopping

 - Simulation
   - Went opposite and drove clockwise. Just about 2K images.

 - Interesting to see what happens without the augmented data.
  - Note validation vs. training loss changes. 
  - Lesser overfitting in the new one. Still there is.

 - Wanted to see how much we could do just with CPU.

 - How does one robustly identify why something is failing? 
Other than blackbox system level heuristics? 
How to decide what kind of data that I should augment it with?


[//]: # (Image References)
[ValAug]: ./img/ValidationAugment.png
[ValNoAug]: ./img/ValidationNoAugment.png
[FailNoAug]: ./img/FailureCaseNoAugment.png

![Validation MSE][ValAug]
![Validation MSE for NoAug][ValNoAug]
![Sample failure scenario for no augmentation][FailNoAug]

