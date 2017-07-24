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

It seems to have sufficient representation power (number of parameters and non-linearity), so I did not try much model architecture explorations.

I was actually surprised that when I trained a simple 1 layer model with all the images, the model went a fair distance (probably 50% of the lap) without much issues.

When I tried training with the nVidia architecture, with all images, it went almost the whole lap. It usually failed either near the water body at the very end or the bridge earlier. 

Overfitting/underfitting detection.

## Iterations to improve accuracy

I used the Adam optimizer. So did not tune learning rates and other parameters.

The main tactics I used to improve generalization were data augmentation and early stopping.

Three main things that helped
 1. *Augmenting using LR Flip*: Based on initial runs, I thought that the model may be overfitting driving only to the left. So, I augmented the data with left-right flipped images and reversed steering angles.
 1. *Checkpointing using monitor*: I noticed a lot of overfitting from the training logs, that is, training loss improving but validation loss not improving after early iterations. I found it efficient to add a checkpoint monitor here which saved a lot of manual repeats to identify optimal number of epochs. This will just save model whenever validation loss improved and I always saved the last snapshot so I can visualize the effect of early stopping.
 1. *Augmenting using clockwise driving*: The above two steps got the model to pretty good performance on track 1. It wouldn't always be near center but it would recover from several situations as shown in the checked-in supplementary videos. Based on the failure scenario, it seemed that there still wasn't enough cases of deciding when to go right and when to go left. So I added data from simulator but reversing the car and started driving clockwise to collect training data. A model trained with this data including early stopping shows perfect performance for track 1.

 - Flip LR.
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

