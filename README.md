# Behaviorial Cloning Project

[//]: # (Image References)
[ValAug]: ./img/ValidationAugment.png
[ValNoAug]: ./img/ValidationNoAugment.png
[FailNoAug]: ./img/FailureCaseNoAugment.png

## Overview

The approach I took:
 - First get a simple model, train on 1K images, get end to end training and feeding data to simulator working.
 - Pick an architecture used for this problem (nVidia architecture: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). Make sure end-to-end works on small set of images.
 - Understand errors, improve model, iterate.

The file video.mp4 shows final video of model which is running perfectly for track 1.
I have also included two other videos with models from intermediate iterations to illustrate where things were failing.

## Model architecture

I used the nVidia architecture: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

It seems to have sufficient representation power (5 conv+ReLU layers, 4 FC layers -- covered will with lot of  parameters and non-linearity), so I did not try much model architecture explorations.

I was actually surprised that when I trained a simple 1 layer model with all the images, the model went a fair distance (probably 50% of the lap) without much issues.

When I tried training with the nVidia architecture, with all images, it went almost the whole lap. It usually failed either near the water body at the very end or the bridge earlier. 

## Iterations to improve accuracy

I used the Adam optimize, and did not tune learning rates or any optimizer parameters.

The main tactics I used to improve generalization were data augmentation and early stopping.

Three main things that helped
 1. *Augmenting using LR Flip*: Based on initial runs, I thought that the model may be overfitting driving only to the left. So, I augmented the data with left-right flipped images and reversed steering angles.
 1. *Checkpointing using monitor*: I noticed a lot of overfitting from the training logs, that is, training loss improving but validation loss not improving after early iterations. I found it efficient to add a checkpoint monitor here which saved a lot of manual repeats to identify optimal number of epochs. This will just save model whenever validation loss improved and I always saved the last snapshot so I can visualize the effect of early stopping.
 1. *Augmenting using clockwise driving*: The above two steps got the model to pretty good performance on track 1. It wouldn't always be near center but it would recover from several situations as shown in the checked-in supplementary videos. Based on the failure scenario, it seemed that there still wasn't enough cases of deciding when to go right and when to go left. So I added data from simulator but reversing the car and started driving clockwise to collect training data. A model trained with this data including early stopping shows perfect performance for track 1. It is interesting to note that I added just about 2.6K images to the training data, just about 20% more training data.

*Overfitting*: It is interesting to note the effect of augmented data on overfitting.

Without the clockwise driving data, we can see from this that there is clearly overfitting.
![Validation MSE for NoAug][ValNoAug]
Validation loss improves until epoch 6. After epoch 6, training loss continues to drop from 0.0089 to 0.0069, dropping 23%, but validation loss goes the other way, from a to b, worse by 16%.

With the clockwise driving data, there is still overfitting but to a slightly lesser extent. Training loss improves from 0.0155 to 0.0119 after epoch 6 (30%), but validation loss goes worse only by 9%.
![Validation MSE][ValAug]

More such iterations are required to improve the generalization of the model but this works sufficient for track 1.

Without augmentation, this is an example failure scene.
![Sample failure scenario for no augmentation][FailNoAug]

The attached video, video.mp4, shows the successful model, and the two other videos illustrate effect of lack of early stopping and lack of data augmentation.

*CPU*
 - I found it interesting that it was "sufficient" to just use CPU to do well on track 1. Adding the checkpoint monitor helped to avoid wasted cycles of iterations. 
 - Takes about 30mins for each epoch with more data, so got a decent model in a couple of hours.
 - Of course, for the model to really generalize, GPU will be required for faster iterations and taking lot more data.

## Reflections
 This was a useful exercise to see an end-to-end system though in simulator for predicting steering angle.
One thing that I feel will be really important is how one can robustly identify why a deep learning model is failing? 
I used blackbox-y system level heuristics to identify overfitting. 
It will be really useful to have a good efficient system for understanding why a model fails so that we can systematically and confidently decide what kind of data to augment it with.





