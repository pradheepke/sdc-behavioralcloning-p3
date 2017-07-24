# Behaviorial Cloning Project

## Model architecture

Tried simple first to make sure end to end is working.

nVidia architecture.

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

