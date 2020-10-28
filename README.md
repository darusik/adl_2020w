# Applied Deep Learning Project 

## Desription
The main idea of this project is to create a dataset for the re-identification of people wearing masks.

## Motivation
At present, due to the safety measures, people are supposed to wear masks almost everywhere, which creates difficulties for using face recognition systems in public places. For instance, it became impossible to confirm a payment using FaceId in shops. Naturally, this issue produces a question of the existence of a face recognition system, which can identify a person even while she is wearing a mask. And like any other recognition task, this one requires a qualitative representative dataset. Authors in [1] tested 89 existing face recognition algorithms on the synthetic crafted dataset to check how good they can match a person in a mask with the same person without a mask. Observed test error varies from 5% to 50%, which is  quite critical. However, it is worth to mention that images with masks were not realistic since the authors just covered corresponding face parts without saving any 3-D characteristics: 

![Artificial data](/images/introduction/masks.png)

Thereby, there are two ways to improve presented results: either implement a new face recognition algorithm that will overperform existing ones or create a more realistic dataset, which can help current approaches to make more precise predictions. 
Since data collection is always the initial step of solving any Deep Learning problem, we focus our attention on proper dataset creation.


## Approaches
We propose 4 approaches for dataset creation:

### 1. Create a dataset from skratch by making photos of real people in real masks and without masks. 
As an alternative, people can also might be asked to make photos of themselves, which is more safety approach, but less doable.
#### Drawbacks: 
- requires a lot time and greetfull people who will agree to participate;
- not very safe due to necessity of making photos without masks;
#### Benefits:
- results in the most relevant and feasible dataset one can make.

### 2. Create a dataset from skratch by collecting photos of famous people through the Internet.  
#### Drawbacks: 
- requires a lot time for searching images;
- probably not so many famous people can be recognized and dataset can be not big enough;
- a dataset can be biased to Europe countries;
#### Benefits:
- still relusts in feasible and relevant dataset.


### 3. Create a synthetic dataset by applying AR masks to the existing dataset using 3D face reconstruction techiques. 
The main idea of this approach is to reconstruct 3D face from 2D photo and apply face mask on tope of it. 
3D face reconstruction can be made by using, for instance, [2]. The [code](https://github.com/cleardusk/3DDFA_V2) of this paper is publicly avaible and produces the following result:

![Borat 1](/images/introduction/borat1.jpg "Borat is wearing a mask, but he is still doing it in the wrong way!")
![Borat 2](/images/introduction/borat2.jpg "Don't be like Borat.")
![Borat 3](/images/introduction/borat3.jpg "Cover your mouth and nose.")
![Borat 4](/images/introduction/borat4.jpg "Seriously.")

There are many other papers [3], [4],[5], [6], [7] which are avaible together with code. 
For AR mask creation [Spark AR Studio](https://sparkar.facebook.com/ar-studio/learn/downloads) can be used.
As a result we want to have something like instagram mask filters:

![Mask 1](/images/introduction/insta_mask1.jpg)
![Mask 2](/images/introduction/insta_mask2.jpg)
![Mask 3](/images/introduction/insta_mask3.jpg)
![Mask 4](/images/introduction/insta_mask4.jpg)
![Mask 5](/images/introduction/insta_mask5.jpg)
![Mask 6](/images/introduction/insta_mask6.jpg)
#### Drawbacks: 
- resulting dataset can be not very realistic;
- requires using additional techniques as AR or 3D face reconstruction;
#### Benefits:
- can be fast and cheap way to make a dataset.

### 4.Create a synthetic dataset by applying GAN to the existing dataset.  
For this approach we want to use GAN to "add" masks on images. [Here](https://medium.com/using-deep-learning-dc-gan-to-add-featured-effect/recently-i-started-the-creative-applications-of-deep-learning-with-googles-tensorflow-of-parag-k-14453b215d2b) based on [8] it is shown how sunglasses can be easily added to any face image. Similarly, we can try to add a mask to each face image.
Another possibility is to use results of [this](https://www.cc.gatech.edu/~hays/7476/projects/Avery_Wenchen/) project, where authors were using Generative Adversarial Denoising Autoencoder for Face Completion. The difference is that in our case we need to complete a face with the mask.

#### Drawbacks: 
- training a GAN can be very complicated task;
- in case of bad training GAN can produce not realistic images;
#### Benefits:
- relatively fast and cheap method;
- can geberate more realistic images than previous approach.


## Overview
Besides all proposed approaches, we will mostly concetrate our attention on the last two.
The first and the second approaches are too time consuming, however we may also try to collect some real data.

[1]: https://doi.org/10.6028/NIST.IR.8311 "Ongoing Face Recognition Vendor Test (FRVT) Part 6A: Face recognition accuracy with masks using pre- COVID-19 algorithms"
[2]: https://arxiv.org/abs/2009.09960 "Towards Fast, Accurate and Stable 3D Dense Face Alignment"
[3]: https://arxiv.org/abs/1903.08527 "Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set"
[4]: https://arxiv.org/abs/1612.04904 "Regressing Robust and Discriminative 3D Morphable Models with a very Deep Neural Network"
[5]: https://arxiv.org/abs/1804.01005 "Face Alignment in Full Pose Range: A 3D Total Solution"
[6]: https://arxiv.org/abs/1803.07835 "Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network"
[7]: https://arxiv.org/abs/1703.07834 "Large Pose 3D Face Reconstruction from a Single Image via Direct Volumetric CNN Regression"
[8]: https://arxiv.org/abs/1512.09300 "Autoencoding beyond pixels using a learned similarity metric"
