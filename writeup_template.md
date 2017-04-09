#**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./results/visualization.png "Visualization"
[image2]: ./new_images/0.jpg "Traffic Sign 1"
[image3]: ./new_images/1.jpg "Traffic Sign 2"
[image4]: ./new_images/2.jpg "Traffic Sign 3"
[image5]: ./new_images/3.jpg "Traffic Sign 4"
[image6]: ./new_images/4.jpg "Traffic Sign 5"
[image7]: ./new_images/5.jpg "Traffic Sign 6"
[image8]: ./new_images/6.jpg "Traffic Sign 7"
## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/crack00ns/SLD-Traffic-Signs-Detection-Project2)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 69598
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the traffic signs classes are distributed.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At first I augmented training data by additional training data by adding small and random sheer, rotation, translation to all images. This doubles my training data set. This is done in ipython code cell [1] and the corresponding function is in util.py
As a second step, I decided to convert the images to grayscale because after continuous testing, I was able to get a slightly better performance with grayscale images than color images. As a last step, I normalized the image data with min_max normalization rather than center normalization because of performance benefit (probably due to smaller variance). Code cell [4] describes the process.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5x6     | 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5x16    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Flatten               | 400 output                                    |
| Fully connected	    | 400-> 1024 output  							|
| RELU                  |                                               |
| Dropout               | keep prob = 0.5                               |
| Fully connected       | 1024-> 512 output                             |
| RELU                  |                                               |
| Dropout               | keep prob = 0.5                               |
| Logits Layer          | 512-> 43 output                               |
| Softmax				| etc.        									|
|						|												|

On top of that I added L2 regularization with all the weights of the layers above  

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam Optimizer, Batch size 128, number of epocs 50, L2 regularization paratemer 1e-6. I used a weight initialization scheme of truncated normal with zero mean and std deviation 2/sqrt(number of inputs).

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 97% 
* test set accuracy of 94.5%

I first used LeNet model by changing number of output units from 10 to 43 (number of classes). LeNet model is a successful model for MNSIT so it was a natural choice. The accuracry with 20 epochs was around 90-91%. Thereafter the I changed the model by increasing the number of hidden variables in full connected layers and increasing the model size. This increased the accuracy a little bit. Next I tried dropouts in the convolution layers but there was little change in performance. So I tried dropout in fully connected layers which seemed to fare better. Finally I added L2 regularization parameter which booseted the validation accuracy to 94-95% and test accuracy was around 93%. As a last step, I changed images to grayscale and tried different normalization schemes and augmented data which boosted the validation accuracy to 97% and test accuracy close to 95%. Changing the weight initialization scheme helped the model converge faster with fewer epocs. A final test on images downloaded from internet verfied that the model is working well by successfully classifying all the downloaded images correctly.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image2] ![alt text][image3] ![alt text][image4] 
![alt text][image5] ![alt text][image6] ![alt text][image7]
![alt text][image8]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work    									| 
| Roundabout mandatory  | Roundabout mandatory 							|
| Stop					| Stop											|
| Children crossing	    | Children crossing					 			|
| Keep right			| Keep right      							    |
| Speed limit (60km/h)  | Speed limit (60km/h)                          |
| Priority road         | Priority road                                 |


The model was able to correctly guess 7 out of the 7 traffic signs, which gives an accuracy of 100%. However, if I regenerate the model accuracy sometimes could go down.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the cell [13] of the Ipython notebook. All the softmax probabilites are shown in form of a bar chart. Also top 5 softmax proabilities are shown in Cell [15]. The top probabilities are all more than 0.8 for all signs

