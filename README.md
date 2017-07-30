#**Behavioral Cloning Project** 

---

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[data_original]: ./examples/data_original.png "Data original"
[data_balanced]: ./examples/data_balanced.png "Data reduced"
[data_visualizer]: ./examples/data_visualizer.png "Data visualizer"
[cnn]: ./examples/cnn.png "CNN"
[center_lane]: ./examples/center_lane.png "Center lane driving"
[left]: ./examples/left.jpg "Recovery right to left (left camera)"
[center]: ./examples/center.jpg "Recovery right to left (center camera)"
[right]: ./examples/right.jpg "Recovery right to left (right camera)"

[fs]: ./examples/frame_shadow.png  "Image with shadow"
[fsf]: ./examples/frame_shadow_flipped.png "Image with shadow flipped"
[fsbw]: ./examples/frame_shadow_bw.png "Image with shadow grayscaled"
[fsfbw]: ./examples/frame_shadow_flipped_bw.png "Image with shadow grayscaled and flipped"
[fa]: ./examples/frame_adjusted_sv.png "Image with raised S and V values"
[fabw]: ./examples/frame_adjusted_bw.png "Image with raised S and V values grayscaled"

---
###Files & Code

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

This project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* data.py containing the script to preprocess the input (images) of the convolution neural network
* model.h5 containing a trained convolution neural network 
* utils.py containing the support functions for a multithreaded data generator
* [repository](https://github.com/ValeryToda/DataVisualizer) containing all files used to visualize the data offline in order to perform data sanity checks and select relevant data.
* writeup\_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track /examples/by executing 

```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The files drive.py and data.py show the pipeline I used for training and validating the model, and they contain comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of an adapted version of the convolution neural network described in this [paper](https://arxiv.org/pdf/1604.07316v1.pdf) used by the NVIDIA team in a similar project. The adaptation consist of the inclusion of Keras batch normalisation, dropout and relu layers. The model includes relu layers to introduce nonlinearity.

####2. Attempts to reduce overfitting in the model

The model contains dropout and batch normalisation layers in order to reduce overfitting. 
The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. But depending on the training setting the learning rate can automatically be reduced (divided by 2) after 10 consecutive epochs if the loss is not decreasing.

```
# model.py: train(...)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                    patience=10, min_lr=0.1e-6, verbose=1)
```

####4. Appropriate training data


Training data was chosen to keep the vehicle driving on the road. I used a combination of center, right and left lane driving and recovering from the left and right sides of the road. After driving in track 1 and 2 normally, i then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from unwanted positions. Then I used this [tool](https://github.com/ValeryToda/DataVisualizer):

![][data_visualizer]

*Data visualizer*

to go trough the recorded frames and drop the frames with the vehicle off the road. The right and left lanes data were added to the center lanes data by adding a steering angle correction of 0.2 and -0.2 respectively to the left and the right lanes steering angles. The picture below shows the distribution of the steering angles after the combination.


![][data_original]

*Data (steering angles) distribution*

```	
	# data.py: balance_data(...)
	# Drop 60% of data with steering angles between -0.23 and -0.19
    df = df.drop(df[(-0.23 < df['angles']) & (df['angles'] < -0.19)].sample(frac=0.6).
```

The data was then reduced to get a balanced set of 31446 training data with predominant steering angles around in the interval [-0.25  0.25]:

![][data_balanced]

*Training data after reduction*

###Model Architecture and Training Strategy

####1. Solution Design Approach


My first step was to use a convolution neural network model similar to the model I implemented in Project 2. I thought this model might be appropriate. That model was not able to approximatively predict good steering angles. I then switched to the NVIDIA model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model and added batch normalisation and dropout layers so that the model was already able to drive well in track 1 after just two training epochs.

The final step was to run the simulator to see how well the car was driving around track one. There was one spot (end of the first round) where the vehicle fell off the track. To improve the driving behavior in that case, I recorded more data there and added them to the training data set.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consists of a convolution neural network. Here is a visualization of the architecture:

![alt text][cnn]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_lane]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from unwanted positions

![alt text][left]
![alt text][center]
![alt text][right]

To augment the data set, I also flipped images and angles and added random shadow thinking that this would help the model to deal with spots with low lightning conditions. The images were cropped to get rid of non relevant parts for the training. The S and V channels of some images were raised as stated [here](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html). The data were approximatively doubled on the fly within the generator. To speed up the training the images were resized and convert to grayscale. Here are some examples.

![alt text][fs]
![alt text][fsf]
![alt text][fsbw]
![alt text][fsfbw]
![alt text][fa]
![alt text][fabw]

I finally randomly shuffled the data set and put 10% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over- or underfitting. The ideal number of epochs was 3 with adam optimizer without learning rate optimization.
