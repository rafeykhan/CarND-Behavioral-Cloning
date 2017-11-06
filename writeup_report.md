# **Behavioral Cloning For Autonomous Cars** 

## Writeup 

---

**Goals**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results in this report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

**Description of simulator**

A car simulator by Udacity is used for the purposes of generating data and testing. The simulator contains 2 modes: Training Mode and Autonomous Mode.

The training mode is used to drive the car manually and collect data of good driving behavior. The simulator stores images from the 3 dashboard cameras which are used as training features for the neural network. Apart from the images, the simulator also produces a csv file that contains the image names and corresponding steering angles, throttle, braking and speed. For this project, only the steering angles are used as training labels.

The autonomous mode is used for testing the trained model. It passes the images into model and steers the car based on the prediction form the network.

**Project files**

My project includes the following files and can be used to run the simulator in autonomous mode:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* IPython notebook for visualization

**Running the simulator**
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

**Code**

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

---
### Model Architecture and Training Strategy

#### 1. Model architecture employed

The model architecture is based on the one detailed in NVIDIA's paper 'End to End Learning for Self-Driving Cars'. It consists of a convolution neural network with 5 convolutional layers and 3 fully-connected layers. 

![alt text][image1]

Image processing layers are included in the architecture so that the autonomous driving images are processed accordingly as well. The images are cropped to only keep useful details and then normalized and resized as per NVIDIA's approach. The model is tweaked by adding batch normalization as it is shown to perform better in recent research.
The model includes ELU layers to introduce nonlinearity as it results in faster convergence of the network.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different and large enough data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Initial implementations included dropout layers in order to combat overfitting, but later replaced by batch normalization. The inventors of batch norm detail in their research paper that using batch norm performs equally well, if not better, than using dropout with regularization. In practice networks that use Batch Normalization are significantly more robust to bad initialization. Additionally, it can be interpreted as doing preprocessing at every layer of the network.

#### 3. Model parameter tuning
(model.py line 25)
The model used an adam optimizer, so the learning rate was not tuned manually. Learn rate was initialized at 0.001 for training. However, for fine-tuning the trained model, learn rate was lowered to 0.0001

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, smooth as well as erratic driving (this helps learn instances where the car overshoots a sharp turn but has to recover). 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with existing design approaches and tweak them to fit the given scenario. The two existing models used are outlined by CommaAI and nVidia.

My first step was to use a convolution neural network model similar to the CommaAI's. I thought this model might be appropriate because they are being implemented in their devices and are relatively smaller. After training, I came to the conclusion that the nVidia's model performed better and learnt useful features. One downside was that it takes longer to train.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set and monitor the metrics. Having a low mean squared error on the training set but a high mean squared error on the validation set implies that the model is overfitting. 

To combat the overfitting, large and varying datasets were used. The model was trained with fewer epochs to help with generalization. Modifying the model so that it used dropout with L2 regularization resulted in poor performance (will be investigating on this). 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track perhaps due to fewer training features for that scenario. To improve the driving behavior in these cases, the learn rate was lowered and the network was then fine-tuned with additional selective training features. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

Most of the training data comprised of images from the reverse route so as to keep the training and testing data separate as much as possible. This helps to identify how well the network generalizes and performs given a new scenario. 

To capture good driving behavior, a dataset was generated on track one using center lane driving with smooth turns (using a joystick instead of keyboard helped). Here is an example image of center lane driving:

![alt text][image2]

Second dataset was recorded with the vehicle driving in a erratic manner, recovering from the left side and right sides of the road back to center. This allows the vehicle to learn to recover in cases if its going off-road. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then the process was repeated to generate selective data, that is data of areas where the car had trouble originally navigating.

To augment the data, images from the left and right cameras were used with a correction factor. This approach helps the car learn to steer back to the center of the lane if it goes sideways.
Another technique to augment the data set is to flip the images and angles, essentially artificially creating unique situations. For example, here is an image that has been flipped:

![alt text][image6]
![alt text][image7]

The data is then altered by removing the zero angle instances from the erratic and selective datasets, leaving the smooth driving set untouched. This helps reduce the bias for going straight, but not completely nulled.

Finally, the data set is randomly shuffled and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by stagnation of validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
