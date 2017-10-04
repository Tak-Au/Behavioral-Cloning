My project includes the following files:
* Model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* NvidiaModel.09.h5 containing a trained convolution neural network 

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py NvidiaModel.09.h5
```

####3. Submission code is usable and readable

The Model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model that was used is the same as the one Nivida used to train their self driving car (model.ipynb line 10).  The table below shows the summary of the architect. 


|Layer  | Output Shape  |  Param numbers  |  Connected to|
|-------|---------------|-----------------|--------------|
|cropping2d_1 (Cropping2D) |(None, 65, 320, 3)|0| cropping2d_input_1[0][0]|
|lambda_1 (Lambda)                |(None, 65, 320, 3)    |0           |cropping2d_1[0][0] 
|convolution2d_1 (Convolution2D)  |(None, 31, 158, 24)   |1824        |lambda_1[0][0]   
|activation_1 (Activation)        |(None, 31, 158, 24)   |0           |convolution2d_1[0][0]            
|convolution2d_2 (Convolution2D)  |(None, 14, 77, 36)    |21636       |activation_1[0][0]               
|activation_2 (Activation)        |(None, 14, 77, 36)    |0           |convolution2d_2[0][0]            
|convolution2d_3 (Convolution2D)  |(None, 5, 37, 48)     |43248       |activation_2[0][0]               
|activation_3 (Activation)        |(None, 5, 37, 48)     |0           |convolution2d_3[0][0]            
|convolution2d_4 (Convolution2D)  |(None, 3, 35, 64)     |27712       |activation_3[0][0]               
|activation_4 (Activation)        |(None, 3, 35, 64)     |0           |convolution2d_4[0][0]            
|convolution2d_5 (Convolution2D)  |(None, 1, 33, 64)     |36928       |activation_4[0][0]               
|activation_5 (Activation)        |(None, 1, 33, 64)     |0           |convolution2d_5[0][0]            
|flatten_1 (Flatten)              |(None, 2112)          |0           |activation_5[0][0]               
|dense_1 (Dense)                  |(None, 1164)          |2459532     |flatten_1[0][0]                  
|dropout_1 (Dropout)              |(None, 1164)          |0           |dense_1[0][0]                    
|dense_2 (Dense)                  |(None, 100)           |116500      |dropout_1[0][0]                  
|dropout_2 (Dropout)              |(None, 100)           |0           |dense_2[0][0]                    
|dense_3 (Dense)                  |(None, 50)            |5050        |dropout_2[0][0]                  
|dropout_3 (Dropout)              |(None, 50)            |0           |dense_3[0][0]                    
|dense_4 (Dense)                  |(None, 10)            |510         |dropout_3[0][0]                  
|dense_5 (Dense)                  |(None, 1)             |11          |dense_4[0][0]  

Total params: 2,712,951
Trainable params: 2,712,951
Non-trainable params: 0


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
