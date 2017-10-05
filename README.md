My project includes the following files:
* Model.ipynb containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* Model.h5 containing a trained convolution neural network 

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py Model.h5
```

####3. Submission code is usable and readable

The Model.ipynb file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model that was used is the same as the one Nivida used to train their self driving car.  Refer to Final Model Architecture for more detail.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 132,134, and 136). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 86). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 143).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I used varies known neural network architecture to come up with the best one.  First I did features selection via cropping of image top portion and bottom portion that are not relevant to the road.  
Then I also added a lamda layer which allows custom function to apply on incoming data.  For this lamda function, I normalized the data by mean centering and scaling it so that the data will output -0.5 and +0.5.  

I started with using LeNet architect.  I modified the last layer to only output 1 result so that it can learn the steering angle through regression.  Then I train the network via the training set and validation set that I obtain by spliting the driving data to 80% train and 20% validation.

I also tried to improve the network robustness by generating more data.  There are 3 cameras which captures images simultaneously(Left, Center, and Right).  The center image will not required any steering correction factor since the image is center to the car.  However, the image to from the right and left of the car distort the road. If we were to use the left and right image directly without steering angle adjustment, the car will turn aggressively.  I used trial and error to come up with the steering angle adjustment and it appears that .28 to be the best value. 

After using LeNet network without sucess, I went for the Nvidia Neural network architect that they used for their Self driving car.  After small adjustment to accomdinate for training image cropping and normalization, the network was able to drive the track.   


####2. Final Model Architecture
The final model architecture (model.py lines 107-138) consisted of the following archtiecture:

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

Here is a visualization of the architecture 

![Nvidia Self Driving Car Neural Network](https://github.com/Tak-Au/Behavioral-Cloning/blob/master/cnn-architecture-624x890.png)

Source: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

####3. Creation of the Training Set & Training Process

To create training and validation set, I drive the track twice in both directions.  I tried to drive the car as center as possible.  Then I split the data into training set and validation at 80% and 20% split at random.  Then I train the NN with this and see how the car performed.  

I used Adam optimizer for the training process.  I used keras checkpoint function to generate model on every epoche.  I ran each model to see which one does the best.  I find this to be the best way to pick the best model.  From the best model, I examine where are some of the problematic areas.  Once I identify them, I go back to the simulator and drive to the problematic area and drive the car back from left/right and steer back to the center.  Eventually, the Neural network was able to drive the car consistently.  


