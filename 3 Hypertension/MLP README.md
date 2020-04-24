# MLP-Hypertension-Prediction
Implemented a MLP neural network to predict hypertension from patient data


## Neural Network with Tensorflow(Keras) 

Sequential Multi Layer Percepton (MLP) neural network built using Keras, containing: 


### Model Flow 

### Data Preprocessing
- collect and explore data
- normalize input features 
- split into 5 folds cross validation train/test datasets 
    
   
### Model Construction 
- Utilize Kera's sequential class to construct input, hidden, and output layers. Model consists of performaning matrix multiplications during each level and producing an activation output using mathematical functions. 
    
- Input layer: 
    - Takes feature inputs and intialize with random weights and biases 
    - Utilize activation function (ReLU, tanh, etc) to produce inputs for hidden layers
        
- Hidden Layers:
    - Decide on number of hidden layers and different parameters 
    - Takes our input data and performs mathematical operations to produce inputs to output layer

- Output Layer: 
    - Final step in neural network should produce a probability between 0-1 to classify 
        
   
### Model training 
- We can train (fit) our data using the constructed model. The network will iterativly(number of epochs) train and try to improve it's performance. Uses an optimizer in attempt to reduce the training loss, process know as gradiant descent, and achieve higher accuracy. 
    
- Loss Function:
    - Since this is a binary classification problem, use binary_crosscentropy to calculate the loss function between predicted and actual output
        
- Optimization: 
    - We optimize the neural network with an Adam optimizer.
    - Adam = Adaptive moment estimation, combination of RMSProp + Momentum
        
- Stochastic Gradiant Descent: 
    - Momentum takes past gradiants into account to smooth out the gradiant descent
    
    
### Model Evaluation 
- Confusing Matrix: Using Sklearn metrics library

    True Negative     False Positive 
    False Negative    True Positive 
    
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Accuracy needs to be considered depending on the problem
    
- Recall: TP / (TP + FN)
    - High recall indicates the class is correctly recognized
    - E.g. small number of false negatives
    
- Precision: TP / (TP + FP)
    - High precision indicates an example labled as positive 
        is indeed positive 
    
- F-Score: 2*recall*precision / (recall + precision)
    - F-score helps measure recall and precision at same time
    - It takes harmonic mean instead of arithmitic mean


### Area Under the Curve (AUC): Performance Measurement 

- calculating the AUC using the trapezoidal rule for the 
    ROC-curve
- ROC is a probability curve and AUC represents degree of separability
- Shows how much a model is capable of distinguishing between classes
    - Higher AUC, better at predicting 0 as 0, 1 as 1
    - E.g. better at disease diagnosis

- TPR (True pos rate) / Recall / Sensitivity = TP / (TP+FN)
- FPR (False pos rate) = 1 - Specificity
- Specificity = TN / (TN + FP)

- Sensitivity and Specificity are inversely proportional to each other
