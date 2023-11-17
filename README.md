# Skill Assisessment-Handwritten Digit Recognition using MLP
## Aim:
       To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## Theory:


    Introduction: The "Digit Recognition using Artificial Neural Networks (ANN)" project aims to create an advanced system capable of recognizing and classifying handwritten digits. By leveraging the power of machine learning, specifically Artificial Neural Networks, the project endeavors to accurately identify digits ranging from 0 to 9.

    Dataset: The project utilizes the widely recognized MNIST dataset, a staple in the machine learning community. Comprising a collection of 28x28 grayscale images of handwritten digits, the dataset also includes corresponding labels, making it an ideal resource for training and testing the neural network.

    Artificial Neural Network (ANN): The architecture of the Artificial Neural Network comprises multiple layers, including the input layer, hidden layers, and the output layer. By employing a combination of feedforward and backpropagation techniques, the network is designed to learn the intricate patterns and nuances within the dataset.

    Implementation Steps:

    i) Data Preprocessing: The initial step involves the normalization of pixel values and the appropriate formatting of labels, ensuring the data is conducive for training the neural network.

    ii) Model Architecture: The ANN's architecture is meticulously crafted, considering the specific number of layers, neurons, and activation functions. The selection of these components is critical to the network's overall performance and accuracy.

    iii) Model Training: The model is trained using mini-batch gradient descent and backpropagation. These optimization techniques are essential in fine-tuning the network's parameters and enhancing its ability to accurately recognize and classify handwritten digits.

    iv) Model Evaluation: The project employs various metrics, including accuracy, precision, recall, and the F1 score, to comprehensively evaluate the model's performance and its ability to make accurate predictions.

    v) Model Deployment: The final model is deployed with a user-friendly interface, allowing users to input their own handwritten digits for real-time recognition and visualization of the model's predictions.

    Conclusion: In summary, the "Digit Recognition using Artificial Neural Networks (ANN)" project showcases the prowess of deep learning in accurately classifying handwritten digits. By demonstrating the application of ANN in image recognition tasks, the project lays the foundation for further exploration and advancement in the field of computer vision and deep learning.

## Algorithm :

    Load the MNIST dataset containing handwritten digit images and labels.
    Preprocess the dataset
    Design the architecture of the Artificial Neural Network:
    Define the number of layers, neurons in each layer, and the activation functions.
    Initialize the weights and biases for the neural network.
    Set the hyperparameters for training the model:
    Evaluate the trained model
    Deploy the model: Create a user-friendly interface for users to input their own handwritten digits. Implement the functionality to visualize the model's predictions in real time.
    Conclude the project, highlighting the success of the ANN in accurately recognizing and classifying handwritten digits.

## Program:
NAME : Maithreyan D
REF NO : 212222220021

## DEPENDENCIES:
```
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.models import load_model
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
## LOADING AND DATA-PREPROCESSING:
```
(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train[0].shape
x_train[0]
plt.matshow(x_train[7])
y_train[7]
x_train_flattened=x_train.reshape(len(x_train),28*28)
x_test_flattened=x_test.reshape(len(x_test),28*28)
```
## NETWORK ARCHITECTURE:
```
model = Sequential()
model.add(Conv2D(32,(3,3), input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax"))
```
## TRAINING - VALIDATION:
```
model.compile(optimizer='adam',loss=tf.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
model.summary()
f=model.fit(x_train,y_train,epochs=5, validation_split=0.3)
f.history
```
## VISUALIZATION:
```
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(f.history['loss'], color = 'green', label='loss')
plt.plot(f.history['val_loss'], color = 'orange', label = 'val_loss')
fig.suptitle('LOSS', fontsize=20)
plt.legend(loc='upper left')
plt.show()
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(f.history['accuracy'], color = 'green', label='accuracy')
plt.plot(f.history['val_accuracy'], color = 'orange', label = 'val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()
```
## TESTING:
```
prediction = model.predict(x_test)
print(prediction)
print(np.argmax(prediction[0]))
plt.imshow(x_test[0])
```
## SAVING THE MODEL:
```
model.save(os.path.join('model','digit_recognizer.keras'),save_format = 'keras')
```

## PREDICTION:
```
img = cv2.imread('test.png')
plt.imshow(img)
rimg=cv2.resize(img,(28,28))
plt.imshow(rimg)
rimg.shape
new_model = load_model(os.path.join('model','digit_recognizer.keras'))
new_img = tf.keras.utils.normalize(rimg, axis = 1)
new_img = np.array(rimg).reshape(-1,28,28,1)
prediction = model.predict(new_img)
print(np.argmax(prediction))
new_img.shape
```
## Output :
## MODEL SUMMARY:
![283532966-212daaa8-9fd0-4cad-aaf6-866f97563948](https://github.com/MaithreyanDinakaran/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/119104032/d7743167-1524-4ba7-ab32-83d7e49e9287)
## TRAINING LOGS:
![283532990-b57ac0e4-9a9b-432f-a14f-a783d4945aeb](https://github.com/MaithreyanDinakaran/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/119104032/a2add973-a6de-4941-8813-611f051fae8f)
## ACCURACY AND LOSS PERCENTILE:
![283533017-c88d5825-4a0e-4825-a1f8-7d698b1186bc](https://github.com/MaithreyanDinakaran/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/119104032/0d895021-ba20-4f92-911c-c8bacb352087)

![283533057-97dd7edb-f10f-40e2-9e68-01de7e33f0e5](https://github.com/MaithreyanDinakaran/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/119104032/53eafdcc-2391-4e0a-adb2-25c1fa08292a)
## PREDICTION:

![283533078-5b4bc555-8f3c-42e5-9667-0bbaf9821bb5](https://github.com/MaithreyanDinakaran/Ex-6-Handwritten-Digit-Recognition-using-MLP/assets/119104032/480c6c83-c9eb-4276-96c5-b55681e8896b)

## Result:
Thus The Implementation of Handwritten Digit Recognition using MLP Is Executed Successfully.
