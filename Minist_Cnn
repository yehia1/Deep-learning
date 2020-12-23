import numpy as np 
import tensorflow as tf
from keras.utils import np_utils
#getting the minist data sett from tensor flow
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#scalling the data so it will be easier to train
x_train= x_train.reshape(x_train.shape[0],28,28,1)/255
x_test= x_test.reshape(x_test.shape[0],28,28,1)/255

cnn = tf.keras.models.Sequential()
# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3, padding="valid", activation="relu", input_shape=[28, 28,1]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))

# Step 5 - Output Layer
cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])


# Training the CNN on the Training set and evaluating it on the Test set
                                    
cnn.fit(x_train, y_train, batch_size=128, epochs=100, validation_data=(x_test, y_test))   

# first way to see accuracy
score = cnn.evaluate(x_test, y_test);
print(score[1])

# second way 
y_predict = cnn.predict_classes(x_test)

y_predict=list(y_predict)
y_test = list(y_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_predict,y_test))
