import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow import keras as k
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt

(x,y), (x_test, y_test) = k.datasets.mnist.load_data() #obtains dataset and splits into training and test data
#normalizes image pixel data to 0-1 range, one hot encoding for labels to represent as vectors ie. 0 = [1,0,0,...]
x, y = x/255.0, k.utils.to_categorical(y, 10)
x_test, y_test = x_test/255.0, k.utils.to_categorical(y_test,10)

x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.2) #splits into validation and training data

#prevents overfitting due to epochs by restoring the model after an ideal loss curve has been achieved
early_stopping = k.callbacks.EarlyStopping(
    min_delta = 0.01,
    patience = 4,
    restore_best_weights = True,
)

model = k.Sequential([
    k.layers.Flatten(input_shape=(28,28)), #input layer
    k.layers.Dense(128, activation='relu'), #hidden layer
    k.layers.BatchNormalization(), #normalizes output of activation func to simplify learning
    k.layers.Dropout(0.3), #randomly drops 30% of neurons to prevent overfitting
    k.layers.Dense(128, activation='relu'), #hidden layer
    k.layers.BatchNormalization(),
    k.layers.Dropout(0.3),
    k.layers.Dense(10, activation='softmax') #output layer --> softmax creates probabilities, 10 classes to match with
])

model.compile(
    optimizer = 'adam', #popular optimizer
    loss = 'categorical_crossentropy', #suited for multi-class classification IS THIS SPELLED RIGHT
    metrics = ['accuracy'] #tracks model accuracy
)

history = model.fit(
    x_train, y_train, #training data
    validation_data = (x_validation, y_validation),
    batch_size = 128,
    epochs = 30, #feeds 128 rows of data 30 times
    callbacks = early_stopping
)

#test model and output results
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

#to visualize loss and check whether additional epochs are necessary
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.title('Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#confusion matrix - compares predictions to actual values
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1) #distinguishes the class the prediction vector falls under ie. 0-9
actual_labels = np.argmax(y_test, axis=1)

confusion_matrix = metrics.confusion_matrix(actual_labels, predicted_labels)
#display matrix as a heatmap
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('predicted classification')
plt.ylabel('actual value')
plt.title('confusion matrix for digit classification neural network')
plt.show()

model.save('mlp-model-digitclassification.') #saves trained model as h5 file for use in camerascanner.py