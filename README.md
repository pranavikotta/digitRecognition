# digitRecognition
# MNIST Digit Classification with Webcam Integration

This project explores deep learning through the application of a Multi-Layer Perceptron to classify handwritten digits from the MNIST dataset. The model is trained using TensorFlow and Keras to achieve high accuracy in digit classification. The training pipeline encompasses data preprocessing, normalization, and splitting into training, validation, and test datasets. It employs techniques such as dropout, batch normalization, and early stopping to prevent overfitting and improve model performance.

The project consists of two main components:
1. **Training Script**: This script trains the neural network on the MNIST dataset. It includes model creation, compilation, training, and evaluation. The merit of the MLP is assessed through the graphical visualization of error and the creation of confusion matrix, following which the model is saved for later use.
2. **Webcam Integration**: This script avails of OpenCV to capture real-time images from a webcam, preprocesses them, and classifies the digits using the trained model. The predicted digit is displayed on the live video feed.


## Dependencies
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
