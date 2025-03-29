import cv2
import numpy as np
from tensorflow import keras

#retrieve trained neural network
model = keras.models.load_model('mlp-model-digitclassification.h5')

cam = cv2.VideoCapture(0) #opens default camera
while True:
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow('Webcam Feed', frame) #creates a window where the frame can be recorded
    key = cv2.waitKey(1) & 0xFF #checks for key presses

#capture the frame if enter is pressed
    if key == 32:
        frame_resized = cv2.resize(frame, (28, 28))  #resize to 28x28
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)  #convert to grayscale
        frame_normalized = frame_gray / 255.0  #normalize pixel values to [0, 1]

        # Add batch dimension (shape: (1, 28, 28, 1))
        frame_input = np.expand_dims(frame_normalized, axis=-1) #add channel dimension
        frame_input = np.expand_dims(frame_input, axis=0) #batch dimension

        prediction = model.predict(frame_input)
        predicted_value = np.argmax(prediction)  #retrieve index of the highest probability

        cv2.putText(frame, f"Predicted: {predicted_value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        print(f"Predicted Digit: {predicted_value}")

    elif key == 27: #if escape key is pressed
        break    

#release the webcam and close OpenCV windows
cam.release()
cv2.destroyAllWindows()