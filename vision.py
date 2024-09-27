

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

def valida_item():
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("C:/Users/felip/PycharmProjects/Projeto_Eco/Model cnn/models/cnn1_vf.h5", compile=False)

    # Load the labels
    class_names = {}
    with open("C:/Users/felip/PycharmProjects/Projeto_Eco/Model cnn/models/labels.txt", "r") as file:
        for line in file:
            index, label = line.strip().split(maxsplit=1)
            class_names[int(index)] = label

    print("Class names:", class_names)

    # Initialize the camera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Camera not accessible")
        return

    while True:
        # Grab the webcamera's image
        ret, image = camera.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize the image to the expected input size for the model
        image_resized = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)

        # Show the image in a window
        cv2.imshow("Webcam Image", image_resized)

        # Prepare the image for prediction
        image_array = np.asarray(image_resized, dtype=np.float32)
        image_array = (image_array / 255.0)  # Normalizing to [0,1]
        image_array = image_array.reshape(1, 150, 150, 3)

        # Predict using the model
        prediction = model.predict(image_array)
        print("Prediction probabilities:", prediction)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score
        print("Class:", class_name, end=" ")
        print("Confidence Score:", str(np.round(confidence_score * 100, 2)) + "%")

        # Listen to the keyboard for presses
        keyboard_input = cv2.waitKey(1)

        if (class_name == "glass" and confidence_score > 0.95):
            print("Detected class 'glass' with high confidence.")
            return 1

        if (class_name == "metal" and confidence_score > 0.95):
            print("Detected class 'metal' with high confidence.")
            return 2

        if (class_name == "plastic" and confidence_score > 0.95):
            print("Detected class 'plastic' with high confidence.")
            return 3

        # 27 is the ASCII for the ESC key
        if keyboard_input == 27:
            print("Exiting...")
            break

    # Release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()