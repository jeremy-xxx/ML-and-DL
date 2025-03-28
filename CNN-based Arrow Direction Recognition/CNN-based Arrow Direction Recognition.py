import cv2
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
import numpy as np
import pickle

sampleX_set = []
sampleY_set = []
IMG_SIZE = 16
bCNNTrained = False
NN = ()

# Add sample data (image and corresponding label) to the dataset
def addSampleData(frame, label):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    frame_data = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))  # Resize frame to fixed size
    print(frame_data)
    sampleX_set.append(frame_data)  # Add image data to X set
    sampleY_set.append(label)  # Add label to Y set
    print("Add sample data" + label)

# Train the neural network (CNN)
def trainCNN():
    global bCNNTrained, NN
    # Step 2: Load dataset
    dataset = load_digits()

    # Step 4: Create the Neural Network Classifier
    NN = MLPClassifier()

    num, nx, ny = np.asarray(sampleX_set).shape  # Get shape of sample set
    print(num, nx, ny)

    X = np.array(sampleX_set).reshape(num, nx * ny)  # Reshape data for training
    Y = np.asarray(sampleY_set)  # Convert labels to numpy array

    # Step 5: Train the model
    NN.fit(X, Y)

    bCNNTrained = True
    saveCNNModel()  # Save the trained model
    print("Training CNN Done!")


# Save the trained CNN model to a file
def saveCNNModel():
    global NN
    # Save the trained model to disk
    with open('model.pkl', 'wb') as f:
        pickle.dump(NN, f)
        print("Model saved")


# Load the pre-trained CNN model from a file
def loadCNNModel():
    global bCNNTrained, NN
    # Load the model from disk
    with open('model.pkl', 'rb') as f:
        NN = pickle.load(f)
        bCNNTrained = True
        print("Model Loaded")


# Predict the direction of the arrow in the frame
def predictCNN(frame):
    global bCNNTrained, NN
    y_pred = "Invalid"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    frame_data = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))  # Resize to match training size
    frame_data = np.array(frame_data).reshape(-1, IMG_SIZE * IMG_SIZE)  # Flatten for prediction

    # Predict the label using the trained model
    if bCNNTrained:
        y_pred = NN.predict(frame_data)
        print("Predict Result:")
        print(y_pred)
    else:
        print("Model not trained yet")
    return y_pred


bPredict = False


# Toggle the prediction state on or off
def togglePredict():
    global bPredict
    bPredict = not bPredict


# Handle user input for CNN operations (train, save, load, predict)
def handleCNN(frame, key, bArrowDetected):
    if key == ord('l') and bArrowDetected:
        addSampleData(frame, "Left")  # Add "Left" sample
    elif key == ord('r') and bArrowDetected:
        addSampleData(frame, "Right")  # Add "Right" sample
    elif key == ord('h') and bArrowDetected:
        addSampleData(frame, "Head")  # Add "Head" sample
    elif key == ord('p'):
        togglePredict()  # Toggle prediction
    elif key == ord('t'):
        trainCNN()  # Train the CNN model
    elif key == ord('s'):
        saveCNNModel()  # Save the trained model
    elif key == ord('o'):
        loadCNNModel()  # Load a pre-trained model


# Draw text on the frame at a specified position
def drawText(frame, text, rectPosition):
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
    x, y, w, h = cv2.boundingRect(rectPosition)  # Get bounding rectangle of detected object
    org = (x, y)  # Text origin position
    fontScale = 1  # Font scale
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 2  # Line thickness
    current_frame = cv2.putText(frame, text[0], org, font, fontScale, color, thickness, cv2.LINE_AA)  # Draw text


# Main function to track arrows and perform CNN-based prediction
def trackArrows():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Error: Webcam not accessible.")
        return

    rectPosition = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])  # Initialize rectangle position
    contour_tmp = ()
    detect_frame = ()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = cv2.resize(frame, (320, 240))  # Resize frame for processing

        # Convert frame to grayscale
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        rect_frame = current_frame.copy()
        line_frame = current_frame.copy()

        bArrowDetected = False

        # Apply thresholding to detect edges and contours
        _, thresh = cv2.threshold(gray, 120, 220, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            rectNum = 0

            # Define desired region of interest (ROI) points
            desired_roi_points = np.float32([[60, 40], [60, 80], [180, 80], [180, 40]])

            for contour in contours:
                # Approximate the contour to a polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                contArea = cv2.contourArea(contour)
                if len(approx) == 4 and contArea > 1000.0:
                    rectNum += 1
                    # Record the polygon approximation of the rectangle around the arrow
                    rectPosition = approx

            if rectNum == 1:
                # Draw the detected rectangle
                cv2.drawContours(rect_frame, [rectPosition], 0, (0, 255, 0), 2)
                x, y, w, h = cv2.boundingRect(rectPosition)
                roi_points = rectPosition - [x, y]
                crop_img = current_frame[y:y + h, x:x + w]

                # Perform perspective transformation
                desired_roi_points = np.float32([[240, 60], [240, 180], [80, 180], [80, 60]])
                transformation_matrix = cv2.getPerspectiveTransform(np.float32(roi_points), desired_roi_points)
                warped_frame = cv2.warpPerspective(crop_img, transformation_matrix, (320, 240))

                # Convert the warped frame to grayscale
                detect_frame = warped_frame[60:180, 80:240]
                warpGray = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2GRAY)

                # Threshold the grayscale image
                _, warpthresh = cv2.threshold(warpGray, 120, 220, cv2.THRESH_BINARY)
                warpcontours, _ = cv2.findContours(warpthresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if len(warpcontours) > 0:
                    for warpcontour in warpcontours:
                        warpContArea = cv2.contourArea(warpcontour)
                        if 1000 < warpContArea < 4000:  # Check for arrow contour size
                            bArrowDetected = True
                            contour_tmp = warpcontour

                cv2.imshow("detect_frame", detect_frame)

        key = cv2.waitKey(10)

        # Handle key events (quit, add data, etc.)
        if key == ord('q'):
            break
        else:
            handleCNN(detect_frame, key, bArrowDetected)

        # If an arrow is detected and prediction is enabled, show predicted result
        if bArrowDetected and bPredict:
            directionString = predictCNN(detect_frame)
            drawText(current_frame, directionString, rectPosition)

        # Display the results
        cv2.imshow('Original image', current_frame)
        cv2.imshow('Threshold image', thresh)
        cv2.imshow('Contours Detection', rect_frame)

    # Release resources after the loop
    cap.release()
    cv2.destroyAllWindows()


# Start tracking arrows
trackArrows()
