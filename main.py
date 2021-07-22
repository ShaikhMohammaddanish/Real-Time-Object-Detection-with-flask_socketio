from flask_socketio import SocketIO
from flask_socketio import send, emit
from flask import Flask, render_template,request
from flask import jsonify
from io import StringIO  
from PIL import Image
import base64 
import io
import cv2
import numpy as np
import imutils
import json
import requests
import time
import os
import glob


labelsPath = 'pretrain_model/yolov3.txt'
LABELS = open(labelsPath).read().strip().split("\n")


def detectBoundigBox(image, netDetection, confidenceVal=0.5,threshold=0.3,networkImgSize=(32,32)):
    '''Detect boundign box in image and detect color '''
    

    
    net = netDetection
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (224, 224),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)


    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidenceVal:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)


    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidenceVal,threshold)
    
    # ensure at least one detection exists


    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])


            color = [225,225,225]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])

            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)

    return image






app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode=None)

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')







@socketio.on('connect')
def test_connect():
    print('connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')


@socketio.on('image')
def image(data_image):



    # decode and convert into image
    b = io.BytesIO(base64.b64decode(data_image))
    pimg = Image.open(b)



    ## converting RGB to BGR, as opencv standards
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    netDetection = cv2.dnn.readNetFromDarknet("pretrain_model/yolov3.cfg","pretrain_model/yolov3.weights"  )
    frame = detectBoundigBox(frame,  netDetection)

    # Process the image frame
    frame = imutils.resize(frame, width=700, height=1400)
    # frame = cv2.flip(frame, 1)
    imgencode = cv2.imencode('.jpg', frame)[1]

    # base64 encode
    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', json.dumps({"stringData":stringData}) )




if __name__ == "__main__":
    print('[INFO] Starting server at http://localhost:5000')
    # socketio.run(app=app, host='192.168.1.4', port=5000, debug=True)
    socketio.run(app=app, host='127.0.0.1', port=5000, debug=True)