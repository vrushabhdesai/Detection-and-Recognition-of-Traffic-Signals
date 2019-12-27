import cv2
import numpy as np
import time

def status(X1):
   
    ''' This funtion declare if to Go, Slow Down and Stop '''
    
    Hsv= cv2.cvtColor(X1, cv2.COLOR_BGR2HSV)
    hsv.append(Hsv)
    countg,county,countr=0
    
    MASK1= cv2.inRange(Hsv,L_red1,u_red1)
    mask1.append(MASK1)
    
    MASK2= cv2.inRange(Hsv,L_red2,u_red2)
    mask2.append(MASK2)
    MASKG= cv2.inRange(Hsv, l_green,u_green)
    maskg.append(MASKG)
    
    MASKy= cv2.inRange(Hsv, l_yellow, u_yellow)
    masky.append(MASKy)
    MASKr= cv2.add(MASK1, MASK2)
    maskr.append(MASKr)
    dime1=np.shape(MASKr)
    dime.append(dime1)
    (H1,W1)= dime1
    for j in range(H1):
        for k in range(W1):
           if MASKr[j][k]==255:
               countr+=1
           if MASKG[j][k]==255:
               countg+=1
           if MASKy[j][k]==255:
               county+=1
    
    L= [0, countr, county, countg]
    final_color= L.index(max(L))
    
    if final_color==1:
        status="STOP"
    elif final_color==2:
        status="SLOW DOWN"
    elif final_color==3:
        status="GO"
    else:
        status=""
    return status


# Load Yolov3 model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]    
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# text color on image 
color= (200,0,200)


# Loading ongoing 
cap = cv2.VideoCapture('Test1_Trim.mp4')

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
L_red1= np.array([0,100,100])
u_red1=np.array([10,255,255])
L_red2=np.array([160,100,100])
u_red2=np.array([180,255,255])
l_green=np.array([50,100,100])
u_green=np.array([90,255,255])
l_yellow=np.array([15,150,150])
u_yellow=np.array([35,255,255])
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    
    # (imgae, scale_factor, (size), (mean subtraction), invert blue with red {True/False}, crop yes or no)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0, 0, 0), True, crop=False)
    """
    for b in blob:
    for n, img_blob in enumerate(b):
        cv2.imshow(str(n),img_blob)
    """
    
    # giving blob as input to the net
    net.setInput(blob) 
    
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    X1=[]
    
    # non-maximal supression removes double identifications for same objects
    X=[]
    Y=[]
    hsv=[]
    mask1=[]
    mask2=[]
    maskr=[]
    masky=[]
    maskg=[]
    dime=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.6 and class_id==9:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.4)
    frame_new= frame.copy()
    print("len of box :",len(boxes))
    for i in range(len(boxes)):
        
        if i < len(indexes):
            
        #if str(classes[class_ids[i]]) == 'traffic signal':
            x, y, w, h = boxes[i]
            #label = str(classes[class_ids[i]])
            label= 'Traffic Signal'
   
            confidence = confidences[i]
            #print(label)
            X.append(x)
            Y.append(y)
            #color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 50+25), font, 1.2, color, 2)
           
            #y1= frame[Y[i]:Y[i]+h,X[i]:X[i]+w]
            X1.append(frame_new[Y[i]:Y[i] + h, X[i]:X[i] + w])
            #print(np.shape(X1[i]))
            status= Status(X1[i])
            cv2.putText(frame, status + " " , (x, y -27 +25), font, 2, color, 3)
            #for j in range (H1):
                #print (j)
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 1)
    
    cv2.imshow("Image", frame)
    cv2.imshow("red", maskr[1])
    cv2.imshow("yellow", masky[1])
    cv2.imshow("green", maskg[1])
    #cv2.imshow("n img", y1)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()