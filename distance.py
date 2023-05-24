import torch
import cv2
import pandas as pd
import json
from ultralytics import YOLO

target = "Safekeeper"

classes = []
img = "C:\\Users\\OLD_MECHANICA\\Downloads\\2.1.jpg"
img = cv2.imread(img)
model = torch.hub.load('yolov5', 'custom', path = "C:\\Users\\OLD_MECHANICA\\Desktop\\RetailShop\\Program\\yolov5-master\\runs\\weights\\only_marks.pt", source='local')

def bbox_generator(img):
    results = model(img)
    coords = pd.DataFrame(results.pandas().xyxy[0])
    coords.sort_values(by=['xmin'], inplace=True)
    coords.drop(coords[coords['confidence'] <= 0.7].index, inplace = True)  #drop detected object, if confidence lower than 0.7
    #print(coords)
    return coords

def bbox_draw(img, coords):
    for index, row in coords.iterrows():
        start_point = (int(row['xmin']), int(row['ymin']))
        end_point = (int(row['xmax']), int(row['ymax']))

        img = cv2.rectangle(img, start_point, end_point,(0,255,255),2)
        cv2.putText(img, str(index+1), start_point, cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)

def find_distance(coords,img):
    start_point_x = []
    end_point_x = []
    start_point_y = []
    end_point_y = []
    for index, row in coords.iterrows():
        start_point_x.append(int(row['xmin']))
        end_point_x.append(int(row['xmax']))

        start_point_y.append(int(row['ymin']))
        end_point_y.append(int(row['ymax']))

    x1 = (end_point_x[0] + start_point_x[0])/2
    x2 = (end_point_x[1] + start_point_x[1])/2
    x3 = (end_point_x[2] + start_point_x[2])/2
    x4 = (end_point_x[3] + start_point_x[3])/2
    x5 = (end_point_x[4] + start_point_x[4])/2       

    y1 = (end_point_y[0] + start_point_y[0])/2
    y2 = (end_point_y[1] + start_point_y[1])/2
    y3 = (end_point_y[2] + start_point_y[2])/2
    y4 = (end_point_y[3] + start_point_y[3])/2
    y5 = (end_point_y[4] + start_point_y[4])/2 

    

    img = cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 3)
    img = cv2.line(img, (int(x2),int(y2)), (int(x3),int(y3)), (255,0,0), 3)
    img = cv2.line(img, (int(x3),int(y3)), (int(x4),int(y4)), (255,0,0), 3)
    img = cv2.line(img, (int(x4),int(y4)), (int(x5),int(y5)), (255,0,0), 3)

    img = cv2.circle(img,(int(x1),int(y1)), 3, (0,0,255), -1)
    img = cv2.circle(img,(int(x2),int(y2)), 3, (0,0,255), -1)
    img = cv2.circle(img,(int(x3),int(y3)), 3, (0,0,255), -1)
    img = cv2.circle(img,(int(x4),int(y4)), 3, (0,0,255), -1)
    img = cv2.circle(img,(int(x5),int(y5)), 3, (0,0,255), -1)

    

    distK = 12/(x2-x1)
    print(distK)
    
    dist1 = distK*x3
    
    dist2 = distK*x4
    dist3 = distK*x5
    print("Actual Distance : 18cm, Predicted Distance:", dist1, "cm")
    print("Actual Distance : 25cm, Predicted Distance:", dist2, "cm")
    print("Actual Distance : 30cm, Predicted Distance:", dist3, "cm")
    
    cv2.putText(img, str(12), (int((x1+x2)/2), int(y1)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    cv2.putText(img, str("%.2f"% dist1), (int((x2+x3)/2), int(y1)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    cv2.putText(img, str("%.2f"% dist2), (int((x3+x4)/2), int(y1)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
    cv2.putText(img, str("%.2f"% dist3), (int((x4+x5)/2), int(y1)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)    

coordinates = bbox_generator(img)
bbox_draw(img, coordinates)
#print(coordinates)
find_distance(coordinates,img)
cv2.imshow("Image",img)

cv2.waitKey(0)
