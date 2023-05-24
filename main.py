import torch
import cv2
import pandas as pd
import json
from ultralytics import YOLO

target = "Safekeeper"

classes = []
img = cv2.VideoCapture(0)

model = torch.hub.load('yolov5', 'custom', path = "C:\\Users\\OLD_MECHANICA\\Desktop\\RetailShop\\Program\\yolov5-master\\runs\\weights\\yolov5_100_mark_best.pt", source='local')



def product_counter(img,coords,target):
    image_product_map_count = { "Seylon Tea" : 0,
                            "Orenge Delight Biscuit" : 0,
                            "Bombey Jhal Muri" : 0,
                            "ACI Pure Salt" : 0,
                            "Meredian Thai Stick Noodles" : 0,
                            "Deco Noodles" : 0,
                            "Pure Chicken Curry Masala" : 0,
                            "Pure Beef Curry Masala" : 0,
                            "Cashi Cinigula Rice" : 0,
                            "Spa" : 0}
    view_classes = set([])
    for index, row in coords.iterrows():
        start_point = (int(row['xmin']), int(row['ymin']))
        end_point = (int(row['xmax']), int(row['ymax']))
        
        class_name = row['name']
        view_classes.add(class_name)
        
        img = cv2.rectangle(img, start_point, end_point,(0,255,255),2)
        cv2.putText(img, str(index+1), start_point, cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)

        if class_name in image_product_map_count:
            image_product_map_count[class_name] = image_product_map_count.get(class_name) + 1

    
    print(list(view_classes))
    
    
    return img, image_product_map_count  


def bbox_generator(img):
    results = model(img)
    coords = pd.DataFrame(results.pandas().xyxy[0])
    coords.sort_values(by=['xmin'], inplace=True)
    coords.drop(coords[coords['confidence'] <= 0.7].index, inplace = True)  #drop detected object, if confidence lower than 0.7
    #print(coords)

    return coords

board_message = "OpenCamera"

while True:
    ret, frame = img.read()
    #frame = frame[10:470, 230:370]
    coords = bbox_generator(frame)
    frame, count = product_counter(frame, coords, target)

    #frame = cv2.flip(frame, 1)

    cv2.imshow('Frame', frame)
    
        
        
    #print(message)    
    print(count)
    #print(message)
    
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
            
            

img.release()
cv2.destroyAllWindows()