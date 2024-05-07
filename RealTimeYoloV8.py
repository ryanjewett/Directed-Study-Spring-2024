
import cv2
from ultralytics import YOLO


model_Nano = YOLO("yolov8n.pt")
print("Done")


def process_image(model_input): 
    model = model_input
    
    cap = cv2.VideoCapture(0)
    j ,image = cap.read()
    image_path = image
    
    
    results = model.predict(image_path)
    result = results[0]                    #takes the strongest results    
    len(result.boxes)
    box = result.boxes[0]
    
    for box in result.boxes: #prints out the information for each object detected 
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        
        
        
        x_min, y_min, x_max, y_max = cords
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        label = f"{class_id}: {conf}"
        cv2.putText(image, label, (int(x_min), int(y_min) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
  
    return image


def main():
   
    frame_number = 0
    model = model_Nano

    while True:

        frame_number += 1
        
        if frame_number % 1 == 0:
            processed_frame = process_image(model)
            cv2.imshow('Processed Frame', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break

    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

