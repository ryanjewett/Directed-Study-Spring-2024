
import cv2
import os
from ultralytics import YOLO

model1 = YOLO("yolov8m.pt")

def process_image(image_path, blank_image):

    blank_image_copy = blank_image.copy()  # Make a copy of the blank image

    results = model1.predict(image_path)
    result = results[0]
    
    for box in result.boxes:
        class_id = result.names[box.cls[0].item()]
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        x_min, y_min, x_max, y_max = cords
        if conf >= .5 and (x_max - x_min >= 10 and y_max - y_min > 10):
            # x_min, y_min, x_max, y_max = cords
            cv2.rectangle(blank_image_copy, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            label = f"{class_id}: {conf}"
            cv2.putText(blank_image_copy, label, (int(x_min), int(y_min) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    return blank_image_copy


def main():


    # input_folder = input("Enter the path to the folder containing images: ")       #get all folder location  
    # output_folder = input("Enter the desired save location for the video file: ")
    input_folder = r"C:\Users\ryanj\Documents\2024ResearchV2\Ford_AV_DataSet"      #get all folder location  
    output_folder = r"C:\Users\ryanj\Documents\2024ResearchV2\SAVE_HERE"
    named_video_file = input("Enter the name of the completed video file: ")


    if not os.path.exists(output_folder):                                          #make sure the folder acutally exists
        print("Not correct file for saving video")
        exit(0)
    if not os.path.exists(input_folder):
        print("No image folder found!")
        exit(0)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]      #save all images to image_files
    blank_image = r"C:\Users\ryanj\Documents\2024ResearchV2\blankimage.jpg"
    blank_image = cv2.imread(blank_image)
    first_image = cv2.imread(os.path.join(input_folder, image_files[0]))           #get the size of the images inputed. ALL MUST BE SAME SIZE
    height, width, _ = first_image.shape
    blank_image = cv2.resize(blank_image,(width,height))
    width = width*2                                                                #width needs to be doubled if using orignal and new image
    
    named_video_file = named_video_file + ".avi"                                   #adds .mp4 to end of inputed file name 
    output_file = os.path.join(output_folder, named_video_file)                    #creates the output video file location
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')                                       #avc1 is the only codec that I found that worked

    out = cv2.VideoWriter(output_file, fourcc, 5.0, (width, height))               #changed third input to VideoWriter to adjust frame rate

    
    for image_file in image_files:                         
        img_path = os.path.join(input_folder, image_file)                          #gets the image path for the current image in the folder
        original_img = cv2.imread(img_path)                                        #uses cv2.imread to store the image in original_img

        
        processed_img = process_image(original_img,blank_image)                                #get image with bounding boxes

        new_org_image = cv2.imread(img_path)                                       #had to create new image variable to save orgianl image

        result_img = cv2.hconcat([new_org_image, processed_img])                   #stacks both images side by side

        
        out.write(result_img)                                                      #save the image to the .avi file to make a video

    
    out.release()
    print(f"Video saved at: {output_folder}")

if __name__ == "__main__":
    main()
