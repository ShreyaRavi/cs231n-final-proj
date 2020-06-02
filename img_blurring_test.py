import cv2
import argparse

WIDTH_SCALE = 4.5
HEIGHT_SCALE = 4.5
import os


def load_image(image_path):
    # Load the image
    #image_path = "/Users/jbayrooti/Documents/Stanford/CS/CS231N/cs231n-proj/imgs/train/c4/img_173.jpg"
    image = cv2.imread(image_path)
    return image
    
if __name__ == "__main__":
    
    image_path = "/Users/jbayrooti/Documents/Stanford/CS/CS231N/cs231n-proj/imgs_out/train/c2/img_271.jpg"
    image = load_image(image_path)
    result_image = image.copy()
    
    # Get the classifier
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface_default.xml')

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    objects = cascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=3,
      minSize=(15, 15),
      maxSize=(80,80),
      flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(objects) != 0:
        for o in objects:
            x, y, w, h = [v for v in o]
            
            cv2.rectangle(image, (x,y), (x+w,y+h), 5)
            sub_face = image[y:y+h, x:x+w]
            
            # apply a gaussian blur on this new recangle image
            sub_face = cv2.blur(sub_face,(23, 23), 30)
            
            # merge this blurry rectangle to our final image
            result_image[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
    cv2.imwrite("./result.jpg", result_image)

