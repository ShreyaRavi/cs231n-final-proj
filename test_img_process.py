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
    
def findEyes(gray):
    eyes = eye_cascade.detectMultiScale(
       gray,
       scaleFactor=1.1,
       minNeighbors=3,
       minSize=(15, 15),
       maxSize=(50,50),
       flags = cv2.CASCADE_SCALE_IMAGE
    )
    return eyes
    
def getFaceCoordinates(e, image):
    # Get the origin co-ordinates and the length and width of eye
    ex, ey, ew, eh = [v for v in e]
    # get the rectangle img around the eye
    # print("eye coordinates: ", ex, " ", ey, " ", ew, " ", eh)
    cv2.rectangle(image, (ex,ey), (ex+ew,ey+eh), 5)
    sub_eye = image[ey:ey+eh, ex:ex+ew]

    x = max(ex - int(ew * WIDTH_SCALE / 2), 0)
    y = max(ey - int(eh * HEIGHT_SCALE / 3), 0)
    w = int(ew * WIDTH_SCALE)
    h = int(eh * HEIGHT_SCALE)
    
    #print("face coordinates: ", x, " ", y, " ", w, " ", h)
    return x,y,w,h
    
def getBlurredFaceImg(x, y, w, h, image):
    result_image = image.copy()
    cv2.rectangle(image, (x,y), (x+w,y+h), 5)
    sub_face = image[y:y+h, x:x+w]
    
    # apply a gaussian blur on this new recangle image
    sub_face = cv2.blur(sub_face,(23, 23), 30)
    
    # merge this blurry rectangle to our final image
    result_image[y:y+sub_face.shape[0], x:x+sub_face.shape[1]] = sub_face
    return result_image
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--inputdir",
                  default= "/Users/jbayrooti/Documents/Stanford/CS/CS231N/cs231n-proj/imgs_out",
                  type=str,
                  required=False,
                  help="The imgs directory")
    args = parser.parse_args()
    inputdir = args.inputdir

    traindir = inputdir + "/train"
    testdir = inputdir + "/test"
    #os.mkdir('imgs_blur')
    #os.mkdir('imgs_blur/train')
    os.mkdir('imgs_blur/test/')

    for i in range(0,10):
        c = "c" + str(i)
        cdir = testdir + "/" + c
        ctestdir = testdir + "/" + c
        os.mkdir('imgs_blur/test/' + c)
        for image_name in os.listdir(cdir):
            image_path = ctestdir + "/" + image_name
            # Load the image
            image = load_image(image_path)

            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

            # Preprocess the image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            # Run the classifier
            eyes = findEyes(gray)

            if len(eyes) == 0:         # If there are eyes in the images
                #print("Found no eyes")
                result_image = image
            elif (len(eyes) >= 1):
                #print("Found ", len(eyes), " eyes")
                e = eyes[0] # grab the first eye
                x,y,w,h = getFaceCoordinates(e, image)
                result_image = getBlurredFaceImg(x,y,w,h, image)

            outpath = "imgs_blur/test/" + c + "/"
            os.chdir(outpath)
            if not cv2.imwrite(image_name, result_image):
                print("exception writing")
            os.chdir("../../../")
    '''         
    for i in range(0,10):
        c = "c" + str(i)
        cdir = traindir + "/" + c
        ctestdir = testdir + "/" + c
        os.mkdir('imgs_blur/train/' + c)
        for image_name in os.listdir(cdir):
            image_path = cdir + "/" + image_name
            # Load the image
            image = load_image(image_path)
            
            # Get the classifier
            # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface_default.xml')
            
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Preprocess the image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            
            # Run the classifier
            eyes = findEyes(gray)

            if len(eyes) == 0:         # If there are eyes in the images
                #print("Found no eyes")
                result_image = image
            elif (len(eyes) >= 1):
                #print("Found ", len(eyes), " eyes")
                e = eyes[0] # grab the first eye
                x,y,w,h = getFaceCoordinates(e, image)
                result_image = getBlurredFaceImg(x,y,w,h, image)
            
            outpath = "imgs_blur/train/" + c + "/"
            os.chdir(outpath)
            if not cv2.imwrite(image_name, result_image):
                print("exception writing")
            os.chdir("../../../")
            #cv2.imwrite("./result.jpg", result_image)
'''