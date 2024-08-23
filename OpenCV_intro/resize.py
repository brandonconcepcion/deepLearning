import cv2 as cv

if False: 
    #Loading an image
    img = cv.imread('Photos/cat.jpg')

    cv.imshow('Cat', img)
    cv.waitKey(0)

    #Reading Videos 
    
    capture = cv.VideoCapture("OpenCV_intro/Videos/dog.mp4")
    
    while True: 
        isTrue, frame = capture.read()
        
        frame_resize = rescaleFrame(frame, 0.2)
        
        cv.imshow('Video', frame)
        cv.imshow('Frame Resized', frame_resize)
        
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    
    capture.release()
    cv.destroyAllWindows()
    
    
def rescaleFrame(frame, scale = 0.75): 
    #Images, Videos, and Live Videos
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)


def changeRes(width, height): 
    #Live Videos only
    capture.set(3, width)
    capture.set(4,height)

capture = cv.VideoCapture("OpenCV_intro/Videos/dog.mp4")
    
while True: 
    isTrue, frame = capture.read()
    
    frame_resize = rescaleFrame(frame, 0.2)
    
    cv.imshow('Video', frame)
    cv.imshow('Frame Resized', frame_resize)
    
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
