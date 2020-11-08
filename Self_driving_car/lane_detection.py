import cv2
import numpy as np
import matplotlib.pyplot as plt

# READ IMAGE FILE
image = cv2.imread('Image/test_image.jpg')
#ALWAYS AS A COPY
lane_image = np.copy(image)
#CONVERTING TO GREY IMAGE
"""grey = cv2.cvtColor(lane_image,cv2.COLOR_RGB2GRAY)"""
#IMAGE NOISE REDUCTION
"""blur = cv2.GaussianBlur(grey,(5,5),0)"""
#EDGE DETECTION USING GRADIENT CHANGE USING DERIVATIVE OF IMAGE MATRIX
"""canny = cv2.Canny(blur,50,150)"""
#SHOW RESULTANT IMAGE
def show_img(image):
    cv2.imshow('result',image)
    cv2.waitKey(0)

#CREATING FUNCTION TO PERFORM IMAGE POST PROCESSING DONE ABOVE INDIVIDUALLY
def canny(image):
    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),0)
    canny = cv2.Canny(blur,50,150)
    return canny

#FUNCTION FOR CREATING REGION OF INTEREST LIMITING VIEW POINT
def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([[(200,height),(1100,height),(550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,triangle,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image
#FUNCTION TO DISPLAY MADE UP LINES IN REAL IMAGE
def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

#FROM THE AVERAGE SLOPE, INTERSECTION GETTING CORDINATES X AND Y BACK
def make_cordinates(image,line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*3/5)
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])




#AVERAGING THE LINES CREATED FOR BETTER PERCEPTION using slope and intersection
def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []

    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameter = np.polyfit((x1,x2),(y1,y2), 1)
        slope = parameter[0]
        intercept = parameter[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
        
    if len(left_fit) and len(right_fit):
        left_fit_average = np.average(left_fit,axis=0)
        right_fit_average = np.average(right_fit,axis=0)
        left_line = make_cordinates(image,left_fit_average)
        right_line = make_cordinates(image,right_fit_average)
        return ([left_line,right_line])



canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
average_line = average_slope_intercept(lane_image,lines)
line_image = display_lines(lane_image,average_line)
combo_image = cv2.addWeighted(lane_image,0.8,line_image,1,1)
show_img(combo_image)


"""cap = cv2.VideoCapture('Image/test2.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=10)
        average_line = average_slope_intercept(frame,lines)
        line_image = display_lines(frame,average_line)
        combo_image = cv2.addWeighted(frame,0.8,line_image,1,1)
        cv2.imshow('results',combo_image)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()"""
