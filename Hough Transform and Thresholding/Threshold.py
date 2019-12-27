import numpy as np
import cv2

def hsv_values(b,g,r):
    
    ''' Function to calcutalte the HSV values of given RGB Value Input: BGR value; Output: HSV Range '''
    
    c = np.uint8([[[b,g,r]]])
    
    hsvg = cv2.cvtColor(c,cv2.COLOR_BGR2HSV)
    
    lower = hsvg[0][0][0] - 10,100,100
    upper = hsvg[0][0][0] + 10,255,255
    print(lower)
    print(upper)
    
################################################################################################
def track(frame):
    
    #Convert BGR to HSV    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([90, 255, 255])
    
    lower_red1 = np.array([0,100,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])
    
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])
    
    # Threshold the HSV image to get only blue colors
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(mask1, mask2)
    
    '''Here it will take the HSV img value and if the value lies in the range
     it will store that particular index value as 1 (255) else 0(0). Thus we get our mask'''
    
    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(frame,frame, mask= maskg)
    #cv2.imshow('res',res)
    '''Here it will take the mask and see which index has 1(255) and it will allow
   that BGR value to display in the image rest (0 0 0)'''
    return maskg,masky,maskr
################################################################################################
def _open(maskg,masky,maskr):
    
    ''' Remove the noise from the filter mask using opening '''
    
    kernel = np.ones((2,2),np.uint8)
    opening_g = cv2.morphologyEx(maskg,cv2.MORPH_OPEN,kernel, iterations = 4)
    opening_y = cv2.morphologyEx(masky,cv2.MORPH_OPEN,kernel, iterations = 1)
    opening_r = cv2.morphologyEx(maskr,cv2.MORPH_OPEN,kernel, iterations = 1)
    
    return opening_g,opening_y,opening_r

def close(opening_g,opening_y,opening_r):
    
    kernel = np.ones((2,2),np.unit8)
    sure_bg_g = cv2.dilate(opening_g,kernel,iterations=3)
    sure_bg_y = cv2.dilate(opening_y,kernel,iterations=3)
    sure_bg_r = cv2.dilate(opening_r,kernel,iterations=3)
    
    return sure_bg_g,sure_bg_y,sure_bg_r
################################################################################################    
def detect(maskg,masky,maskr,cimg):
    
    ''' Main detection is done here using the Red, Green Mask using Hough Transform and thresholding. ''' 
    
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,
                               param1=50, param2=10, minRadius=0, maxRadius=30)

    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                 param1=50, param2=10, minRadius=0, maxRadius=30)

    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 30,
                                 param1=50, param2=5, minRadius=0, maxRadius=30)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = cimg.shape
    r = 5
    bound = 4.0 / 10
    
    
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskr[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskr, (i[0], i[1]), i[2]+20, (255, 255, 255), 2)
                cv2.putText(cimg,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))

        for i in g_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskg[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 100:
                cv2.circle(cimg, (i[0], i[1]), i[2]+20, (0, 255, 0), 2)
                cv2.circle(maskg, (i[0], i[1]), i[2]+10, (255, 255, 255), 2)
                cv2.putText(cimg,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if y_circles is not None:
        y_circles = np.uint16(np.around(y_circles))

        for i in y_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += masky[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(cimg, (i[0], i[1]), i[2]+20, (0, 255, 0), 2)
                cv2.circle(masky, (i[0], i[1]), i[2]+10, (255, 255, 255), 2)
                cv2.putText(cimg,'YELLOW',(i[0], i[1]), font, 0.7,(255,0,0),2,cv2.LINE_AA)

    return cimg
################################################################################################
def main():
        
    cimg1 = cv2.imread(r' local_drive_path \Image\red_s.jpg')
    cimg = cv2.resize(cimg1,(1080,720))
    
    
    #check HSV ramge Values for a particular colour 
    hsv_values(0,255,0)
    
    
    #get the masked images
    maskg,masky,maskr = track(cimg) 
    
    #Filter out the outliers 
    opening_g,opening_y,opening_r = _open(maskg,masky,maskr)
    
    # sure background area
    sure_bg_g,sure_bg_y,sure_bg_r = close(opening_g,opening_y,opening_r) 
    
    #detectection part 
    cimg = detect(sure_bg_g,sure_bg_y,sure_bg_r,cimg)   
    
    cv2.imshow('maskg', opening_g)
    cv2.imshow('maskr', sure_bg_g)
    cv2.imshow('masky', maskg)
    
    cv2.imshow('detected results', cimg)
    #cv2.imwrite(os.path.join('H:\Masters Study\Computer Vision\Project', 'green.jpg'),cimg)
    
       
    cv2.waitKey(0)
    cv2.destroyAllWindows()

################################################################################################
if __name__=="__main__":
    main()