import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


from mpl_toolkits import mplot3d 
'''Once this submodule is imported, a three-dimensional axes can be created by passing the 
    keyword projection='3d' to any of the normal axes creation routines'''
        
def _3Dplot(img1,r,col):

    '''Funtion to polt the 3D Histogram of a given particular image. Helps us to study the intensity of pixels
    through out the image. Input: Image, row and column values; Output: 3D Plot'''
    
    ax = plt.axes(projection='3d')      
    x = np.linspace(0, col, col)        #define the axis X
    y = np.linspace(0, r, r)            #define the axis Y                           
    X, Y = np.meshgrid(x, y)
    
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, img1, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
    ax.set_title('surface')
        ############# used of rotation of the plot ##################### 
    for angle in range(0, 360):
        ax.view_init(90, angle)
        plt.draw()
        plt.pause(.001) 

########################################################################################################    
def threshold(img,r,col):
    
    ''' Threshold the image pixels at a particular given intensity Input: Image, row, column; output: threshold image'''
    
    for i in range(r):
        for j in range(col):
            if (img[i][j] >= 70).all():
                img[i][j] = 255
    img_g = cv.GaussianBlur(img,(5,5),4)
    return img_g

##########################################################################################################

def row_col(img):
    ''' Calculates the number of rows and columns in a given image. Input: Image; Output: Row and Column'''
    r = img.shape[0]                   # stores height of the image
    col = img.shape[1]                 # stores widhth of the image    
    return r,col

########################################################################################################
    
def Shift(img,minHessian):
    
    ''' Perform Shift on image and return keypoints, Descriptor and Image with keypoints. Input: Image, Hessian value;
    Output: keypoints, Descriptor and Image with keypoints'''
    
    surf = cv.xfeatures2d.SURF_create(minHessian)
    surf.setExtended(True)
    
    keypoints, des = surf.detectAndCompute(img,None)
    print("number of key points:",np.size(keypoints))
    
    img_keypoints = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    img_key = cv.drawKeypoints(img, keypoints, img_keypoints)
   
    #img_key = cv.drawKeypoints(img,keypoints,None,(255,0,0),4)
    return keypoints,des,img_key


#########################################################################################################
def match(img1,img2,des1,des2,kp1,kp2):
    ''' Matchng the best keypoints in the two images using Knn. '''
    
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,outImg = None,flags=2)
    return img3

##########################################################################################
def main():
    
    img1 = cv.imread(r' local_drive_path \Images\red_s.jpg')  # Open the Images  
    img2 = cv.imread(r'local_drive_path \Images\g11.jpg') 
    
    r1,col1 = row_col(img1) 
    r2,col2 = row_col(img2)
   
    #_3Dplot(img1,r1,col1)
    
    #img1 = threshold(img1,r1,col1)
    #img2 = threshold(img2,r2,col2)
        
    kp1,des1,img_s1 = Shift(img1,minHessian = 2000)
    kp2,des2,img_s2 = Shift(img2,minHessian = 2000)
    
        
    imgf1 = match(img1,img2,des1,des2,kp1,kp2)
    imgm = cv.resize(imgf1,(1080,720))
    
        
    cv.imshow('match', imgm)
    cv.waitKey(0)
    cv.destroyWindow('match')
###########################################################################################    
if __name__=="__main__":
    main()
