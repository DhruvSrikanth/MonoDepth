import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image
import math
from scipy.optimize import curve_fit 

img_path = 'F:/Work/Work/Outdu Internship/nyu_depth_v2_dataset/test_results/test_depth_4.png'
img = cv2.imread(img_path, 0)
#img = cv2.bitwise_not(img)
#cv2.imshow('inv',img)

def disparity_to_depth1(img, focal_length, baseline, maxDepth):
    height,width = img.shape
    baseline_x_focal = focal_length * baseline
    disparity_scaled = img * width
    for y in range(0,height):
        for x in range(0,width):
            if disparity_scaled[y][x] == 0:
                disparity_scaled[y][x] = baseline_x_focal / maxDepth
    depth = np.round(baseline_x_focal / disparity_scaled, decimals = 3)
    return depth

def disparity_to_depth2(depth_file):
    depth = Image.open(depth_file).convert('I')
    scaling_factor = 1000
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")

    depth_vals = []    
    for v in range(depth.size[1]):
        for u in range(depth.size[0]):
            Z = depth.getpixel((u,v)) / scaling_factor
            depth_vals.append(Z)

    return depth_vals

def disparity_to_depth3(img, focal_length, baseline, maxDepth):
    height,width = img.shape
    disparity_scaled = img * 1#width
    for y in range(0,height):
        for x in range(0,width):
            if disparity_scaled[y][x] == 0:
                disparity_scaled[y][x] = baseline_x_focal / maxDepth
            distance = 123.6 * math.tan ( disparity_scaled[y][x]/2842.5 + 1.1863 )
            #distance = 1000/ (-0.00307 * rawDisparity + 3.33 )
            #distance= ((1000/disparity_scaled[y][x]) - 3.33)/-0.00307
            disparity_scaled[y][x] = distance
    depth = np.round(disparity_scaled, decimals = 3)
    return depth

def disparity_to_depth4(depth_file):
    depth = Image.open(depth_file).convert('I')
    depth_vals = []    
    for v in range(depth.size[1]):
        for u in range(depth.size[0]):
            Z = depth.getpixel((u,v))/1
            depth_vals.append(Z)
    return depth_vals

def disparity_to_depth5(depth_file,maxDepth):
    depth = Image.open(depth_file).convert('I')
    depth_vals = []    
    for v in range(depth.size[1]):
        for u in range(depth.size[0]):
            Z = depth.getpixel((u,v))
            d = maxDepth*(255.0-Z)/255.0
            depth_vals.append(round(d/1000,3))
    return depth_vals

def disparity_to_depth6(depth_file, focal_length, baseline, maxDepth):
    depth = Image.open(depth_file).convert('I')
    print(np.amax(depth),np.amin(depth))
    depth_vals = []
    baseline_x_focal = focal_length * baseline
    for v in range(depth.size[1]):
        for u in range(depth.size[0]):
            Z = depth.getpixel((u,v))
            if Z == 255:
                depth_val = 0
            elif Z == 0:
                depth_val = maxDepth
            else:
                depth_val = baseline_x_focal/(Z*1.0)#depth.size[0])
            depth_vals.append(round(depth_val,3))
    return depth_vals

def disparity_to_depth7(depth_file, function_coeff):
    depth = Image.open(depth_file).convert('I')
    disparity = []
    depth_vals = []
    for v in range(depth.size[1]):
        for u in range(depth.size[0]):
            Z = depth.getpixel((u,v))
            disparity.append(Z)
            depth_val = np.polyval(function_coeff, Z)
            depth_vals.append(round(depth_val,3))
    return disparity,depth_vals

def find_baseline(depth_value,disparity,focal_length):
    baseline = round((depth_value*disparity)/focal_length,3)
    return baseline

focal_length_pixels = 580.0
baseline_cm = 9.58
maxDepth_cm = 500.0

#disparity
x0 = 25.0
x1 = 34.5
x2 = 58.5
x3 = 80.0
x4 = 82.5
x5 = 106.5
xn = 120.0
x = [x0,x1,x2,x3,x4,x5,xn]

#depth cm (without scaling by depth img width)
y0 = 0.0
y1 = 88.0
y2 = 95.0
y3 = 134.0
y4 = 190.0
y5 = 275.0
yn = 500.0
y = [y0,y1,y2,y3,y4,y5,yn]

def find_func(x,yreal):
    import numpy as np
    from numpy.polynomial.chebyshev import chebfit,chebval
    
    f = np.polyfit(x,yreal,len(yreal))
    c = chebfit(x, yreal, len(yreal))
    
    ytest_poly = [round(np.polyval(f, xi),3) for xi in x]
    ytest_cheb = [round(chebval(xi, c),3) for xi in x]

    ytest = ytest_poly
    function_coeff = f
    
    return ytest,function_coeff

def plot_poly(x,yreal,ytest):
    import numpy as np
    import matplotlib.pyplot as plt

    plt.plot(x, ytest,color = 'blue', marker = "o")
    plt.plot(x, yreal,color = 'red', marker = "o")
    plt.grid()
    plt.xlabel("Disparity") 
    plt.ylabel("Depth")
    plt.title("Depth v Disparity Polynomial Comparison")
    plt.show()

ytest,function_coeff = find_func(x,y)
plot_poly(x,y,ytest)


#depth_cm = disparity_to_depth1(img, focal_length_pixels, baseline_cm, maxDepth)
#depth_cm = disparity_to_depth2(img_path)
#depth_cm = disparity_to_depth3(img, focal_length_pixels, baseline_cm, maxDepth)
#depth_cm = disparity_to_depth4(img_path)
#depth_cm = disparity_to_depth5(img_path,maxDepth)
#depth_cm = disparity_to_depth6(img_path, focal_length_pixels, baseline_cm, maxDepth_cm)
disparity,depth_cm = disparity_to_depth7(img_path, function_coeff)
plt.plot(disparity, depth_cm,color = 'blue', marker = "o")
plt.grid()
plt.xlabel("Disparity") 
plt.ylabel("Depth")
plt.title("Depth v Disparity Polynomial Comparison")
plt.show()


print(np.amax(depth_cm),np.amin(depth_cm))

#plt.hist(img.ravel(),256,[0,256])
#plt.show()
