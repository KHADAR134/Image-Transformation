# Image-Transformation
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import the required libraries and image for transformation.
<br>

### Step2:
Perform operations on the image like translaton, rotation and other.
<br>

### Step3:
Use the warpPerspective(image , matrix, (rows, columns)) function.
<br>

### Step4:
Plot the Image and Transformed Image on the graph using matplotlib for identifying changes.
<br>

### Step5:
Diifferent operations has been performed on the image.
<br>

## Program:
```python
Developed By: SHAIK KHADAR BASHA
Register Number: 212220230045


i)Image Translation

import numpy as np
import matplotlib.pyplot as plt 
import cv2 as cv 

#plotting of an image 
image = cv.imread("tata.jpg")
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(image)
plt.show()

#translation of an image 
rows,cols,dim = image.shape
M = np.float32([[1,0,100], [0,1,50],[0,0,1]])

translated_image= cv.warpPerspective(image, M, (cols, rows))

plt.axis("off")
plt.imshow(translated_image)
plt.show()

ii) Image Scaling

#SCALING 
rows,cols,dim = image.shape

M_scale = np.float32([[2,0,0], [0,1.6,0],[0,0,1]])

scale_image= cv.warpPerspective(image, M_scale, (cols, rows))


plt.axis("off")
plt.imshow(scale_image)
plt.show()

iii)Image shearing

#shearing image 
M_x = np.float32([[1,1,0], [0,1,0],[0,0,1]])

M_y = np.float32([[1,0,0], [0.4,1,0],[0,0,1]])


shear_imagex= cv.warpPerspective(image, M_x, (cols, rows))
shear_imagey= cv.warpPerspective(image, M_y, (cols, rows))


plt.axis("off")
plt.imshow(shear_imagex)
#plt.imshow(shear_imagey)
plt.show()


plt.axis("off")
#plt.imshow(shear_imagex)
plt.imshow(shear_imagey)
plt.show()


iv)Image Reflection

#reflect an image 
M_x = np.float32([[1,0,0],[0,-1,rows],[0,0,1]])

M_y = np.float32([[-1,0,cols], [0,1,0],[0,0,1]])

ref_imagex= cv.warpPerspective(image, M_x, (cols, rows))
ref_imagey= cv.warpPerspective(image, M_y, (cols, rows))


plt.axis("off")
plt.imshow(ref_imagex)
plt.show()


plt.axis("off")
plt.imshow(ref_imagey)
plt.show()


v)Image Rotation

angle=np.radians(10)
matrix=np.float32([[np.cos(angle),-np.sin(angle),0],
                                [np.sin(angle),np.cos(angle),0],
                                [0,0,1]])
Rotated_image=cv.warpPerspective(image,matrix,(cols,rows))
plt.axis("off")
plt.imshow(Rotated_image)


vi)Image Cropping

# cropping 
    
crop_img = image[600:750, 400:500]


plt.axis("off")
plt.imshow(crop_img)
plt.show()

```
## Output:
### i)Image Translation
<br>
<br>

![DIP(51)](https://user-images.githubusercontent.com/75235233/165789017-0e8a4751-d309-4a5e-a1a6-89ef2d894dfb.png)

<br>
<br>

### ii) Image Scaling
<br>
<br>

![DIP(52)](https://user-images.githubusercontent.com/75235233/165789041-df1d8e18-3f83-40e4-a5e1-d0ad19e19465.png)

<br>
<br>


### iii)Image shearing
<br>
<br>

![DIP(53)](https://user-images.githubusercontent.com/75235233/165789097-8b3363c7-b9f0-4bae-830c-58f772808326.png)

<br>
<br>


### iv)Image Reflection
<br>
<br>

![DIP(54)](https://user-images.githubusercontent.com/75235233/165789164-f961b086-5688-4d33-987f-b9ff7bcb40f0.png)


<br>
<br>



### v)Image Rotation
<br>
<br>

![DIP(55)](https://user-images.githubusercontent.com/75235233/165789218-45393215-ac37-4620-a6e6-da709f45b4ce.png)


<br>
<br>



### vi)Image Cropping
<br>
<br>

![DIP(56)](https://user-images.githubusercontent.com/75235233/165789267-402c2718-9291-4c26-8bab-34d123b4c3ee.png)


<br>
<br>




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
