## Automatic Image data analyse

This project aims at analysing data in case that we can not obtain the original data but only the output image, we use color mask to choose data points, RANSAC to fit line and OCR to find coordinate, the result is like:

![](https://github.com/XiaoyuBIE1994/Auto_ImageData_Analyse/blob/master/Result_example.png)

=============Resualt=============


The origin of coordinate is: (57, 259)  
The firs marker in X axis is: (135, 259)  
The second marker in y axis is: (57, 158)  
The distance of every tick mark in X axis is 78 pixel  
The distance of every tick mark in y axis is 50.5 pixel  
The scale in x axis is: ['0', '391.0']  
The scale in y axis is: ['-0.13', '0.26']  
The x scale is 5.013/pixel  
The y scale is 0.00772/pixel  
Line 1 is: y = -0.385x + 237.874  
Line 2 is: y = -86.250x + 38152.000  
The first point is: (1927.711, 1.347)  
The second point is: (1930.488, 1.71)  


=============Finish=============

