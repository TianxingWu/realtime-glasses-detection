# Real-time Glasses Detection

## Introduction
This is an light-weight glasses detector written in Python for real-time videos. The algorithm is based on the method presented by the paper in [*Reference*](#Reference) with some modifications. Note that the goal is to determine the presence but not the precise location of glasses.

## Requirements
* python 3.6 
- numpy 1.14.0
* opencv-python 3.4.0
- dlib 19.7.0

## What's Next
The threshold used for determining the presence of glasses is manually chosen based on experiment results at present. My next goal is to develop an algorithm that can choose the threshold automatically in order to enchance robustness.

Welcome to fork and try it on your own! :blush:

## Reference
Jiang, X., Binkert, M., Achermann, B. et al. Pattern Analysis & Applications (2000) 3: 9. https://doi.org/10.1007/s100440050002
