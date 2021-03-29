import numpy as np
import os
import math

pi=math.pi

def greyscale1(img):
    grey = np.uint8(img*255)
    return (grey)

def greyscale2(img):
    A = (img<=0.5)*img
    A = np.uint8(A*255*2)
    
    B = (img>0.5)*img
    B = np.uint8(((img>0.5)-B)*255*2)
    
    grey = A + B
    
    return (grey)

def RGB(img):
    n=len(img)
    
    U = np.cos(2*pi*img)
    V = np.sin(2*pi*img)
    
    R=np.uint8(U*127.5+127.5)
    G=np.uint8(V*127.5+127.5)
    B=np.uint8(np.zeros(n))
    
    RGB = np.uint8(np.zeros((n,n,3)))
    RGB[:,:,0] = R 
    RGB[:,:,1] = G 
    RGB[:,:,2] = B 
    
    return (RGB)

    