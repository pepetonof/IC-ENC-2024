# -*- coding: utf-8 -*-
"""
Created on Fri Apr 2 00:44:09 2022

@author: PC
"""


from matplotlib import pyplot as plt
import numpy as np
import torch 
import kornia as K
from kornia import morphology as morph

splited = 100

#%% Funciones Morfológicas
def Erosion(img_rgb):
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)
    contador=0
    for img in img_rgb:
        # img = img.float()/255.
        torch.cuda.empty_cache()
        img = morph.erosion(img, kernel)
        # img = img.float()*255.
        # img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        contador+=1
        del img
        torch.cuda.empty_cache()
    
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    del kernel
    torch.cuda.empty_cache()
    img_rgb = img_rgb.to("cpu")
    torch.cuda.empty_cache()

    return img_rgb

def Dilation(img_rgb):
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)
    contador=0
    for img in img_rgb:
        # img = img.float()/255
        torch.cuda.empty_cache()
        img = morph.dilation(img, kernel)
        # img = img.float()*255
        # img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    
    del kernel
    img_rgb = img_rgb.to("cpu")
    torch.cuda.empty_cache()
    return img_rgb

def Closing(img_rgb):
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)
    contador=0
    for img in img_rgb:
        # img = img.float()/255
        torch.cuda.empty_cache()
        img = morph.closing(img, kernel)
        # img = img.float()*255
        # img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    del kernel
    img_rgb = img_rgb.to("cpu")
    torch.cuda.empty_cache()
    return img_rgb

def Opening(img_rgb):
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)
    contador=0
    for img in img_rgb:
        # img = img.float()/255
        torch.cuda.empty_cache()
        img = morph.opening(img, kernel)
        # img = img.float()*255
        # img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)

    del kernel
    img_rgb = img_rgb.to("cpu")
    torch.cuda.empty_cache()
    return img_rgb

#%%Detección de bordes
def Sobel(img_rgb):
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    contador=0
    for img in img_rgb:
        # img = img.float()/255
        torch.cuda.empty_cache()
        img = K.filters.sobel(img)
        # img = img.float()*255
        img = torch.clamp(img,min=0.0,max=1.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    img_rgb = img_rgb.to("cpu")
    torch.cuda.empty_cache()

    return img_rgb

def LaPlacian(img_rgb):
    """LaPlacian"""
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    contador=0
    for img in img_rgb:
        # img = img.float()/255
        torch.cuda.empty_cache()
        img = K.filters.laplacian(img, kernel_size=5)
        # img = img.float()*255
        img = torch.clamp(img,min=0.0,max=1.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
        
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    img_rgb = img_rgb.to("cpu")
    torch.cuda.empty_cache()
    return img_rgb

def Gradient(img_rgb):
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)
    contador=0
    for img in img_rgb:
        # img = img.float()/255
        torch.cuda.empty_cache()
        img = morph.gradient(img, kernel)
        # img = img.float()*255
        img = torch.clamp(img,min=0.0,max=1.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)

    del kernel
    img_rgb = img_rgb.to("cpu")
    torch.cuda.empty_cache()
    return img_rgb

#%%Funciones aritméticas
def suma_imgs2(img1, img2):
    device = "cuda:0"
    img1 = img1.to(device)
    # img1 = img1.float()/255
    
    img2 = img2.to(device)
    # img2 = img2.float()/255
    
    suma = img1+img2
    # suma = suma.float()*255
    suma = torch.clamp(suma, min=0.0,max=1.0)
    
    img1 = img1.to("cpu")
    img2 = img2.to("cpu")
    suma = suma.to("cpu")
    return suma

def suma_imgs3(img1, img2, img3):
    device = "cuda:0"
    img1 = img1.to(device)
    # img1 = img1.float()/255
    
    img2 = img2.to(device)
    # img2 = img2.float()/255
    
    img3 = img3.to(device)
    # img3 = img3.float()/255
    
    suma = img1+img2+img3
    # suma = suma.float()*255
    suma = torch.clamp(suma, min=0.0,max=1.0)
    
    img1 = img1.to("cpu")
    img2 = img2.to("cpu")
    img3 = img3.to("cpu")
    suma = suma.to("cpu")
    return suma

def resta_imgs(img1, img2):
    device = "cuda:0"
    img1 = img1.to(device)
    # img1 = img1.float()/255
    
    img2 = img2.to(device)
    # img2 = img2.float()/255
    
    resta = img1-img2
    # resta = resta.float()*255
    # resta = torch.clamp(resta, min=0.0,max=255.0)
    
    img1 = img1.to("cpu")
    img2 = img2.to("cpu")
    resta = resta.to("cpu")
    
    return resta

def sqrt(img):
    device = "cuda:0"
    img = img.to(device)
    # img = img.float()/255
    sq = torch.sqrt(img)
    # sq = sq.float()*255
    # sq = torch.clamp(sq, min=0.0, max=255.0)
    
    img = img.to("cpu")
    sq = sq.to("cpu")
    return sq

#%%Funciones de filtrado y transformaciones de intensidad
def Gaussian_blur_2d(img_rgb):
    """Gaussian blur 2D = GB_2D
    Size kernel should be odd int positive (3,5,7,9,11,13,15,17) are the possible numbers to operate the function
    The seccond parameter is the standard deviation of the kernel. Must be a float number
    """
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    
    kernel_size=7

    contador=0
    for img in img_rgb:
        # img = img.float()/255
        torch.cuda.empty_cache()
        img = K.filters.gaussian_blur2d(img, (kernel_size,kernel_size), (10.0, 10.0))
        # img = img.float()*255
        # img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
        
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    img_rgb = img_rgb.to("cpu")
    torch.cuda.empty_cache()
    # print(img_rgb.max(), img_rgb.min())
    return img_rgb

def En_adbright(img_rgb):
    """Enhance adjust brightness
    factor = must be float between [-0.5,0.5]
    """
    device = "cuda:0"
    img_rgb = img_rgb.to(device)
    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    contador=0
    factor=0.2
    for img in img_rgb:
        # img = img.float()/255
        torch.cuda.empty_cache()
        img =  K.enhance.adjust_brightness(img, factor)
        # img = img.float()*255
        # img = torch.clamp(img,min=0.0,max=255.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
        
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0) 
    img_rgb = img_rgb.to("cpu")
    torch.cuda.empty_cache()
    # print(img_rgb.max(), img_rgb.min())
    return img_rgb


def En_equal(img_rgb):
    # print('before',img_rgb.max(), img_rgb.min())
    """Enhance adjust brightness """
    device = "cuda:0"
    img_rgb = img_rgb.to(device)

    img_rgb :torch.Tensor = torch.split(img_rgb,splited)
    img_rgb = list(img_rgb)
    contador=0
    for img in img_rgb:
        #print(img.shape)
        # img = (img.float()/255)
        img =  K.enhance.equalize(img)
        # img = img.float()*255
        img = torch.clamp(img,min=0.0,max=1.0)
        img_rgb[contador]=img
        del img
        torch.cuda.empty_cache()
        contador+=1
        
    img_rgb: torch.Tensor = torch.cat(img_rgb,dim=0)
    img_rgb = img_rgb.to("cpu")
    torch.cuda.empty_cache()
    return img_rgb    
