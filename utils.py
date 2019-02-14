import numpy as np
import file_walker
import re
import cv2
import matplotlib.pyplot as plt
from config import *
from scipy.integrate.quadrature import simps
import math


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def unnormalizeToCV(input = []):
    #input is unnormalized [batch_size, channel, height, width] tensor from pytorch
    #inputGT is [batch_size, 136] tensor landmarks 
    #Output is [batch_size, height,width,channel] BGR, 0-255 Intensities opencv list of landmarked image
    output = []
    
    #unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    err = []
    
    for i in range(input.shape[0]) : 
        #Unnormalized it, convert to numpy and multiple by 255. 
        theImage = unorm(input[i]).numpy()*255
        
        #Then transpose to be height,width,channel, to Int and BGR formate 
        theImage = cv2.cvtColor(theImage.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
        
        output.append(theImage)
     
    return output
    
def unnormalizedAndLandmark(input = [], inputPred = [],inputGT = None,cv = True,gt_ia = False):
    #input is unnormalized [batch_size, channel, height, width] tensor from pytorch
    #inputGT is [batch_size, 136] tensor landmarks 
    #Output is [batch_size, height,width,channel] BGR, 0-255 Intensities opencv list of landmarked image
    output = []
    
    if not type(inputPred) is np.ndarray : 
        inputPred = inputPred.numpy()
        
    if inputGT is not None :
        if cv : 
            inputGT = inputGT.numpy()
    
    #unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    err = []
    
    for i in range(inputPred.shape[0]) : 
        #Unnormalized it, convert to numpy and multiple by 255. 
        theImage = unorm(input[i]).numpy()*255
        
        #Then transpose to be height,width,channel, to Int and BGR formate 
        theImage = cv2.cvtColor(theImage.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
        
        #Now landmark it. 
        for y in range(68) : 
            cv2.circle(theImage,(int(inputPred[i,y]),int(inputPred[i,y+68])),4,(0,255,0),-1 )
            if inputGT is not None : 
                if cv or gt_ia:
                    cv2.circle(theImage,(int(inputGT[i,y]),int(inputGT[i,y+68])),4,(0,0,255),-1 )
                    err.append(calcLandmarkError(inputPred[i],inputGT[i]))
                else :  #cv used only on one gt
                    cv2.circle(theImage,(int(inputGT[y]),int(inputGT[y+68])),4,(0,0,255),-1 )
        
        if inputGT is not None :
            if cv or gt_ia :  
                err.append(calcLandmarkError(inputPred[i],inputGT[i]))
            else : 
                err.append(calcLandmarkError(inputPred[i],inputGT))
        
        output.append(theImage)
    
    if inputGT is None : 
        return output
    else :  
        return output, err
    


def calcLandmarkError(pred,gt): #for 300VW
    '''
    input : 
        pred : 1,num points
        gt : 1, num points 
        
        according to IJCV
        Normalized by bounding boxes
    '''
    
    #print pred,gt
    
    
    num_points = pred.shape[0]
    
    num_points_norm = num_points//2
    
    bb = get_bb(gt[:68],gt[68:])
    
    #print(gt)
    #print(bb)
    
    '''width = np.abs(bb[2] - bb[0])
    height = np.abs(bb[3] - bb[1])
    
    gt_bb = np.sqrt(np.square(width) + np.square(height))
    
    
    print("1 : ",width,height,gt_bb)'''
    
    width = np.abs(bb[2] - bb[0])
    height = np.abs(bb[3] - bb[1])
    
    gt_bb = math.sqrt((width*width) +(height*height))
    
    #print("2 : ",width,height,(width^2) +(height^2),gt_bb)
    '''print(bb) 
    print(gt_bb)
    print("BB : ",gt)
    print("pred : ",pred)'''
    
    '''print(num_points_norm)
    print("BB : ",bb)
    print("GT : ",gt)
    print("PR : ",pred)'''
    #print(num_points)
    
    '''error = np.mean(np.sqrt(np.square(pred-gt)))/gt_bb
    return error''' 
    
    summ = 0
    for j in range(num_points_norm) : 
        #summ += np.sqrt(np.square(pred[j]-gt[j])+np.square(pred[j+num_points_norm]-gt[j+num_points_norm]))
        summ += math.sqrt(((pred[j]-gt[j])*(pred[j]-gt[j])) + ((pred[j+num_points_norm]-gt[j+num_points_norm])*(pred[j+num_points_norm]-gt[j+num_points_norm])))
    #err = summ/(num_points_norm * (gt_bb))
    err = summ/(num_points_norm*gt_bb)
    
        
    return err
