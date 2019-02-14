'''
Created on Jun 11, 2018

@author: deckyal
'''
from operator import truediv
import utils
import tensorflow as tf
import inception_resnet_v1
from  face_localiser import face_localiser
from pathlib import Path
from config import *
from lib_yolo.model import FaceDetectionRegressor
import math
import os

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def localise(catTesting):
    
    write_data = False
        
    baseSize = 128 #this is for the classifier
    cropSize = baseSize + int(baseSize*.5)
    
    patchOSize = int(baseSize*.1)
    patchSize = int(patchOSize/2)
    
    
    doResize = False;
    
    ext=".dfaltxt"
    is3Dx = [False]
    iBB = 1
    utilized_BB = ['MD','MT']
    trailing_BB = ['TR','DT']

    arrName = ['300VW-Test/cat3']#['300VW-Test/cat1','300VW-Test/cat2','300VW-Test/cat3']
    arrName3D = ['300VW-Test_M/cat1','300VW-Test_M/cat2','300VW-Test_M/cat3']
    
    from face_classifier_simple import face_classifier_simple
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    config.gpu_options.per_process_gpu_memory_fraction = .1
    g_c = tf.Graph() ## This is another graph for classifier
    sess_c = tf.InteractiveSession(graph = g_c,config=config)
    with g_c.as_default():
        print("Inititaing classifier")
        f = face_classifier_simple(patchOSize,1)
        x_c,_,pred_c = f.build()
    
    
        saver2 = tf.train.Saver()
        #print("Tensor2 : ",saver2.saver_def.filename_tensor_name)
        saver2.restore(sess_c, tf.train.latest_checkpoint(curDir + 'src/models/classifier'))
    
    name_save = "dt-inception"
    g_d = tf.Graph() ## This is graph for localiser
    
    with g_d.as_default() : 
        f = face_localiser(128,False,3)
        x,y,pred = f.build()
        sess = tf.InteractiveSession(graph = g_d, config = config)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(curDir + 'src/models/'+name_save))
        
    
    
    import torch
    import torchvision.utils as vutils
    from torchvision.transforms import transforms
    from utils import UnNormalize#, unnormalizedAndLandmark, plotImages
    import cv2
    
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np 
    
    image_size = 224
    
    from NetworkModels import FacialLocaliser, DAEE, LogisticRegression, DAEEH,\
        GeneralDAE
    
    #Denoiser 
    
    cl_type = 3#1 is combine, 2 is half, 0 is general 
    usePrevious = True
    
    model_dir = ['model/','model/']
    unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    
    #General Denoiser 
    
    net = DAEE()
    net.load_state_dict(torch.load(model_dir[1]+"AESE_WB_WE_3x3_224"+'.pt'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    net.eval()
    
    #Multiple expert denoiser 
    
    GD = GeneralDAE(type=2)
    
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
        ])
    
    if cl_type == 1 : 
        netAEC = DAEE()
        netAEC.load_state_dict(torch.load(model_dir[0]+'combineAE.pt'))
        netAEC = netAEC.cuda()
        netAEC.eval()
        
        model_lg = LogisticRegression(512, 5)
        model_lg.load_state_dict(torch.load(model_dir[0]+'combineCL.pt'))
        model_lg = model_lg.cuda()
        model_lg.eval()
    else : 
        netAEC = DAEEH()
        netAEC.load_state_dict(torch.load(model_dir[0]+'combineAEH.pt'))
        netAEC = netAEC.cuda()
        netAEC.eval()
        
        model_lg = LogisticRegression(512, 5)
        model_lg.load_state_dict(torch.load(model_dir[0]+'combineCLH.pt'))
        model_lg = model_lg.cuda()
        model_lg.eval()
    
    
    #Now the testing 
    for is3D in is3Dx :
        
        channels = 3
        doTransformation = True
    
        name_save = "dt-inception"
        
        if is3D : 
            name_save += "-3D"
            
        n_o = 136
        err_name = name_save+"-err"
        
        fullDimension = False
        fourChannel = addChannel
        
        if fullDimension  : 
            name_save += "-full"
            err_name += "-full"
            gpu_fract = 1
        
        
        for catTesting in range(1): 
            
            if is3D : 
                folderToTry = arrName3D[catTesting]#"Menpo-3D"#arrName[catTesting]
            else : 
                folderToTry = arrName[catTesting]
                
            all_batch,all_labels,_ = utils.get_kp_face(None,[folderToTry],per_folder=True,n_skip = 1, is3D = is3D)
            print(("Total folder "+str(len(all_batch))))
            
            list_err_file = []
            totalData = 0
            
            config = tf.ConfigProto()
            config.gpu_options.visible_device_list = "0"
            
            u_bb = utilized_BB[iBB]
            
            for folder_i in range(len(all_batch)): 
                
                folder_name = None 
                #now getting the folder name 
                sample_image = all_batch[folder_i][0]
                folder_name = os.path.split(os.path.split(os.path.dirname(sample_image))[0])[1]
                
                list_err_file.append(curDir + "images/"+folderToTry+"/"+str(folder_name)+"_"+ext)
                        
                my_file = Path(curDir + "images/"+folderToTry+"/"+str(folder_name)+"_"+ext)
                if my_file.is_file():
                    print((curDir + "images/"+folderToTry+"/"+str(folder_name)+"_"+ext+" is exist "))
                    #continue
                
                if write_data : 
                    file = open(curDir + "images/"+folderToTry+"/"+str(folder_name)+"_"+ext,'w') 
                    
                np_name = curDir + "images/"+folderToTry+"/"+str(folder_name)+"_"+np_ext
                
                #get the utilized BB , 537_TR.RT_err_txt
                fileBB = curDir + "images/"+folderToTry+"/"+str(folder_name)+"_"+trailing_BB[iBB]+"."+u_bb+"_bbs_txt"
                
                l_bb = []
                file2 = open(fileBB)
                for line in file2 : 
                    #print float(line)
                    #print(line)
                    data = [ float(j) for j in line.split()] 
                    #print(data)
                    l_bb.append([float(data[0]),float(data[1]),float(data[2]),float(data[3])])
                file2.close()
                
                #Now taking the data and groundtruth for this folder
                list_images =[]
                cBBPoints2 = []
                y_batch = np.zeros([len(all_batch[folder_i]),n_o])
                 
                for b_j in range(len(all_batch[folder_i])) : 
                    #print len(all_batch[folder_i])
                    #fetch the data for each batch 
                    list_images.append(all_batch[folder_i][b_j])
                    cBBPoints2.append(all_labels[folder_i][b_j])
                    
                seq_length = len(list_images)
                totalData+=seq_length
                indexer = 0
                l_t = None
                
                the_res = []
                anyHard = False
                
                while (indexer < seq_length) :
                    if write_data : 
                        l_hard = open(curDir + "src/hardDFAL.txt",'w')
                     
                    tImage = cv2.imread(list_images[indexer])
                    
                    
                    
                    #####
                    datasetName = '300VW-Test'
                    if cl_type == 0 : 
                        targetDir = '/media/deckyal/78DCCAA06404648F/v-cleanedAE/'+datasetName + "/" 
                    elif cl_type == 1 : 
                        targetDir = '/media/deckyal/78DCCAA06404648F/v-cleanedAECL-E-CB/'+datasetName + "/"
                    elif cl_type == 2 : 
                        targetDir = '/media/deckyal/78DCCAA06404648F/v-cleanedCL-E-CB/'+datasetName + "/" 
                    else : 
                        targetDir = '/media/deckyal/78DCCAA06404648F/v-ori/'+datasetName + "/" 
                    
                    
                    #now getting the name and file path 
                    filePath = list_images[indexer].split(os.sep)
                    ifolder = filePath.index(datasetName)
                    
                    image_name = filePath[-1]
                    annot_name = os.path.splitext(image_name)[0]+'.pts'
                    
                    middle = filePath[ifolder+2:-2]
                    #print(middle)
                    middle = '/'.join(middle)
                    
                    finalTargetPathI = targetDir+middle+'/img/'
                    finalTargetPathA = targetDir+middle+'/annot/'
                    
                    checkDirMake(finalTargetPathI)
                    checkDirMake(finalTargetPathA)
                    
                    finalTargetImage = finalTargetPathI+image_name
                    finalTargetAnnot = finalTargetPathA+annot_name
                    #print(finalTargetImage, finalTargetAnnot)
                    #####
                    
                    
                    print(list_images[indexer])
                    
                    y_batch = np.asarray(cBBPoints2[indexer]).copy()
                    notFace = False 
                    
                    if usePrevious : 
                        if l_t is None  : 
                            l_t = l_bb[indexer]
                        t = l_t
                    else : 
                        t = l_bb[indexer]
                    
                    if t[0]>= 9999 : 
                        res = None
                        notFace = True
                    else : 
                        the_bb = [t[0],
                              t[1],
                              t[2],
                              t[3]]
                        #print the_bb
                        print("Original bb : ",the_bb)
                        
                        '''for z22 in range(68) :
                            cv2.circle(tImage,(int(y_batch[z22]),int(y_batch[z22+68])),2,(0,255,0))
                        cv2.imshow('test',tImage)
                        cv2.waitKey(0)'''
                        #first clean the image
                        #1 crop the image and resize given bb  
                        l_x = (t[2]-t[0])/2
                        l_y = (t[3]-t[1])/2  
                        
                        x1 = int(max(t[0] - l_x,0))
                        y1 = int(max(t[1] - l_y,0))
                        
                        #print tImage.shape
                        x2 = int(min(t[2] + l_x,tImage.shape[1]))
                        y2 = int(min(t[3] + l_y,tImage.shape[0]))
                        
                        tImage2 = tImage[y1:y2,x1:x2].copy()
                        
                        print((x1,y1,x2,y2))
                        #for recovery 
                        height, width,_ = tImage2.shape
                        
                        ratioWidth =truediv(width,image_size)
                        ratioHeight =truediv(height,image_size)
                        
                                                    
                        #2 normalize and change to PIL 
                        cv2_im = cv2.cvtColor(tImage2,cv2.COLOR_BGR2RGB)
                        pil_im = Image.fromarray(cv2_im)
                        pil_im = transform(pil_im.resize((image_size,image_size)))
                        pil_im = pil_im.unsqueeze(0).cuda()
                        
                        #print(pil_im.shape)
                        #Now the cleaning 
                        if cl_type == 0 : 
                            res  = net.forward(pil_im)[0].detach().cpu()
                            #print('res',res.shape)
                            res = res[0]
                        elif cl_type == 1 : 
                            recon_batch,xe = netAEC(pil_im)
                            labels = model_lg(xe)
                            x_, y_ = torch.max(labels, 1)
                            print('status ',y_)
                            res = GD.forward(pil_im, y_[0])[0]
                        elif cl_type == 2 :  
                            xe = netAEC(pil_im)
                            labels = model_lg(xe)
                            x_, y_ = torch.max(labels, 1)
                            print('status ',y_)
                            res = GD.forward(pil_im, y_[0])[0]
                        else : 
                            res = pil_im[0].cpu()
                            
                        #now the landmark
                        print('res',res.shape)                            
                        #1 unnormalize and change to cv2. 
                        resN =  unorm(res.clone()).numpy()*255 
                        resN = resN.transpose((1,2,0))
                        #print('resN',resN.shape)
                        img_orix = cv2.cvtColor(resN.astype(np.uint8 ),cv2.COLOR_RGB2BGR)
                        
                        if cl_type < 3 : 
                            print(height,width,y1)
                            tImage[y1:y2,x1:x2] = cv2.resize(img_orix,(width,height))
                            
                            #saving the intermediate image 
                            cv2.imwrite(finalTargetImage,tImage)
                            
                            if cl_type in [1,2] : 
                                if y_[0]!= 0 : 
                                    cv2.rectangle(tImage,(x1,y1),(x2,y2),(255,0,255),1)
                            
                        
                        #print('resn',resN)
                        '''cv2.imshow('img_orix',img_orix)
                        cv2.waitKey(0)
                        cv2.imshow('img_orix',tImage2)
                        cv2.waitKey(0)'''
                        #res = fa.get_landmarks_1(input_image =tImage,bb=the_bb)
                        #res = pred.forward(img_orix)
                        
                        height, width, channels = img_orix.shape
                        ratioHeightR =truediv(height,128)
                        ratioWidthR =truediv(width,128)
                        r_image = cv2.resize(img_orix, (128,128))
                        
                        '''cv2.imshow('img_orix',r_image)
                        cv2.waitKey(0)'''
                                     
                        predicted = pred.eval(feed_dict = {x:np.expand_dims(r_image, axis=0)},session = sess)[0]
                        
                        '''for z22 in range(68) :
                            cv2.circle(r_image,(int(predicted[z22]),int(predicted[z22+68])),2,(0,255,0))
                        cv2.imshow('test',r_image)
                        cv2.waitKey(0)
                        '''
                        predicted[0:68]*=ratioWidthR
                        predicted[68:136]*=ratioHeightR 
                        
                        '''for z22 in range(68) :
                            cv2.circle(img_orix,(int(predicted[z22]),int(predicted[z22+68])),2,(0,255,0))
                        cv2.imshow('test',img_orix)
                        cv2.waitKey(0)'''
                        
                        #now projecting result to the original coordinate
                        predicted[:68] *= ratioWidth
                        predicted[68:] *= ratioHeight
                        
                        '''for z22 in range(68) :
                            cv2.circle(tImage2,(int(predicted[z22]),int(predicted[z22+68])),2,(0,255,0))
                        cv2.imshow('test',tImage2)
                        cv2.waitKey(0)'''
                        
                        predicted[:68] += x1 
                        predicted[68:] += y1
                        
                        '''predicted = res[-1]
                        x_list = predicted[:,0]
                        y_list = predicted[:,1]
                        
                        predicted = np.concatenate((x_list,y_list),axis=0)
                        '''
                        
                        '''for z22 in range(68) :
                            cv2.circle(tImage,(int(predicted[z22]),int(predicted[z22+68])),2,(0,255,0))
                        cv2.imshow('test',tImage)
                        cv2.waitKey(0)'''
                        
                        #temp = res[-1]
                        temp = predicted
                        #now check if the result is face or not. 
                        exp_face = np.zeros([1,68,patchOSize,patchOSize,3])
                        t = utils.get_bb(temp[:68], temp[68:])
                    
                        l_x = (t[2]-t[0])/2 + (t[2]-t[0])/4
                        l_y = (t[3]-t[1])/2 + (t[3]-t[1])/4 
                        
                        x1 = int(max(t[0] - l_x,0))
                        y1 = int(max(t[1] - l_y,0))
                        
                        #print tImage.shape
                        x2 = int(min(t[2] + l_x,tImage.shape[1]))
                        y2 = int(min(t[3] + l_y,tImage.shape[0]))
                        
                        if (np.abs(y1-y2) <= 1): 
                            y2 = y1+10;
                        if (np.abs(x1-x2) <= 1): 
                            y2 = y1+10;
                        
                        
                        tImage2 = tImage[y1:y2,x1:x2].copy();
                        
                        height, width,_ = tImage2.shape
                                
                        ratioHeight =truediv(cropSize,height)
                        ratioWidth =truediv(cropSize,width)
                        
                        '''
                        print(image_size,height,width)
                        print(ratioHeight,ratioWidth)'''
                        
                        tImage2 = cv2.resize(tImage2,(cropSize,cropSize)).copy()
                        
                        #Now fixing the groundtruth 
                        kpX = (temp[:68] - x1)*ratioWidth
                        kpY = (temp[68:] - y1)*ratioHeight
                        
                        '''for z22 in range(68) :
                            cv2.circle(tImage2,(int(kpX[z22]),int(kpY[z22])),2,(0,255,0))
                        cv2.imshow('test',tImage2)
                        cv2.waitKey(0)'''
                        
                        for k_2 in range(68) : 
                            x_2,y_2 = int(kpX[k_2]),int(kpY[k_2])
                            t_image = np.zeros([patchOSize,patchOSize,3])
                            t_image[0:(utils.inBound(int(y_2+patchSize),0,tImage2.shape[0]) - utils.inBound(int(y_2-patchSize),0,tImage2.shape[0])),
                                    0:(utils.inBound(int(x_2+patchSize),0,tImage2.shape[1]) - utils.inBound(int(x_2-patchSize),0,tImage2.shape[1]))
                                    ] = tImage2[utils.inBound(int(y_2-patchSize),0,tImage2.shape[0]):utils.inBound(int(y_2+patchSize),0,tImage2.shape[0]),
                                               utils.inBound(int(x_2-patchSize),0,tImage2.shape[1]):utils.inBound(int(x_2+patchSize),0,tImage2.shape[1])]
                            exp_face[0,k_2] = t_image
                            
                            '''cv2.rectangle(tImage2,(utils.inBound(int(x_2-patchSize),0,tImage2.shape[1]),utils.inBound(int(y_2-patchSize),0,tImage2.shape[0])),
                                               (utils.inBound(int(x_2+patchSize),0,tImage2.shape[1]),utils.inBound(int(y_2+patchSize),0,tImage2.shape[0])),(0,255,0),1)
                            '''
                            #cv2.imwrite('test_'+str(k_2)+".jpg",t_image)
                        #cv2.imshow("test",tImage2)
                        #cv2.waitKey(0)
                        
                        is_face = sess_c.run(pred_c,feed_dict = {x_c:exp_face})
                        
                        f_index = sigmoid(np.squeeze(is_face))
                        
                        if f_index < 0.5 : 
                            print("Not a face ")
                            notFace = True 
                    
                    print(indexer)
                    #if not face, use the last_bb instead
                    if indexer > 0 and notFace and l_t is not None :
                        if usePrevious : 
                            print("do redetection")
                            l_t = l_bb[indexer]
                        else : 
                            print("Trying to use the last one") 
                        t = l_t
                        the_bb = [t[0],
                          t[1],
                          t[2],
                          t[3]]
                        #print the_bb
                        print("length : ",t[2]-t[0], t[3]-t[1],t)
                        if(np.abs(t[2]-t[0]) <= 5 or np.abs(t[3]-t[1]) <= 5): 
                            res = None
                        else :
                            
                            l_x = (t[2]-t[0])/2
                            l_y = (t[3]-t[1])/2  
                            
                            x1 = int(max(t[0] - l_x,0))
                            y1 = int(max(t[1] - l_y,0))
                            
                            #print tImage.shape
                            x2 = int(min(t[2] + l_x,tImage.shape[1]))
                            y2 = int(min(t[3] + l_y,tImage.shape[0]))
                            
                            tImage2 = tImage[y1:y2,x1:x2].copy()
                            
                            print((x1,y1,x2,y2))
                            #for recovery 
                            height, width,_ = tImage2.shape
                            
                            ratioWidth =truediv(width,image_size)
                            ratioHeight =truediv(height,image_size)
                            
                                                        
                            #2 normalize and change to PIL 
                            cv2_im = cv2.cvtColor(tImage2,cv2.COLOR_BGR2RGB)
                            pil_im = Image.fromarray(cv2_im)
                            pil_im = transform(pil_im.resize((image_size,image_size)))
                            pil_im = pil_im.unsqueeze(0).cuda()
                            
                            print(pil_im.shape)
                            #Now the cleaning 
                            if cl_type == 0 : 
                                res  = net.forward(pil_im)[0].detach().cpu()
                                #print('res',res.shape)
                                res = res[0]
                            elif cl_type == 1 : 
                                recon_batch,xe = netAEC(pil_im)
                                labels = model_lg(xe)
                                x_, y_ = torch.max(labels, 1)
                                print('status ',y_)
                                res = GD.forward(pil_im, y_[0])[0]
                            elif cl_type == 2 :  
                                xe = netAEC(pil_im)
                                labels = model_lg(xe)
                                x_, y_ = torch.max(labels, 1)
                                print('status ',y_)
                                res = GD.forward(pil_im, y_[0])[0]
                            else : 
                                res = pil_im[0].cpu()
                                                        #now the landmark
                            #1 unnormalize and change to cv2. 
                            resN =  unorm(res.clone()).numpy()*255 
                            resN = resN.transpose((1,2,0))
                            #print('resN',resN.shape)
                            img_orix = cv2.cvtColor(resN.astype(np.uint8 ),cv2.COLOR_RGB2BGR)
                            
                            height, width, channels = img_orix.shape
                            ratioHeightR =truediv(height,128)
                            ratioWidthR =truediv(width,128)
                            r_image = cv2.resize(img_orix, (128,128))
                            
                            predicted = pred.eval(feed_dict = {x:np.expand_dims(r_image, axis=0)},session = sess)[0]
                            
                            predicted[0:68]*=ratioWidthR
                            predicted[68:136]*=ratioHeightR 
                            #now projecting result to the original coordinate
                            predicted[:68] *= ratioWidth
                            predicted[68:] *= ratioHeight
                            
                            predicted[:68] += x1 
                            predicted[68:] += y1
                        
                                
                        
                    if res is None : 
                        print("no face were detected")
                        curr_err = 99999
                        print(("Curr err : "+str(curr_err)+"_"+str(indexer)+"/"+str(seq_length)))
                        if write_data : 
                            file.write("%.4f\n" % (curr_err))
                    else :
                        for z22 in range(68) :
                            cv2.circle(tImage,(int(predicted[z22]),int(predicted[z22+68])),2,(0,255,0))
                            cv2.circle(tImage,(int(y_batch[z22]),int(y_batch[z22+68])),2,(0,0,255))

localise(0)
        







#####----#############

'''
Created on Feb 20, 2018

@author: deckyal
'''

import cv2
from face_classifier_simple import face_classifier_simple
import tensorflow as tf
from operator import truediv
import numpy as np
import utils
from config import *
from TrackerModifiedInception import frameTracking
import math
from face_localiser import face_localiser
from face_classifier_simple import face_classifier_simple


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def show_webcam(mirror=False):
    
    from MTCNN import MTCNN
    
    bb_mode = True
    
    addChannelLocaliser = False
    trained_length = 4
    
    is3D = True
    
    channels = 3
    baseSize = 128
    
    cropSize = baseSize + int(baseSize*.5)
    
    patchOSize = int(baseSize*.1)
    patchSize = int(patchOSize/2)
    
    n_o = 136
    
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "1"
    
    
    name_save = "kp-transfer"
    
    if is3D : 
        name_save += "-3D"
    
    name_save+="-"+str(trained_length)
            
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = "0"
    config.gpu_options.per_process_gpu_memory_fraction = .2
    
    g_t = tf.Graph() ## This is graph for tracking
    g_c = tf.Graph() ## This is another graph for classifier
    g_l = tf.Graph() ## This is another graph for localiser
    
    sess = tf.InteractiveSession(graph = g_t,config=config)
    sess_c = tf.InteractiveSession(graph = g_c,config=config)
    sess_l = tf.InteractiveSession(graph = g_l,config=config)
    
    with g_t.as_default():
              
        print("*Initiating FT")
        ft = frameTracking(1, 2, crop_size, channels, n_neurons, n_o,learning_rate=.01,test=True,model_name=name_save,dataType = 2,n_adder=0,CNNTrainable=True,realTime =True)
        print("*Initiating builder")
        #   x,c_state   ,c_state2   ,h_state,h_state2,  y,states,   states_to_ori,  l_gates     ,xy,final_y,sample_image,LSTMState,z2           ,initial_BB
        if useDoubleLSTM: 
            x,c         ,c2         ,h      ,h2,        y,preds     ,   preds_ori,      the_gates   ,xy,finalY, sampleImage ,LSTMState,phase_train  ,in_bb,     LSTMState2= ft.buildTracker()
        else : 
            x,c,c2,h,h2,y,pgt,preds_ori,the_gates,xy,finalY,sampleImage,LSTMState,phase_train,in_bb= ft.buildTracker()
        
        saver = tf.train.Saver(var_list=tf.trainable_variables(scope="(?!additional)"))
        print(tf.trainable_variables(scope="(?!additional)"))
        print(curDir + 'src/models/'+name_save)
        #print("Tensor1 : ",saver.saver_def.filename_tensor_name)
        saver.restore(sess, tf.train.latest_checkpoint(curDir + '/models/'+name_save))
        the_state = tf.contrib.rnn.BasicLSTMCell(n_neurons).zero_state(1, tf.float32)
    
    
    with g_c.as_default():
        print("Inititaing classifier")
        f = face_classifier_simple(patchOSize,1)
        x_c,_,pred_c = f.build()
    
    
        saver2 = tf.train.Saver()
        #print("Tensor2 : ",saver2.saver_def.filename_tensor_name)
        saver2.restore(sess_c, tf.train.latest_checkpoint(curDir + '/models/classifier'))
    
    channels_l = 3
    
    if addChannelLocaliser : 
        g_l2d = tf.Graph() ## This is another graph for localiser    
        sess_l2d = tf.InteractiveSession(graph = g_l,config=config)
        with g_l2d.as_default(): 
            f2d = face_localiser(crop_size,False,channels_l)
            x_l2d,y_l2d,pred_l2d = f.build()
            name_localiser = "dt-inception"
            
            if is3D : 
                name_localiser+= "-3D"
                
            saver32d = tf.train.Saver()
            #print("Tensor2 : ",saver2.saver_def.filename_tensor_name)
            saver32d.restore(sess_l, tf.train.latest_checkpoint(curDir + '/models/'+name_localiser))
        

        
    with g_l.as_default():
        print("Inititaing localiser")    
        
        if addChannelLocaliser : 
            channels_l = 4
        
        f = face_localiser(crop_size,False,channels_l)
        x_l,y_l,pred_l = f.build()
        
        name_localiser = "dt-inception"
        
        if is3D : 
            name_localiser+= "-3D"
        
        if addChannelLocaliser : 
            name_localiser+="-4D"
            
        saver3 = tf.train.Saver()
        #print("Tensor2 : ",saver2.saver_def.filename_tensor_name)
        saver3.restore(sess_l, tf.train.latest_checkpoint(curDir + '/models/'+name_localiser))
    
    model = MTCNN()
    
    restartStateEvery = trained_length
            
    stt = sess.run(the_state) #restatrt the state
    stt2 = sess.run(the_state) #restatrt the state
    stt_before = sess.run(the_state)
    stt_before2 = sess.run(the_state)
    
    toRestart = False #configured in loop when to do restart
    
    doRestart = True #whter to restart or not. Parameter
    use_shifted_kp = True
    
    always_detect = False
    
    lastBB = None
    detected_BB = None
    
    l_r = None
    indexer = 0
    
    lp_BB = None
    distanceLength = bbDiagLength
    
    countBeforeRestart = 0 
    
    whenToRestart = 0#restartStateEvery
    runningCount = 0
    prevDetect = False
    use_shadow = True#is to use the predicted KP as the basis of classifieer 
    firstBB = None
    
    
    choosen_kp = None #This is to select whether to use the localised from BB of predicted lstm kp or from detected bb
    lastImage = None
    
    i = 0
    anyFace = False
    y_t = None
    ratioWidth_ori = None
    
    if useWebCam :
        target = 0 
    else: 
        target = videoName 
        
    cam = cv2.VideoCapture(target)
    
    writeVideo = True
    
    if writeVideo :
        file_path = target  # change to your own video path
        vid = cv2.VideoCapture(file_path)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        
        print(height,width)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        cap = cv2.VideoWriter(outputVideo ,fourcc,20.0, (int(width),int(height)))
        
    while cam.isOpened():
        
        print("Seq ",i)
        
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        
        try : 
            image = img.copy();
        except : 
            break; 
        
        if ratioWidth_ori is None : 
            height, width, _ = img.shape
            ratioWidth_ori = truediv(imWidth, width )
            ratioHeight_ori = truediv(imHeight,height)
        
        #print((ratioWidth,ratioHeight,width, imWidth,height,imHeight))
            
        if not anyFace : 
            
            predictions,_ = model.doFaceDetection(img)
        
            if len(predictions) > 0: 
                
                for box in predictions : 
                    print(box)
                    
                    t=np.zeros(4)
                    left = t[0] = int(box[0])
                    right = t[2]  = int(box[2]) 
                    top = t[1]=  int(box[1]) 
                    bot = t[3] = int(box[3])
                        
                    #x1,y1, x2,y2 
                    print((left, top, right, bot))
                    cv2.rectangle(img,(left, top), (right, bot),(0,255,0),3)
                    
                    lastBB_pp = np.expand_dims(np.array([left*ratioWidth_ori,right*ratioWidth_ori,top*ratioHeight_ori,bot*ratioHeight_ori]), axis=0)
                    
                    #input is  be x1,x2,y1,y2
                    lastBB = np.expand_dims(np.array([left*ratioWidth_ori,right*ratioWidth_ori,top*ratioHeight_ori,bot*ratioHeight_ori]), axis=0)
                    anyFace = True;
                    
                    lp_BB = lastBB
                    
                    if firstBB is None : 
                        firstBB = lp_BB
                        
                    indexer = 0;
        else :
            detected_BB = None
            if (toRestart and doRestart) or always_detect:
                print("Now detecting")
                prevDetect = True
                
                imgD = image
                    
                height, width, _ = imgD.shape
                
                predictions,_ = model.doFaceDetection(imgD)
                    
                    
                l_lastBB_p = []
                l_distBB = []
                l_diag_p = []
                l_diag_pp = []
                
                if len(predictions) > 0:
                    for box in predictions : 
                        t=np.zeros(4)
                        
                        left = t[0] = int(box[0])
                        right = t[2]  = int(box[2]) 
                        top = t[1]=  int(box[1]) 
                        bot = t[3] = int(box[3])
                        
                        imgD = cv2.resize(imgD,(imWidth,imHeight)).copy()
                        #cv2.rectangle(img,(int(left*ratioWidth), int(top*ratioHeight)), (int(right*ratioWidth), int(bot*ratioHeight)),(0,255,0),3)
                        
                        #input is  be x1,x2,y1,y2
                        lastBB_pp = np.expand_dims(np.array([left*ratioWidth_ori,right*ratioWidth_ori,top*ratioHeight_ori,bot*ratioHeight_ori]), axis=0)
                        cBB_p = np.array([(lp_BB[0,0]+lp_BB[0,1])/2,(lp_BB[0,2]+lp_BB[0,3])/2])
                        cBB_pp = np.array([(lastBB_pp[0,0]+lastBB_pp[0,1])/2,(lastBB_pp[0,2]+lastBB_pp[0,3])/2])
                        
                        diag_p = np.sqrt(np.square(lp_BB[0,0]-lp_BB[0,1]) + np.square(lp_BB[0,2]-lp_BB[0,3]))  
                        diag_pp = np.sqrt(np.square(lastBB_pp[0,0]-lastBB_pp[0,1]) + np.square(lastBB_pp[0,2]-lastBB_pp[0,3]))
                        
                        #distance between center of last bb and current detected BB
                        dist = np.sqrt(np.square(cBB_p[0]-cBB_pp[0]) + np.square(cBB_p[1]-cBB_pp[1]))
                        
                        l_distBB.append(dist)
                        l_lastBB_p.append(lastBB_pp)
                        l_diag_p.append(diag_p)
                        l_diag_pp.append(diag_pp)
                    
                    anyClose = False
                    n_close = 0
                    minDistance = 9999
                    
                    
                    if len(predictions) >= 1: 
                        print("multiple BB!",len(predictions))
                        
                        for lnx in range(0,len(l_distBB)) :
                            if ( (l_distBB[lnx] < l_diag_p[lnx] * distanceLength)):# and np.abs(l_diag_p[lnx] - l_diag_pp[lnx]) < bbDiagLength*l_diag_p[lnx]):
                                print("got  the bb",lnx) 
                                if (l_distBB[lnx] < minDistance): 
                                    lastBB = l_lastBB_p[lnx]
                                    minDistance =l_distBB[lnx]
                                     
                                if not anyClose : 
                                    anyClose = True
                                n_close+=1
                                
                    if not anyClose : 
                        print("Does not found any close")
                        lastBB = lp_BB
                        
                else :#if there's no face 
                    print("No face detected!")
                    lastBB = lp_BB 
                #toRestart = False
                
                    print(firstBB, firstBB[0])
                t_bb = lastBB[0]
                detected_BB = [t_bb[0],t_bb[2],t_bb[1],t_bb[3]]
                
            if (runningCount % restartStateEvery == 0 or toRestart) and doRestart :
                print("Restarting the state")
                runningCount = 0
                stt = stt_before
                stt2 = stt_before2
            elif indexer == 1 : 
                stt_before = stt
                stt_before2 = stt2
                
            temp_batch,cBBPoints = [],[]
            
            t_b,y_b,y_o,ori_y = [],[],[],[] 
            
            for j in range (2): 
                
                if j == 0: 
                    index = lastImage 
                else : 
                    index = image
                    
                images = index
                
                
                height, width,_ = images.shape
            
                ratioHeight =truediv(imHeight,height)
                ratioWidth =truediv(imWidth,width)
                
                images = cv2.resize(images,(imWidth,imHeight)).copy()
                t_b.append(images)
                
            temp_batch.append(t_b)
            cBBPoints.append(y_b)
        
            x_batch = np.asarray(temp_batch)
            
            y_batch  = np.zeros([1,2,136])#l_r
            
            input_y = y_batch
            if indexer == 0 : 
                input_y[:,1] = y_batch[:,0]
                
                #toFind = np.squeeze(y_batch[:,0])
                lastBB = lp_BB#np.expand_dims(utils.get_bb(toFind[:68],toFind[68:],68,True),axis=0)
                
                #print(lastBB)
            else : 
                if lastBB is None : 
                    lastBB = np.expand_dims(utils.get_bb(np.squeeze(l_r[:,0,:68]),np.squeeze(l_r[:,0,68:]),68,True),axis = 0)
        
            #print(x_batch.shape,input_y.shape)
            
            #******* This is the tracking part 
            
            print("l_bb : ",lastBB,' ',indexer)
            cb = lastBB[0]
            
            proposeBB = None
            
            if np.abs(cb[0]-cb[1]) <= 50 or np.abs(cb[2]-cb[3]) <=50: #check if the bb is 0
                print ("less than 0")
                if True :   
                    toFind = firstBB#np.squeeze(y_batch[:,0])
                    lastBB = toFind #np.expand_dims(utils.get_bb(toFind[:68],toFind[68:],68,True),axis=0) #recover using the first BB information
                else : 
                    #recover using the height and width of first BB information 
                    tempBB_GT = firstBB[0] #utils.get_bb(toFind[:68],toFind[68:],68,True)
                    tempBB= lastBB[0]
                    
                    l_x = np.abs(tempBB_GT[1] - tempBB_GT[0])/2
                    l_y = np.abs(tempBB_GT[3] - tempBB_GT[2])/2
                    
                    c_x = (tempBB[1] + tempBB[0])/2 
                    c_y = (tempBB[2] + tempBB[3])/2
                    
                    height, width,_ = lastImage.shape
                    proposeBB = [max(c_x - l_x,0),min(c_x+l_x,width),max(c_y - l_y,0),min(c_y+l_y,height)  ]
                    print("Propose BB ", proposeBB)
                    
                    lastBB = np.expand_dims(proposeBB,axis = 0)
                
                t_bb = lastBB[0]    
                detected_BB = [t_bb[0],t_bb[2],t_bb[1],t_bb[3]]
                
                print("Detected bb : ",detected_BB)
                if proposeBB is not None : 
                    print("Propose BB : ", proposeBB)
            
            
            if not (useDoubleLSTM) :
                l_r,stt= sess.run([preds_ori,LSTMState],feed_dict = {x:x_batch, y:input_y,c : stt.c,h : stt.h,c2 : stt.c,h2 : stt.h,phase_train:False,in_bb: lastBB})
            else : 
                print("using double lstm")
                l_r,st,fy,stt,stt2= sess.run([preds_ori,preds,finalY,LSTMState,LSTMState2],feed_dict = {x:x_batch, y:input_y, c : stt.c, h : stt.h, c2 : stt2.c, h2 : stt2.h, phase_train:False,in_bb: lastBB})
            
            print(utils.get_enlarged_bb(l_r[0][0],2,2,images))
            
            if(detected_BB is not None) : 
                print(utils.get_enlarged_bb(detected_BB,2,2,images,is_bb=True))
            
            if bb_mode : 
                if prevDetect : 
                    numCheck = 2;
                else :
                    numCheck = 1;
                    
                if indexer == 0 : 
                    correctedBB = [lastBB[0][0],lastBB[0][2],lastBB[0][1],lastBB[0][3]]
                    l_bbs = [utils.get_enlarged_bb(correctedBB,2,2,images,is_bb=True),utils.get_enlarged_bb(correctedBB,2,2,images,is_bb=True)]
                    the_kp = [y_t, y_t]
                else : 
                    if detected_BB is None : 
                        detected_BB = utils.get_bb(l_r[0][0][:68],l_r[0][0][68:])
                        
                    #get the list of BB to be evaluated 
                    print("None",detected_BB,np.asarray(detected_BB).shape)
                    
                    l_bbs = [utils.get_enlarged_bb(l_r[0][0],2,2,images),utils.get_enlarged_bb(detected_BB,2,2,images,is_bb=True)]
                    the_kp = [l_r[0][0],choosen_kp]#problem o#n the second one, that to use the localized one 
                
                l_predict = []
                
                #****** This is the localisation part 
                
                for i_check in range(numCheck): #check to use between the detected BB or from KP
                    
                    #print(l_bbs,l_bbs[i_check], len(l_bbs), len(l_bbs[i_check]))
                    t,l_x,l_y,x1,y1,x_min,y_min,x2,y2 = l_bbs[i_check][0],l_bbs[i_check][1],l_bbs[i_check][2],l_bbs[i_check][3],l_bbs[i_check][4],l_bbs[i_check][5],l_bbs[i_check][6],l_bbs[i_check][7],l_bbs[i_check][8]
                    # detectedBB
                    
                    #get the heatmap 
                    if is3D : 
                        add ="3D"
                    else : 
                        add ="2D"
                        
                    print(i_check,t)
                    
                    print("test",x1,x2,y1,y2)
                    croppedImage = images[y1:y2,x1:x2];
                    height, width, channels = croppedImage.shape
                
                    ratioHeightR =truediv(height,crop_size)
                    ratioWidthR =truediv(width,crop_size)
                
                    #print ratioHeight,import configratioWidth
                    r_image = cv2.resize(croppedImage, (crop_size,crop_size))[:,:,:3] #dismiss the channel from the previous one

                    #get the recently calculated heatmap. If any use it, otherwise calculate it
                    if addChannelLocaliser :
                        b_channel,g_channel,r_channel = r_image[:,:,0],r_image[:,:,1],r_image[:,:,2]
                        
                        to_use_kp = pred_l2d.eval(feed_dict = {x_l:np.expand_dims(r_image, axis=0)},session = sess_l2d)[0]
                        
                        newChannel = utils.make_heatmap(list_images[indexer],images,add,to_use_kp,False,.1,.05)
                        images = cv2.merge((b_channel, g_channel,r_channel, newChannel))
                        heatmaps = cv2.merge((b_channel, newChannel,newChannel, newChannel))
                        
                    predicted = pred_l.eval(feed_dict = {x_l:np.expand_dims(r_image, axis=0)},session = sess_l)[0]
                    
                    predicted[:68] = predicted[:68]*ratioWidthR + x_min
                    predicted[68:] = predicted[68:]*ratioHeightR +y_min
                    l_predict.append(predicted)
                    
            if bb_mode : 
                listCenterPred  = l_predict
            else : 
                listCenterPred  = [np.array(l_r[0][0])]
            
            if use_shadow : 
                listCenterPred.append(np.array(l_r[0][0])) #use shadow mean to use the predicted (direct) kp as also measurement of whether it is face or not
                
            
            #after the detection is made. Usually happen on the hard problem.
            min_classifier = 1;
            
            l_f_classifier = []
            
            #print(listCenterPred)#
            
            #*****this is the classification part
            
            if indexer !=0 :#and indexer%restartStateEvery == 0 :
                for i_chosen in range(0,len(listCenterPred)) :  
                    
                    images = x_batch[0][0].copy()
                    
                    b_channel,g_channel,r_channel = images[:,:,0],images[:,:,1],images[:,:,2]
                    tImage = cv2.merge((b_channel, g_channel,r_channel))
                    
                    exp_face = np.zeros([1,68,patchOSize,patchOSize,3])
                    
                    toEvalClassifier = listCenterPred[i_chosen]#l_r[0][0]
                    
                    t = utils.get_bb(toEvalClassifier[:68], toEvalClassifier[68:])
                    
                    l_x = (t[2]-t[0])/2 + (t[2]-t[0])/4
                    l_y = (t[3]-t[1])/2 + (t[3]-t[1])/4 
                    
                    x1 = int(max(t[0] - l_x,0))
                    y1 = int(max(t[1] - l_y,0))
                    
                    #print tImage.shape
                    x2 = int(min(t[2] + l_x,tImage.shape[1]))
                    y2 = int(min(t[3] + l_y,tImage.shape[0]))
                    
                    if (np.abs(y1-y2) <= 1): 
                        y2 = y1+10;
                    if (np.abs(x1-x2) <= 1): 
                        y2 = y1+10;
                    
                    
                    tImage = tImage[y1:y2,x1:x2].copy();
                    
                    height, width,_ = tImage.shape
                        
                    ratioHeight =truediv(cropSize,height)
                    ratioWidth =truediv(cropSize,width)
                                
                    tImage = cv2.resize(tImage,(cropSize,cropSize)).copy()
                    
                    #Now fixing the groundtruth 
                    kpX = (toEvalClassifier[:68] - x1)*ratioWidth
                    kpY = (toEvalClassifier[68:] - y1)*ratioHeight
                    
                    
                    for k_2 in range(68) : 
                        x_2,y_2 = int(kpX[k_2]),int(kpY[k_2])
                        t_image = np.zeros([patchOSize,patchOSize,3])
                        t_image[0:(utils.inBound(int(y_2+patchSize),0,tImage.shape[0]) - utils.inBound(int(y_2-patchSize),0,tImage.shape[0])),
                                0:(utils.inBound(int(x_2+patchSize),0,tImage.shape[1]) - utils.inBound(int(x_2-patchSize),0,tImage.shape[1]))
                                ] = tImage[utils.inBound(int(y_2-patchSize),0,tImage.shape[0]):utils.inBound(int(y_2+patchSize),0,tImage.shape[0]),
                                           utils.inBound(int(x_2-patchSize),0,tImage.shape[1]):utils.inBound(int(x_2+patchSize),0,tImage.shape[1])]
                        exp_face[0,k_2] = t_image
                        
                    
                    is_face = sess_c.run(pred_c,feed_dict = {x_c:exp_face})
                    
                    f_index = sigmoid(np.squeeze(is_face))
                    if f_index < min_classifier: 
                        min_classifier = f_index
                    
                    l_f_classifier.append(f_index)
                
                print("is face : \t \t",min_classifier, "CBR : ",countBeforeRestart," whentorestart ",whenToRestart)
                if min_classifier < thresholdFace :
                    print("Evaluating whether to restart or not **")
                    if countBeforeRestart > whenToRestart : 
                        toRestart = True
                        countBeforeRestart = 0 
                        #continue
                    else :  
                        print("skipping restart**")
                        countBeforeRestart+=1
                        toRestart = False
                else : 
                    print("Face is ok, keep tracking **")
                    prevDetect = False
                    toRestart = False
            
            if prevDetect: 
                if l_f_classifier[0] > l_f_classifier[1]: 
                    choosen_kp =  listCenterPred[0]
                else: 
                    choosen_kp =  listCenterPred[1]
            else : 
                choosen_kp = l_predict[0]#l_r[0][0]
            
            lp_BB = np.expand_dims(utils.get_bb(np.squeeze(choosen_kp[:68]),np.squeeze(choosen_kp[68:]),68,True),axis = 0)
            
            if firstBB is None : 
                firstBB = lp_BB
                
            color = (0,255,0)
            colorBB = [0,255,0]
            
            
            choosen_kp_pr = choosen_kp.copy()
            back_kp_pr = l_r[0][0].copy()
            
            choosen_kp_pr[0:68]*= 1/ratioWidth_ori
            choosen_kp_pr[68:]*= 1/ratioHeight_ori
            
            back_kp_pr[0:68]*= 1/ratioWidth_ori
            back_kp_pr[68:]*= 1/ratioHeight_ori
            
            
            #print('circling',choosen_kp_pr,back_kp_pr)
            for z22 in range(68) :
                    cv2.circle(img,(int(choosen_kp_pr[z22]),int(choosen_kp_pr[z22+68])),3,color)
                    #cv2.circle(img,(int(back_kp_pr[z22]),int(back_kp_pr[z22+68])),2,(0,255,255))
        
            #bbox = utils.get_bb(back_kp_pr[:68],back_kp_pr[68:])
            bbox = utils.get_bb(choosen_kp_pr[:68],choosen_kp_pr[68:])
            
            cv2.rectangle(img,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    colorBB, 2)
             
            runningCount+=1
            
        cv2.imshow('my webcam ', img)
        
        if writeVideo : 
            cap.write(img)
        
        #cv2.waitKey(1)
        
        if l_r is not None : 
            indexer +=1
        
        print("Lastt bb for next ",lp_BB)
        
            
        lastImage = cv2.resize(image,(imWidth,imHeight)).copy()
        
        
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
        
    cam.release()
    if writeVideo :
        cap.release()
    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=True)

if __name__ == '__main__':
    main()