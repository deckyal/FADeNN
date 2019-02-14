'''
Created on Nov 26, 2018

@author: deckyal
'''

from NetworkModels import FacialLocaliser, DAEE, LogisticRegression, DAEEH,\
    GeneralDAE
from utils import calcLandmarkError, get_bb
###########Network definitions############3

#Facial landmarker


import torch
import torchvision.utils as vutils
from torchvision.transforms import transforms
from utils import UnNormalize, unnormalizedAndLandmark, plotImages,read_kp_file,\
    unnormalizeToCV
import cv2

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 

image_size = 224


cl_type = 2#1 is combine, 2 is half, 0 is general 
model_dir = ['model/','model/']
sel_dir = 1;

#General Denoiser 

net = DAEE()
net.load_state_dict(torch.load(model_dir[1]+"AESE_WB_WE_3x3_224"+'.pt'))

print(torch.cuda.device_count)
print(torch.cuda.get_device_name(0))

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
    dircl1 = model_dir[sel_dir]+'combineAE.pt'
    dircl2 = model_dir[sel_dir]+'combineCL.pt'
    netAEC = DAEE()
    netAEC.load_state_dict(torch.load(dircl1))
    netAEC = netAEC.cuda()
    #netAEC.eval()
    
    model_lg = LogisticRegression(512, 5)
    model_lg.load_state_dict(torch.load(dircl2))
    model_lg = model_lg.cuda()
    #model_lg.eval()
else : 
    dircl1 = model_dir[sel_dir]+'combineAEH.pt'
    dircl2 = model_dir[sel_dir]+'combineCLH.pt'
    netAEC = DAEEH()
    netAEC.load_state_dict(torch.load(dircl1))
    netAEC = netAEC.cuda()
    #netAEC.eval()
    
    model_lg = LogisticRegression(512, 5)
    model_lg.load_state_dict(torch.load(dircl2))
    model_lg = model_lg.cuda()
    #model_lg.eval()

f = FacialLocaliser()

print(cl_type, dircl1,dircl2)
#Now the testing 

img_name = 'indoor_120'
kp_file ='ex_images/'+img_name+'.pts'
img_name += '.png'
#listImage = ['photo_gb1.jpg','photo_cs1.jpg','photo_bs1.jpg','photo_lr1.jpg']
listImage = ['bl_'+img_name,'lr_'+img_name,'no_'+img_name,'dr_'+img_name]
    
#listImage =  ['bl_indoor_001.png','lr_indoor_001.png','lr_indoor_104.png','bl_indoor_104.png']
#['bl_indoor_104.png','lr_indoor_001.png','no_indoor_104.png','dr_indoor_104.png']

tl = []
#for x,nt,npram in zip(listImage,listNTP,listNP) : 
for x in listImage : 
    tImageB = Image.open('ex_images/'+x)#.resize((image_size,image_size))
    #tImageB.show()
    
    #tImageB = generalNoise(tImageB,nt,npram)
    tImageB = transform(tImageB)
    tl.append(tImageB.unsqueeze(0))

li = torch.Tensor(len(tl),3,image_size,image_size)
torch.cat(tl, out=li)
li = li.to(device)    

'''img = li.cpu()
#theRest = cv2.cvtColor(t_res.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
plt.axis("off")
plt.title("Trainings Images")
plt.imshow(np.transpose(vutils.make_grid(img, padding=2, normalize=True).cpu(),(1,2,0)))        
#plt.show()'''

tl = []

ldmarkOri = []
ldmarkDe= []

#Image denoising 
#y2 = [2,1,3,0]
for i,n_imgs in enumerate(li) :
    if cl_type == 1 : 
        recon_batch,xe = netAEC(n_imgs.unsqueeze(0))
    else :  
        xe = netAEC(n_imgs.unsqueeze(0))
    #print(xe)
    labels = model_lg(xe)
    #labels = F.log_softmax(labels,1)
    x, y = torch.max(labels, 1)
    print('x',x,'y',y,'labels',labels)
    res = GD.forward(n_imgs.unsqueeze(0), y[0])
    tl.append(res)

res_experts = torch.Tensor(len(tl),3,image_size,image_size)
torch.cat(tl, out=res_experts)


res_general  = net.forward(li)[0].detach().cpu()
cvImageOri= []
cvImageDe = []


showIntermediate = True

if showIntermediate :    
    plt.figure()
    plt.title("Original")
    plt.imshow(np.transpose(vutils.make_grid(li.to(device), padding=2, normalize=True).cpu(),(1,2,0)))
    
    #theRest = cv2.cvtColor(t_res.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
    plt.figure()
    plt.title("Intermediate General")
    plt.imshow(np.transpose(vutils.make_grid(res_general.detach().to(device), padding=2, normalize=True).cpu(),(1,2,0)))
    
    plt.figure()
    #theRest = cv2.cvtColor(t_res.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
    plt.title("Intermediate Experts")
    plt.imshow(np.transpose(vutils.make_grid(res_experts.detach().to(device), padding=2, normalize=True).cpu(),(1,2,0)))
    #plt.show()


#Now landmarking
unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

gt =np.array(read_kp_file(kp_file)).flatten('F')

bb = get_bb(gt[:68],gt[68:])


useGeneral = False 

if useGeneral : 
    resTemp = res_general.cpu() 
else : 
    resTemp = res_experts.cpu()
    
for img_ori,img_de in zip(li.cpu(),resTemp) : 
    #first unnormalize the image 
    #Unnormalized it, convert to numpy and multiple by 255.
    
    #print(img_ori.shape,img_de.shape)
     
    img_orix = unorm(img_ori.clone()).numpy()*255 
    img_dex = unorm(img_de.clone()).numpy()*255
    
    
    img_orix = img_orix.transpose((1,2,0))
    img_dex = img_dex.transpose((1,2,0))
    
    #print(img_ori.shape,img_de.shape)
    #Then transpose to be height,width,channel, to Int and BGR formate 
    img_orix = cv2.cvtColor(img_orix.astype(np.uint8 ),cv2.COLOR_RGB2BGR)
    img_dex = cv2.cvtColor(img_dex.astype(np.uint8 ),cv2.COLOR_RGB2BGR)
    
    #add to the list
    cvImageOri.append(img_orix)
    cvImageDe.append(img_dex)
    
    #Now get the landmark 
    resOri = f.forward(img_orix,bb = bb)
    resDe = f.forward(img_dex,bb = bb)
    
    #add to the list 
    ldmarkOri.append(resOri)
    ldmarkDe.append(resDe)

ldmarkOri = np.asarray(ldmarkOri)
ldmarkDe = np.asarray(ldmarkDe)

gt =np.array(read_kp_file(kp_file)).flatten('F')
 
#now landmark on normalized image 
'''imgOriLdmrk,err_ori = unnormalizedAndLandmark(li.cpu(),ldmarkOri,gt,cv = False)
imgDeLdmrk,err_de = unnormalizedAndLandmark(res_experts.cpu(),ldmarkDe,gt, cv = False)'''
imgOriLdmrk,err_ori = unnormalizedAndLandmark(li.cpu(),ldmarkOri,gt,cv = False)
imgDeLdmrk,err_de = unnormalizedAndLandmark(li.cpu(),ldmarkDe,gt, cv = False)
interImages = unnormalizeToCV(resTemp)

#print(err_ori, err_de)

add = "-g" if useGeneral  else ""

for img_name2,img in zip(listImage,imgOriLdmrk): 
    cv2.imwrite(img_name2+add+'.png',img)
    
for img_name2,img,inter in zip(listImage,imgDeLdmrk,interImages): 
    cv2.imwrite(img_name2+'de_'+add+'.png',img) 
    cv2.imwrite(img_name2+'inter_'+add+'.png',inter)
#exit(0)
#show it

oriName = "ori-g2.png" if useGeneral  else "ori2.png"
afterName = "after-g2.png" if useGeneral  else "after2.png"
trueName = "trueImage-g2.png"  if useGeneral  else "trueImage2.png"
 
plotImages(imgOriLdmrk,err_ori,fileName=oriName,show=False)
plotImages(imgDeLdmrk,title=err_de,fileName=afterName,show=False)

dirOImage = 'ex_images/'+img_name
print(dirOImage)

if useGeneral :
    tImageB = Image.open(dirOImage)#.resize((image_size,image_size))
    tImageB = transform(tImageB)
    tImageB = tImageB.unsqueeze(0)
    
    tImageB = tImageB.to(device)    
    res_ib  = net.forward(tImageB)[0].detach().cpu().squeeze()
    
    print(res_ib.shape)
    
    res_ib = unorm(res_ib.clone()).numpy()*255
    res_ib = res_ib.transpose((1,2,0)) 
    trueImageTemp = cv2.cvtColor(res_ib.astype(np.uint8 ),cv2.COLOR_RGB2BGR)
    #trueImageTemp =cv2.imread(dirOImage)
    
trueImage = cv2.imread(dirOImage)
cv2.imshow('tset',trueImage)
cv2.waitKey(0)

if useGeneral : 
    prLdmrk = f.forward(trueImageTemp,bb = bb)
else : 
    prLdmrk = f.forward(trueImage,bb = bb)
#landmark 
for y in range(68) : 
    cv2.circle(trueImage,(int(prLdmrk[y]),int(prLdmrk[y+68])),3,(255,0,0),-1 )
    cv2.circle(trueImage,(int(gt[y]),int(gt[y+68])),3,(0,0,255),-1 )
cv2.imwrite(trueName,trueImage)
if useGeneral : 
    cv2.imwrite(trueName+'-g.png',trueImageTemp)

err = calcLandmarkError(prLdmrk,gt)
print('true err : ',err)
