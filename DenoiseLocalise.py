'''
Created on Nov 26, 2018

@author: deckyal
'''

from NetworkModels import FacialLocaliser, DAEE, LogisticRegression, DAEEH,\
    GeneralDAE

import torch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    
import torchvision.utils as vutils
from torchvision.transforms import transforms
from utils import UnNormalize, unnormalizedAndLandmark,unnormalizeToCV
import cv2

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 


from config import *

#General Denoiser 
net = DAEE()
net.load_state_dict(torch.load(model_directory+"AESE_WB_WE_3x3_224"+'.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
net.eval()

#Multiple expert denoiser 

GD = GeneralDAE()

transform = transforms.Compose([
    transforms.Resize((image_size,image_size)),
    transforms.ToTensor(), 
    transforms.Normalize(mean = (.5,.5,.5), std = (.5,.5,.5))
    ])

if cl_type == 1 : 
    dircl1 = model_directory+'combineAE.pt'
    dircl2 = model_directory+'combineCL.pt'
    netAEC = DAEE()
    netAEC.load_state_dict(torch.load(dircl1))
    netAEC = netAEC.cuda()
    #netAEC.eval()
    
    model_lg = LogisticRegression(512, 5)
    model_lg.load_state_dict(torch.load(dircl2))
    model_lg = model_lg.cuda()
    #model_lg.eval()
else : 
    dircl1 = model_directory+'combineAEH.pt'
    dircl2 = model_directory+'combineCLH.pt'
    netAEC = DAEEH()
    netAEC.load_state_dict(torch.load(dircl1))
    netAEC = netAEC.cuda()
    #netAEC.eval()
    
    model_lg = LogisticRegression(512, 5)
    model_lg.load_state_dict(torch.load(dircl2))
    model_lg = model_lg.cuda()
    #model_lg.eval()

f = FacialLocaliser()
    
tl = [] 
for x in listImage : 
    tImageB = Image.open(image_directory+x).resize((image_size,image_size))
    tImageB = transform(tImageB)
    tl.append(tImageB.unsqueeze(0))

li = torch.Tensor(len(tl),3,image_size,image_size)
torch.cat(tl, out=li)
li = li.to(device)    

tl = []

ldmarkDe= []

#Image denoising 
for i,n_imgs in enumerate(li) :
    if cl_type == 1 : 
        recon_batch,xe = netAEC(n_imgs.unsqueeze(0))
    else :  
        xe = netAEC(n_imgs.unsqueeze(0))
    labels = model_lg(xe)
    x, y = torch.max(labels, 1)
    print('x',x,'y',y,'labels',labels)
    res = GD.forward(n_imgs.unsqueeze(0), y[0])
    tl.append(res)

res_experts = torch.Tensor(len(tl),3,image_size,image_size)
torch.cat(tl, out=res_experts)

res_general  = net.forward(li)[0].detach().cpu()
cvImageDe = []

if showAllInter :    
    plt.figure()
    #theRest = cv2.cvtColor(t_res.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
    plt.title("Intermediate Experts")
    plt.imshow(np.transpose(vutils.make_grid(res_experts.detach().to(device), padding=2, normalize=True).cpu(),(1,2,0)))
    
    #theRest = cv2.cvtColor(t_res.transpose((1,2,0)).astype(np.uint8 ),cv2.COLOR_RGB2BGR)
    plt.figure()
    plt.title("Intermediate Direct")
    plt.imshow(np.transpose(vutils.make_grid(res_general.detach().to(device), padding=2, normalize=True).cpu(),(1,2,0)))
    
    
    plt.figure()
    plt.title("Original")
    plt.imshow(np.transpose(vutils.make_grid(li.to(device), padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()

#Now landmarking
unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

if useGeneral : 
    resTemp = res_general.cpu() 
else : 
    resTemp = res_experts.cpu()
    
for img_ori,img_de in zip(li.cpu(),resTemp) : 
     
    img_dex = unorm(img_de.clone()).numpy()*255
    img_dex = img_dex.transpose((1,2,0))
    img_dex = cv2.cvtColor(img_dex.astype(np.uint8 ),cv2.COLOR_RGB2BGR)
    cvImageDe.append(img_dex)
    resDe = f.forward(img_dex,bb = bb)
    ldmarkDe.append(resDe)

ldmarkDe = np.asarray(ldmarkDe)
imgDeLdmrk= unnormalizedAndLandmark(li.cpu(),ldmarkDe, cv = False)
interImages = unnormalizeToCV(resTemp)

add = "-G-" if useGeneral  else "-SM-"

for img_name2,img,inter in zip(listImage,imgDeLdmrk,interImages): 
    cv2.imwrite(img_name2+'-withDenoising_'+add+'.png',img) 
    cv2.imwrite(img_name2+'-Intermediate_'+add+'.png',inter)