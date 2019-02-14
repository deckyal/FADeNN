bb = [56, 56, 168, 168]
image_size = 224

image_directory = 'ex_images/'

cl_type = 1#1 is combine, 2 is half, 0 is general 
if cl_type > 0 : 
    useGeneral = False 
else : 
    useGeneral = True
    
showAllInter = True
img_name = 'indoor_120.png'
#listImage = [img_name]
trySynthetic = True

if not trySynthetic : 
    listImage = ['0030.jpg','0101.jpg','0412.jpg']
    model_directory = 'model/300VW/'
else : 
    model_directory = 'model/300WM/'
    listImage = [img_name,'bl_'+img_name,'lr_'+img_name,'no_'+img_name,'dr_'+img_name]