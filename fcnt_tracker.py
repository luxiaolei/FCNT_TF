"""
Main script for FCNT tracker. 
"""

## Define parameters for networks

sel_CNN_params = {}
SGNet_params = {}







## At t=0. Train all the networks, and fix parameters of sel-CNN net and GNet
## after trainning.  

# Get the first frame, and its associated target bbox.

# Pass image to vgg16 net, and retrives Conv4_3 and Conv5_3 layer.

# Performs sel-CNN trainning. 

# Pass selected features maps to GNet and SNet, performs initializations 
# for both networks.


## At t>0. Perform target localization and distracter detection at every frame,
## perform SNget adaptive update every 20 frames, perform SNet discrimtive 
## update if distracter detection return True.

# Forward pass an image through vgg16 and GNet.

# If t % 20 == 0, adaptive finetunes SNet.

# Performs distracter detecion operation, if detects distracters, then update 
# SNet with descrimtive method. 

# Localise the target using particle filter method.

# Draw bbox on image. And print associated IoU score.


