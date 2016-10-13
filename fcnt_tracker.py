"""
Main script for FCNT tracker. 
"""

## Define parameters for networks

sel_CNN_params = {}
SGNet_params = {}

## Define varies path
imgs_path = '..'
gt_path = '..'
vgg16_model_path = 'vgg16.tfmodel'








## At t=0. Train all the networks, and fix parameters of sel-CNN net and GNet
## after trainning.  

# Get the first frame, and its associated target bbox.
Inputs = InputProducer(imgs_path, gt_path) 
img, indx, bbox = Inputs.get_image()
assert indx == 0

# Pass image to vgg16 net, and retrives Conv4_3 and Conv5_3 layer.
with open(vgg16_model_path, mode='rb') as f:
    fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
images = tf.placeholder("float", [None, 224, 224, 3])
tf.import_graph_def(graph_def, input_map={ "images": images })
graph = tf.get_default_graph()
feed_dict = { images: img }
conv4_3 = graph.get_tensor_by_name('import/conv4_3/Conv2D:0')
conv5_3 = graph.get_tensor_by_name('import/conv5_3/Conv2D:0')
sess = tf.Session()
sess.run(tf.initialize_all_variables())
res4, res5 = sess.run([conv4_3, conv5_3], feed_dict=feed_dict)

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


