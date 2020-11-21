import os
import glob
import argparse
import matplotlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

import model as depth_estimate_model
from tensorflow.keras.models import load_model
from utils import predict, load_images, display_images,save_images
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = depth_estimate_model.DepthEstimate()
model_weights = 'F:/Work/Work/Outdu Internship/nyu_depth_v2_dataset/src/models/1596012197-n25344-e25-bs2-lr0.0001-densedepth_nyu/weights.23-0.12.ckpt'
model.load_weights(model_weights)
print('Model weights loaded from path - ', model_weights)

# Input images
input_path = 'F:/Work/Work/Outdu Internship/nyu_depth_v2_dataset/test_image/*.png'
inputs = load_images( glob.glob(input_path) )
print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

# Compute results
outputs = predict(model, inputs)

#Save results
save_images('test_depth.png',outputs.copy(),is_rescale=False) #set is_rescale to True to save heat map


#matplotlib problem on ubuntu terminal fix
#matplotlib.use('TkAgg')   

# Display results
viz = display_images(outputs.copy(), inputs.copy())
plt.figure(figsize=(10,5))
plt.imshow(viz)
plt.savefig('test.png')
plt.show()
