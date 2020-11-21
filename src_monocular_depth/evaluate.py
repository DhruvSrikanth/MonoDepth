import os
import glob
import time
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

from tensorflow.keras.models import load_model
from loss import depth_loss_function
import model as depth_estimate_model
from utils import predict, load_images, display_images, evaluate
from matplotlib import pyplot as plt

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'depth_loss_function': depth_loss_function}

# Load model into GPU / CPU
print('Loading model...')
#model = load_model('/home/user01/storage/NYU Depth Analysis/src/models/1595602642-n25344-e25-bs2-lr0.0001-densedepth_nyu/model', custom_objects=custom_objects, compile=False)
model = depth_estimate_model.DepthEstimate()
model_weights = 'F:/Work/Work/Outdu Internship/nyu_depth_v2_dataset/src/models/1596012197-n25344-e25-bs2-lr0.0001-densedepth_nyu/weights.23-0.12.ckpt'
model.load_weights(model_weights)#, by_name = True, skip_mismatch = True)
print('Model weights loaded from path - ', model_weights)

# Load test data
print('Loading test data...', end='')
import numpy as np
from data import extract_zip
data = extract_zip('F:/Work/Work/Outdu Internship/nyu_depth_v2_dataset/nyu_test.zip')
from io import BytesIO
rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
depth = np.load(BytesIO(data['eigen_test_depth.npy']))
crop = np.load(BytesIO(data['eigen_test_crop.npy']))
print('Test data loaded.\n')

start = time.time()
print('Testing...')

e = evaluate(model, rgb, depth, crop, batch_size=6)

print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

end = time.time()
print('\nTest time', end-start, 's')


