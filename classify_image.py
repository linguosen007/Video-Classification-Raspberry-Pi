""" Load persistant model then start rawCapture stream, takes two images per
second from video port as bgr, Converts bgr raw image into numpy array.

Resoution has specific constraints: See camera setup section for info.
takes 
"""

import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import tensorflow as tf
import argparse
import os.path
import re

import numpy as np

FLAGS = None

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.relpath('imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.relpath('imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.
    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]

def run_classification():

    # In PiCamera, resolutions are treated as such:
    # There are three ports on the PiCamera. Video Port, still port, preview port.
    # There is a table of supported resolutions for each port.
    # The lowest supported resolution for the still port is full sensor size(3280x2464)
    # The lowest supported resolution for the video port is 640x480 at
    # framerates < 90. These are limitations on the frame size that can be sent
    # directly to the GPU. If a size is specified that is not directly supported,
    # the cameras resolution and frame rate are set to the closest supported resolution and
    # framerate, then are downscaled or upscaled to the specified resolution. 
    # The resize parameter uses the GPU to resize the image. 
    camera = PiCamera()
    camera.resolution = (640,480)
    
    # setting framerate below 2 negatively effects accuracy of prediction??
    ##camera.framerate = 10
    
    # capture image as an rgb array
    ##rawCapture = PiRGBArray(camera, size = (320,240))
    
    # camera warmup time
    time.sleep(2)
    
    
    node_lookup = NodeLookup()
    
    # load persistant model from pb
    with tf.gfile.FastGFile(os.path.relpath('classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    
    
    # Start tf session And capture stream continuously until program terminated.
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        # we set use_video_port as true to be able to capture 640X480 images then 
        # use gpu resizing in the pipeline to get 320X240
        for i in range(0,5):  
          camera.capture('/home/pi/Desktop/image.jpg',resize=(320,240))
          image = ('/home/pi/Desktop/image.jpg')
          image_data=tf.gfile.FastGFile(image, 'rb').read()
          start = time.time()
          # convert image to numpy array
          ##np_image = image.array
            
          # Make the prediction.            
          predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
            
          predictions = np.squeeze(predictions)
            
          # Sort list of predictions, only return the top prediction
          top = predictions.argsort()[-1:][::-1]
          end = time.time()-start
          # Lookup the human readable string with node_lookup
          for node_id in top:
              human_string = node_lookup.id_to_string(node_id)
              score = predictions[node_id]
              print('%s (score = %.5f)' % (human_string, score))
          print('time: ', end)
          # Truncate the buffer for next image/clear stream
            
if __name__ == '__main__':
    run_classification()
