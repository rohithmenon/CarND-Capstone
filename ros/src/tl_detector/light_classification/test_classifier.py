import os
import numpy as np
import random
import tensorflow as tf
import time
from PIL import Image

class TLClassifier(object):
    def __init__(self):
        PATH_TO_MODEL = '/home/rohith.menon/models/research/object_detection/carnd/fine_tuned_model.ssd_inception_v2_coco/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(image, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        cls = classes[0, 0]
        if cls == 1:
            return "red"
        elif cls == 2:
            return "yellow"
        else:
            return "green"

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

classifier = TLClassifier()
IMG_PATH = '/home/rohith.menon/CarND-Capstone/ros/data/bag_and_sim'
all_paths = []
for i in range(3):
    path = os.path.join(IMG_PATH, str(i))
    all_paths.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith('jpg') and f.startswith('b')])
#all_paths = ['/tmp/test_imgs/{}.jpg'.format(i) for i in range(28)]

random.shuffle(all_paths)

for path in all_paths:
    image = Image.open(path)
    image_np = load_image_into_numpy_array(image)
    st = time.time() 
    pred = classifier.get_classification(image_np)
    elapsed = time.time() - st
    idx = int(path.split('/')[7])
    act = "unknown"
    if idx == 0:
        act = "red"
    elif idx == 1:
        act = "yellow"
    elif idx == 2:
        act = "green"
    else:
        act = "unknown"

    if act != pred:
        print('Pred: {}, Act: {} in {} sec'.format(pred, act, elapsed))
