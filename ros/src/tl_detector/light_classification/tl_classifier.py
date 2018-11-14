import os
import numpy as np
import rospkg
import tensorflow as tf
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        rospack = rospkg.RosPack()
        PATH_TO_MODEL = os.path.join(rospack.get_path('tl_detector'), 'light_classification/frozen_inference_graph.pb')
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
        red = len(classes[(classes == 1) & (scores >= 0.8)])
        yellow = len(classes[(classes == 2) & (scores >= 0.8)])
        green = len(classes[(classes == 3) & (scores >= 0.8)])
        cnts = np.array([red, yellow, green])
        idx = np.argmax(cnts)
        if cnts[idx] > 0:
            if idx == 0:
                return TrafficLight.RED
            elif idx == 1:
                return TrafficLight.YELLOW
            else:
                return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN
