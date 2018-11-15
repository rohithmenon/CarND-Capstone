import json
import os
import re
import tensorflow as tf
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS

CLASS_TEXT = {
    0: 'red',
    1: 'yellow',
    2: 'green'
}

def create_tf_example(label_and_data_info):
  # TODO START: Populate the following variables from your example.
  height = 600 # Image height
  width = 800 # Image width
  filename = os.path.join('/home/rohith.menon/CarND-Capstone', label_and_data_info['path']) # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  with tf.gfile.GFile(filename, 'rb') as fid:
    encoded_image_data = fid.read()
  image_format = 'jpeg' # b'jpeg' or b'png'
  boxes = label_and_data_info['boxes']
  num_boxes = len(boxes)

  xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  for box in boxes:
      ymin, xmin, ymax, xmax = box
      xmins.append(xmin)
      xmaxs.append(xmax)
      ymins.append(ymin)
      ymaxs.append(ymax)
  
  cls = int(label_and_data_info['path'].split('/')[3])
  classes_text = [CLASS_TEXT[cls].encode('utf8')] * num_boxes # List of string class name of bounding box (1 per box)
  classes = [cls + 1] * num_boxes # List of integer class id of bounding box (1 per box)
  # TODO END
  tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_label_and_data

def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  file_loc = '../../data/label_data.json'
  all_data_and_label_info = None
  with open(file_loc) as f:
    all_data_and_label_info = json.load(f)

  for data_and_label_info in all_data_and_label_info:
    tf_example = create_tf_example(data_and_label_info)
    writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  tf.app.run()
