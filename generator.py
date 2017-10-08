import tensorflow as tf
import json
from object_detection.utils import dataset_util
import os
import io
import pandas as pd

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict



# flags = tf.app.flags
# flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
# FLAGS = flags.FLAGS

Dir="/Volumes/wd/Data/Test2/"

train=False

if train:
  FileName='train'   #train
  subpath='Rec/'
  jsonName='tag.json'
  recordNum=10000
else:
  FileName='test'   #test
  subpath='TestRec/'
  jsonName='TestTag.json'
  recordNum=300


#/Volumes/LaCie/train/data/TFRecord
def create_tf_example(example,path):
  # TODO(user): Populate the following variables from your example.
  height = 100 # Image height
  width = 100 # Image width
  filename = example["filename"].decode('string_escape')   # Filename of the image. Empty if image is not from file
  
  with tf.gfile.GFile(os.path.join(path, '{}'.format(filename)), 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = Image.open(encoded_jpg_io)

  print(filename)
  image_format = b'jpeg' # b'jpeg' or b'png'

  xmins = [example["xmins"]] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [example["xmaxs"]] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [example["ymins"]] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [example["ymaxs"]] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = ['cir'] # List of string class name of bounding box (1 per box)
  classes = [1] # List of integer class id of bounding box (1 per box)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_example

def readF():
  f = open(Dir+"tag.json",'r')  
  setting = json.load(f)
  # family = setting['BaseSettings']['size']  
  # size = setting['fontSize']   
  return setting

def main(_):
  writer = tf.python_io.TFRecordWriter(Dir+"record/"+FileName+".tfrecords")
  path = os.path.abspath(Dir+subpath)#image path
  # TODO(user): Write code to read in your dataset to examples variable
  setting=readF()
  for i in range(0,recordNum):
    tf_example = create_tf_example(setting[str(i)+"_jpg"],path)
    print(setting[str(i)+"_jpg"])
    writer.write(tf_example.SerializeToString())

  writer.close()


if __name__ == '__main__':
  tf.app.run()