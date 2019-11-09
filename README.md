# Signature detection using Tensorflow API

* Clone the [tensorflow model](https://github.com/tensorflow/models)

* Download [protocolbuffer](https://github.com/protocolbuffers/protobuf/releases)

* Install dependency package

```
pip install tensorflow
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib
```

* Install [cocoapi](https://github.com/philferriere/cocoapi)

* Extract python file from proto file

     official guide: `./bin/protoc object_detection/protos/*.proto --python_out=.`
     
     use python code: 
     
     enter the research folder, log into the console:
        
      `python use_protobuf.py  Example: python use_protobuf.py object_detection/protos C:/Users/Gilbert/Downloads/bin/protoc`

* Add the paths to enviroment variables in the console:

  ```
  set PYTHONPATH=%PYTHONPATH%;<PATH_TO_TF>/TensorFlow/models/research
  set PYTHONPATH=%PYTHONPATH%;<PATH_TO_TF>/TensorFlow/models/research/slim
  ````
  
* Run python file from TensorFlow/models/research/:
  ```
  python setup.py build
  python setup.py install

  ```

* Test the installation:
  ```
  python object_detection/builders/model_builder_test.py
  ```

* Create images folder in objection_detection folder, create train, test folder inside images folder, put train image and test image separately
  
* Resize the training images, create a console in the folder which contains the `image` folder:
  ```
  python transform_image_resolution.py -d images/ -s 800 600
  ```
  
* [label the train image add bounding box](https://github.com/tzutalin/labelImg)

* download [xml_to_csv.py and generate_tf_record.py](https://github.com/datitran/raccoon_dataset) into objection_detection folder
  Adjust xml_to_csv.py, open a console in objection_detection: `python xml_to_csv.py`:
  ```python
  # Old:
  def main():
      image_path = os.path.join(os.getcwd(), 'annotations')
      xml_df = xml_to_csv(image_path)
      xml_df.to_csv('raccoon_labels.csv', index=None)
      print('Successfully converted xml to csv.')
  # New:
  def main():
      for folder in ['train', 'test']:
          image_path = os.path.join(os.getcwd(), ('images/' + folder))
          xml_df = xml_to_csv(image_path)
          xml_df.to_csv(('images/'+folder+'_labels.csv'), index=None)
          print('Successfully converted xml to csv.')

  ```
  Adjust generate_tf_record.py, open a console in objection_detection: 
  ```
  python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record`
  python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
  ```
  ```python
  # TO-DO replace this with label map
  def class_text_to_int(row_label):
      if row_label == 'signature':
          return 1
      elif row_label == 'others':
          return 2
      else:
          return None

  ```

* create a folder called training inside objectiong_detection folder, create a file called labelmap.pbtxt inside training folder:
  ```
  item {
    id: 1
    name: 'signature'
  }
  item {
      id: 2
      name: 'others'
  }

  ```
* Download [faster_rcnn_inception_v2_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) into objection_detection folder, extract the zipped file into current directory

* Copy faster_rcnn_inception_v2_pets.config in objection_detection/samples/config to training folder, adjut the config file:
  ```
  Line 9: change the number of classes to number of objects you want to detect (2 in my case)
  Line 106: change fine_tune_checkpoint to the path of the model.ckpt file:
  fine_tune_checkpoint: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"

  Line 123: change input_path to the path of the train.records file:
  input_path: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/train.record"

  Line 135: change input_path to the path of the test.records file:
  input_path: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/test.record"

  Line 125–137: change label_map_path to the path of the label map:
  label_map_path: "C:/Users/Gilbert/Downloads/Other/models/research/object_detection/training/labelmap.pbtxt"
  Line 130: change num_example to the number of images in your test folder.

  ```

* Finally train the model, to train the model we will use the train.py file, which is located in the object_detection/legacy folder. We will copy it into the object_detection folder and then we will open a console and type:
  ```
  python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
  ```
  
* Monitor the results: 

  About every 5 minutes the current loss gets logged to Tensorboard. We can open Tensorboard by opening a second console, navigating to the object_detection folder and typing:
  `tensorboard --logdir=training`
 
* Now that we have a trained model we need to generate an inference graph, which can be used to run the model. For doing so we need to first of find out the highest saved step number. For this, we need to navigate to the training directory and look for the model.ckpt file with the biggest index. Then we can create the inference graph by typing the following command in the console:
  ```
  python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
  XXXX represents the highest number.
  ```
  
* test the model: open object_detection_tutorial.ipynb inside objection_detection folder
We only need to replace the fourth code cell:
  ```
  From:
  # What model to download.
  MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
  MODEL_FILE = MODEL_NAME + '.tar.gz'
  DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
  # List of the strings that are used to add a correct label for each box.
  PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

  To:
  MODEL_NAME = 'inference_graph'
  PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
  PATH_TO_LABELS = 'training/labelmap.pbtxt'


  
  ```
  
# Example:
![Image of Yaktocat](https://github.com/chunlei2/objectDetection/blob/master/example.png)
  
# References:
https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85

https://gilberttanner.com/blog/live-object-detection

https://gilberttanner.com/blog/installing-the-tensorflow-object-detection-api

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

https://www.microsoft.com/developerblog/2018/05/07/handwriting-detection-and-recognition-in-scanned-documents-using-azure-ml-package-computer-vision-azure-cognitive-services-ocr/

source of image:

https://www.gsa.gov/real-estate/real-estate-services/leasing-policy-procedures/lease-documents




