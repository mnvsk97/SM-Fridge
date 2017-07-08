
Sai krishna M.N.V
Raviteja R.

# Problem : Image classification and Detection. 
 Initially, when we started out to solve the classification problem, we tried to classify images using features like colour,shape,texture etc. The results obtained after implementing  various image processing algorithms were not consistent. And moreover the features we used to distinguish objects in a given image were not reliable. The main problem with the usage of conventional image processing techniques was that the number of features might vary and features used for classification might extract vary. After going through some research papers on image classification and tech articles, we came to know of the usage of neural networks in classifying images. We made use of a neural network framework called tensor flow, an open source project by google. Keeping in mind the huge community support available, tensor flow seemed to be the right platform to build out program. Tensor Flow's python API is powerful and easy to understand. We followed the official documentation and built an image classifier. To get started with tensor flow, tutorials are available at https://www.tensorflow.org/tutorials/  . Out of all the neural network architectures, Convolutional neural networks (CNNs for short) proved to be the best in industry when it comes to classification problems. To know more about how convolutional neural networks work, we have added some links in the end. A decent understanding of CNNs is necessary to proceed any further. Alex Krizhevsky was the first person to make use of CNNs in an image classification competition. He achieved ground-breaking results by using CNNs. Refer http://www.image-net.org/challenges/LSVRC/2017/index.php for more details about the above mentioned competition. We made use of Inception v3 model, a pre-trained model made open-source by google. This layer has been trained on about 1000 classes of images available at www.image-net.org/. Images required for training can be downloaded from this website. After training the inception model with our own set of images, the classifier seemed to give decent results.
 
We will walk you through the installation and training process in the below sections. We assume that you are using a linux distribution.
## Installation :
<br>Clone this repository : </br>
```
$ git clone https://github.com/mnvsk97/SM-fridge.git
```
Image classifier has been written in python using tensor flow framework. Refer https://www.tensorflow.org/ for more details on tensorflow.To install tensor flow use the following commands:
```
$ sudo pip uninstall tensorflow  # for Python 2.7
$ sudo pip3 uninstall tensorflow # for Python 3.x
 ``` 
```
Note: We assume that pip is already installed. If the local machine has external gpu support, tensor flow can be installed for gpu using $ sudo pip install tensorflow-gpu. Advantages of using a gpu are discussed in the later sections. If messages like “____ has not been installed” popup in your terminal while running any kind of commands, please try to google them.
``` 

Some of the python scripts in the repository make use of OpenCV to read and write images as numpy arrays. Follow the installation tutorial from http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/.
 
### Training a pre-trained model on custom dataset :
In tensorflow's official documentation, a brief explanation of how to train a pretrained model with our own dataset. Follow the instructions from https://www.tensorflow.org/tutorials/image_retraining. If there is any kind of problem when you follow the instructions, we will walk you through the training process in the following section. The next section can be skipped otherwise.The following sections assume that python is already installed in your computer.
 
1. Start by cloning tensor flow repository by running the following command :
```
$ git clone https://github.com/tensorflow/models
```
This is the official tensorflow repository. Many pre-trained models are available here. We will proceed with Inception V3 model.
2. The repository we just cloned has a python script which downloads a pretrained model which we use to train on our own data. The python script also downloads an example image and classifies it. Execute the following commands to run this script.
```
$ cd models/tutorials/image/imagenet
$ python classify_image.py
```
If everything goes well, you will see the following output in your terminal:

The format of the output can be understood as follows:

giant panda, panda, .....  = class or category predicted by the algorithm.
(score = ....)  = the confidence (lies between 0 and 1) of algorithm  

3. Google make use of a tool called BAZEL to complete the build process.bazel can be installed by following steps :
```
$ sudo add-apt-repository ppa:webupd8team/java
$ sudo apt-get update
$ sudo apt-get install oracle-java8-installer
$ echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
$ curl https://bazel.build/bazel-release.pub...
$ sudo apt-get update && sudo apt-get install bazel
$ sudo apt-get upgrade bazel
```
These steps install the latest version of bazel in you computer.
4.Before building the trained model, we need to configure tensor flow from previously cloned tensor flow directory . To do so, run the following commands:
```
$ cd
$ cd tensorflow
$ sudo ./configure
```
After you run the last line, the terminal prompts you to select settings with which you want to configure tensor flow. Some of them may slow down the configuration and training process. It is suggested to follow the below configuration for minimal load on the processor.
• Python location: Default-enter
• Jemalloc: Yes
• Google Cloud: No
• Hadoop: No
• XLA: No
• Python library: Default-enter
• OpenCL: No
• Cuda: No
The configuration process may take a while. Use google to solve any kind of errors.
5. When we try to build programs using bazel, it looks for a workspace. Basically a workspace is a working directory which contains all the files to be built. To create a workspace in the working directory, run
```
$ touch WORKSPACE
```
6.Build the model by running the following commands.
```
$ sudo bazel build tensorflow/examples/image_retraining:retrain
$ sudo bazel build --config opt tensorflow/examples/image_retraining:retrain
```
These commands might take a  while depending on RAM and processor of your computer.
7. Now the model is ready for retraining. A good understanding of how the retraining process works is necessary. The retraining process uses a concept called Transfer learning. To know more about transfer learning go through http://cs231n.github.io/transfer-learning/ .
8. To test the model on example dataset, tensor flow's documentations provides us with a dataset of flowers having four classes. Run the following commands to download and prepare the dataset.
```
$ cd
$ curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
$ tar xzf flower_photos.tgz
```
To get a better understanding of the dataset, go ahead and open the flowers_photos folder and see how the images are arranged.
9. Now that the dataset is ready, we can train the model by running the following command:
```
$ sudo bazel-bin/tensorflow/examples/image_retraining/retrain --image_dir path/to/flower/photos
```
The --image_dir parameter points to the previously downloaded flowers data. You need to replace path/to/flower/photos with the path to flower photos directory. This parameter tells retrain.py script to train the model on this dataset.
Now we are done with training the inception model. Two files with extension .pb and .txt are generated after the training process is completed. When we train a model, the each extracted feature is in the form of a numpy array. There arrays are known as weights. When a image is passed as an argument to the model during training process, these weights are updated accordingly. When we used label_image script to make a prediction on a new image(similar to the previous commands), the script makes use of these weights and make a prediction accordingly. These weights are saved by tensor flow in a protobuf(.pb) file. This file has a binary representation of weights of the entire network. The second(.txt) file contains the classes with which we trained our model.
10. After the training process is done, we need to test the model for correctness. The cloned repository contains a python script to label a given image by using our trained model. To make use of it, we need to build this python program using bazel and pass the image you want to classify as a parameter using --image.
```
$ sudo bazel build tensorflow/examples/label_image:label_image && \
$ sudo bazel-bin/tensorflow/examples/label_image/label_image \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--output_layer=final_result \
--image= path/to your/test_image.jpg
```
--graph loads the .pb file(weights file) generated after training
--labels loads the text file with our classes
--output points to the folder where we get our output
replace --image= with your test image path.
The above commands build the labe_image.py script and display the predictions made on the input image. The format of outputs has already been discussed in step 2.
 
We trained our model with four classes : carrot,brinjal,tomato,potato. When the model makes a prediction, the output will be any of the four classes with different confidence scores. The higher the confidence, higher is the reliability factor. Training accuracy and validation accuracy can be found during the training process. By analysing these accuracy values, one can get an idea of how well the training is being done. Accuracy depends on the different factors like architecture of CNN, number of layers, number of neurons in a layer and the quality of data.    
## Additional notes on this classifier:
In the above sections, we trained a pretrained inception v3 model with our own dataset. Our classifier is very reliable when it comes to single class detection. If we want the algorithm to tell us about all the objects present in a given image, it may fail. 
So its safe to say that the above implemented classifier is useful only when we want it to recognize a single object. Therefore what we have done is a single-label classifier. In a fridge there might be many objects arranged in different ways. Using this model for detection leads to the model unable to identify other objects. We need the classifier to locate all the objects in a given image. One approach to achieve multiple object detection is discussed below.
The training data has two types of files. One type is the images with which we train our model and the other tye of files is the annotations i.e.,what does the image represent. For example, if there is an image named tomato.jpg , there exists a tomato.jpg.txt file with tomato written in it. This text file is the annotation of tomato.jpg . Instead of taking images which contain only tomatoes, we can collect images with other objects as well and annotate the text files with multiple classes. For example, image with potato and tomato has a corresponding text file with both classes in it. This way we can achieve multiple label classification. Read https://medium.com/towards-data-science/multi-label-image-classification-with-inception-net-cbb2ee538e30 to know more about this approach.This method works well enough but the only problem is the collection of images with multiple objects. You can try the approach mentioned in the above blog.
Rather than looking for images with multiple objects, we can localise an object in a given image. For a given image we need to tell the algorithm about the location of an object relative to the image. This method is discussed with an example image below :
 
 This approach is called object localisation. There are many techniques to find where a particular object is located. We implemented the above technique using Selective search. To know more about selective search,go through https://www.koen.me/research/pub/vandesande-iccv2011.pdf. Another technique called Fast RCNN uses a combination of Recurrent neural networks and CNNs which uses object localisation for  detection. A very detailed explanation and implementation of the latter can be found at https://github.com/rbgirshick/fast-rcnn. Microsoft CNTK, a framework similar to tensor flow makes use of fast RCNN for object detection. A brief explanation of how to use cntk has been discussed in the later sections.
When we were exploring for RCNN implementations, we came across a research paper on a neural network called YOLO, which was written using a framework called Darknet. You can read about Yolo in https://arxiv.org/abs/1612.08242. The author of this network explained that YOLO works about 100x faster than fast RCNN. A detailed implementation and installation process can be found here: https://pjreddie.com/darknet/yolo/ . These instructions can be followed with ease. If it's not so clear, you can follow the below steps on how to use YOLO.
 
Before proceeding any further, a good understanding of YOLO neural network is required. The link which we provided for YOLO has a detailed explanation about how yolo works in the initial sections of the website.
1. We start by cloning the yolo repository. A link to this repository is provided in the above mentioned website.
```
$ git clone https://github.com/pjreddie/darknet
```
2. After cloning the repo, go to darknet directory. You can find a make file with which we compile yolo. This file contains the configuration with which we compile yolo network. MakeFile can be manipulated according to our requirements which is discussed in later sections. Run the following commands to configure YOLO with default setup.
```
$  cd darknet
$  make
```
3. The author trained YOLO on two different datasets. One is Pascal VOC which consists of 20 classes and the other is microsoft COCO dataset which consists of 80 classes. The weights obtained after training yolo are available in the website. To download weights, run the following command:
```
$ wget https://pjreddie.com/media/files/yolo.weights
```
4. In darknet directory, there is a directory called cfg which consists of variety of configurations depending on number of convolution layers, filters and classes. Run the following command from darknet directory.
```
$ ./darknet detect cfg/yolo.cfg yolo.weights data/dog.jpg
```
The above command can be broken down as follow:
./darknet = this is a call made to darknet.exe file
detect = this is a function used to detect objects in a given image.alternatively we can use detector test.
cfg/yolo.cfg = this is a path to yolo.cfg file. This file contains a configuration with which we would like to run our network. Many other .cfg files can be found in cfg directory.
yolo.weights = this is a weight file which we downloaded in step3. Replace this with path to previously downloaded weights file.
data/dog.jpg = this is our test image. Author has provided some sample images in ‘data’ directory. This can be replaced with your own image path.
If everything runs perfectly, the command produces output of the following type. 
The above command draws bounding boxes on our input image. This image is saved in darknet folder as predictions.png .
Note: the command in step 4 takes time depending on the processor and RAM configurations. To avoid this time, a tiny version is available. This can be thought of as a lightweight version which takes lesser time but with less accuracy. To use tiny version, download respective weights and change configuration file as follows:
```
$ wget https://pjreddie.com/media/files/tiny-yolo-voc.weights
$ ./darknet detector test cfg/voc.data cfg/tiny-yolo-voc.cfg tiny-yolo-voc.weights data/dog.jpg
```
5. To run this detector real-time using a webcam, we need to change the make file accordingly. It is assumed that latest version of opencv is already installed during tensor flow installation. Open the MakeFile and set OPENCV=1. Open terminal from darknet directory and run “make” to compile with this configuration. A demo file is provided to run run-time detection and can be run using the following command:
```
$ ./darknet detector demo cfg/coco.data cfg/yolo.cfg yolo.weights 
```
With this step we are done with configuring YOLO. Test yolo with different images to analyse how bounding boxes are drawn.
Go through the section where the author explains how to train the network on custom dataset. To get used to training process, go through the training process by using VOC dataset. A step by step process has been illustrated in the website . Otherwise run the following commands to download the dataset:
```
$ curl -O https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
$ curl -O https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
$ curl -O https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
$ tar xf VOCtrainval_11-May-2012.tar
$ tar xf VOCtrainval_06-Nov-2007.tar
$ tar xf VOCtest_06-Nov-2007.tar
```
Note : VOC dataset is about 2gb in size. It might take a while for the download to complete.
Format of annotations: In VOC dataset, the annotations are in the form of xml files. This format will be discussed later. Darknet requires the annotation files to be of the following format : <object-class> <x> <y> <width> <height>. There is a python script available in the website which converts the annotation files to darknet format, creates a train.txt file with all images to be used for training and a test.txt file with which the trained network will be tested. Run the following commands after downloading the vOC dataset:
```
$ curl -O https://pjreddie.com/media/files/voc_label.py
$ python voc_label.py  # python3 voc_label.py for 3.5 version
```
It is to be noted that these command should be run from the darknet directory. Whenever in doubt, please refer to the YOLO website.
After converting the annotations to darknet format, we need to create a train.txt file on which we train our network. Run the following command:
```
$ cat 2007_train.txt 2007_val.txt 2012_*.txt > train.txt
```
Now go to your Darknet directory. We have to change the cfg/voc.data config file to point to your data:
  1 classes= 20
  2 train  = <path-to-voc>/train.txt
  3 valid  = <path-to-voc>2007_test.txt
  4 names = data/voc.names
  5 backup = backup
We will use a weights files pre trained on imagenet available in
```
$ curl -O https://pjreddie.com/media/files/darknet19_448.conv.23
```
Run the following command to start training yolo on VOC data.
```
$ ./darknet detector train cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
```
Now that we are through with how to train yolo on a dataset, we proceed in a similar manner with our own dataset. The only problem is creating a dataset. Images can be downloaded from Imagenet. Annotations can be created using any of the following tools:
1. labelImg - https://github.com/tzutalin/labelImg
2. Bbox-Label-Tool - https://github.com/puzzledqs/BBox-Label-Tool
Any one of the above mentioned tools can be used to create annotations depending on the format requirements. Go through the links provided alongside to understand how to use these tools.
Manipulations :
When we run the detect function, it creates a predictions.png file with bounding boxes drawn on the detected objects. The coordinates of these bounding boxes can be found by manipulating a file called image.c in ‘src’ folder. This file is a c program which is responsible for drawing bounding boxes around the detected objects. A modified image.c file is available in the following link:http://bit.ly/2uNZEAm . Download this file and  to ‘src’ folder in darknet directory. Replace the original image.c file with the downloaded file. Come back to darknet directory and open terminal and run ‘make’ command to compile yolo with the new changes made. After this modification, whenever you run the detect function, a new file called box_cord.txt is created containing bounding box coordinates drawn in predictions.jpg image. These coordinates are used in later stages.
Before going any further, we need to know the classes with which we trained YOLO. For example, when we train YOLO with VOC data, YOLO has the capability to detect the objects from among these classes only. Objects that are not present in these classes are ignored in most of the cases. This is discussed in the end of this documentation
The coordinates of the bounding boxes are used to crop the objects inside the boxes and run our tensor flow classifier on these images. To crop the images, use crop.py program and adjust the following lines as directed:
```
line 2 : path to box_cord.txt file
line 4 : path to our original test image on which you ran the detect program.
Line 12: path to the folder where you want to save the cropped images
```
After running this script, cropped images are saved in a folder of your choice. Now we need to run our classifier on these images. Note that the .pb file and .txt file are now used to make predictions. Copy the classifier.py program from http://bit.ly/2tMn8c8 and edit the following lines:

```
line 7 : replace the existing path with path to the folder with cropped images
line 19: replace the existing path with path to labels.txt file generated during our we trained our inception model
line 20 :  replace the existing path with path to .pb file generated during our we trained our inception model
```
Note : classifier.py program has been written assuming that number of cropped images are less than or equal to 10. Running the classifier on more images imposes heavy burden on the computer.
When you run this program, predictions are made on the cropped images and the results are stored in a text file named output.txt . This file is created where the classify.py exists.
This output.txt is the final output file with all the objects in our test image.
 
The flow of our program can be visualised as follows:


##Possible extensions and suggestions to the reader:
 
Although the above implemented programs produce results good enough for smart fridge, we suggest the reader to try the following approaches. These might produce better results than the above program.
 
1. Darkflow :
	When we searched for an implementation of darknet in tensor flow, we came across this interesting git repository. The author has implemented darknet using tensor flow and made it easy to train yolo. Instead of training tensor flow inception model and yolo separately, this model tries to achieve better detection results and tries to avoid classification part. Again this is possible only with a good dataset. Some of the additional features of this model include producing the output in json format which can be useful to integrate this model with other applications.
```
Github link : https://github.com/thtrieu/darkflow
 ```
2. For implementing darknet in low weight devices like raspberry pi and mobiles , have a look at this repo : https://github.com/DT42/BerryNet
 
3. As of now, training is only possible in gpu and cpu. Tensor Flow provides machine learning community with lightweight model of tensor flow like tf-slim. We did not have time to rewrite our code in tf-slim but we suggest the reader to  try this out.
 
4. Microsoft CNTK worked on a similar project as ours and made it open source. Object detection using CNTK can be found here : https://github.com/Azure/ObjectDetectionUsingCntk . Microsoft went a step further and using their object detection model in a smart fridge. Microsoft collaborated with leibherr appliances. More about it here : https://blogs.technet.microsoft.com/machinelearning/2016/09/02/microsoft-and-liebherr-collaborating-on-new-generation-of-smart-refrigerators/ . Infact, in their git repository, microsoft demonstrates its object detection model using a grocery dataset. We strongly suggest the reader to go through the above mentioned git repository.
 
5. Using IBM watson. The artificial intelligence community is well aware of it. IBM provides visual recognition service where the user gets to train using their own data and use the model for predictions using visual recognition API. More about this at : http://bit.ly/2tnJxvp . The programs which use IBM watson architecture run in highly efficient servers and produce reliable results. The trial version last for about a month and these services can be used with a premium version.
 
6. Using Google vision for image detection. We won't talk much about it here but everything can be found in their official documentation. Be sure to check this out : https://cloud.google.com/vision/ . Note that google vision and IBM watson are paid services.
 
 
##Resources :
 
tensroflow slim version : https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
 
image classification using CNNs : http://cs231n.github.io/classification/
CNN : https://www.youtube.com/watch?v=FmpDIaiMIeA&t=1181s
	https://www.youtube.com/watch?v=2-Ol7ZB0MmU
	https://www.youtube.com/watch?v=40riCqvRoMs (must watch)
inception model training : https://www.youtube.com/watch?v=m2D02eZTB4s
				 : https://www.youtube.com/watch?v=wuo4JdG3SvU&list=PL9Hr9sNUjfsmEu1ZniY0XpHSzl5uihcXZ
 
Using tensorflow models in java : https://medium.com/google-cloud/how-to-invoke-a-trained-tensorflow-model-from-java-programs-27ed5f4f502d
Using trained model in rasberry pi : https://medium.com/@bapireddy/real-time-image-classifier-on-raspberry-pi-using-inception-framework-faccfa150909
tensorfow in android applications  https://medium.com/@daj/creating-an-image-classifier-on-android-using-tensorflow-part-1-513d9c10fa6a
 
yolo training : http://guanghan.info/blog/en/my-works/train-yolo/
		  : https://timebutt.github.io/static/how-to-train-yolov2-to-detect-custom-objects/
yolo training : community  support      :https://groups.google.com/forum/#!forum/darknet  
 					:https://github.com/AlexeyAB/darknet
DATASETS: http://image-net.org/
vegetables dataset : https://drive.google.com/drive/folders/0B-BWEF7_tH_WNVhqZ0ZjcmRzeGc
Annotation tools  :  https://github.com/AlexeyAB/Yolo_mark
      		: https://github.com/puzzledqs/BBox-Label-Tool
			: https://github.com/tzutalin/labelImg
			: https://github.com/tzutalin/ImageNet_Utils
CNTK installation : https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-
CNTK-on-Linux
IBM watson : https://medium.com/unsupervised-coding/dont-miss-your-target-object-detection-with-tensorflow-and-watson-488e24226ef3  
Yolo training : http://machinethink.net/blog/object-detection-with-yolo/


 
 
 
 
 

