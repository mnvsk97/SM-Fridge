import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import sys
i=0

filelist= [file for file in os.listdir('/home/krishna/Desktop/segmented_images') if file.endswith('.jpeg')]
file = open('output.txt', 'w')
for f in filelist:
    image_path="/home/krishna/Desktop/segmented_images/tests"
    plus=str(i)+".jpeg"
    image_path += plus
    i += 1
    image_data = tf.gfile.FastGFile(image_path , 'rb').read()
        #image_data=cv2.imread(os.path.abspath(f))
    label_lines = [line.rstrip() for line 
                          in tf.gfile.GFile("/home/krishna/Desktop/tf_files/retrained_labels.txt")]
    with tf.gfile.FastGFile("/home/krishna/Desktop/tf_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, \
                {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    for node_id in top_k:
        human_string = label_lines[node_id]
        output=human_string
        score = predictions[0][node_id]   
        print('%s (score = %.5f)' % (human_string, score))
        file.write(human_string + '\n')
        break
file.close()