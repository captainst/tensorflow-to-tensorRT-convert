import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from efficientnet import load_model
from tensorflow.python.framework import graph_io

output_names = ['dense_2/Softmax']

classify_graph = tf.Graph()
with classify_graph.as_default():
    cs_graph_def = tf.GraphDef()
    with tf.gfile.FastGFile("./models/efficientnet.pb", 'rb') as fid:
        serialized_graph = fid.read()
        cs_graph_def.ParseFromString(serialized_graph)
        #tf.import_graph_def(cs_graph_def, name='')

        trt_graph = trt.create_inference_graph(
            input_graph_def=cs_graph_def,
            outputs=output_names,
            max_batch_size=16,
            max_workspace_size_bytes=1 << 25,
            precision_mode='FP16',
            minimum_segment_size=3
        )

        graph_io.write_graph(trt_graph, "./models/", "efficientnet_trt.pb", as_text=False)

print('done')