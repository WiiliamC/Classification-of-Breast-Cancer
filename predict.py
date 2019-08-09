import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
# 指定运行设备 0-GPU 1-CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 获取工作路径
current_path = os.getcwd()
# 输入pb模型的路径
# model_path = r'E:\海外毕设-BC-classification\output_pb\2Cata-l0.010-0.99-300-s50000-b400-r0-newIDCmodel-99.0%.pb'
# image_path = r'E:\海外毕设-BC-classification\project\2Cata_photos\bad\SOB_M_DC-14-2523-400-005.png'
model_path = input('Model path:')
if not model_path:
    model_path = os.path.join(current_path, 'output', 'BC-Classifier.pb')
image_path = input('Image path:')
if not image_path:
    image_path = os.path.join(current_path, 'image to be predicted', 'image.png')
# 生成会话
sess = tf.Session()

# 导入pb模型
with gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

# 图片预处理
with tf.name_scope("img_process"):
    # 获取图片内容。
    image_data = gfile.FastGFile(image_path, 'rb').read()
    # 图片解码
    image_data = tf.image.decode_png(image_data)
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    # 图片尺寸变换
    image_data = tf.image.resize_images(image_data, size=[299, 299], method=0)
    image_data = sess.run(image_data)
    image_data = [image_data]

# 获取张量
input = sess.graph.get_tensor_by_name('import/input:0')
output = sess.graph.get_tensor_by_name('import/InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0')
# 获取变量
weights1 = sess.run('final_training_ops/Variable:0')
biases1 = sess.run('final_training_ops/Variable_1:0')
weights2 = sess.run('final_training_ops/Variable_2:0')
biases2 = sess.run('final_training_ops/Variable_3:0')

# 进行预测
with tf.name_scope('predict'):
    logits1 = tf.nn.relu(tf.matmul(output[0][0], weights1) + biases1)
    logits2 = tf.matmul(logits1, weights2) + biases2
    final_tensor = tf.nn.softmax(logits2)
    result = tf.argmax(final_tensor, 1, name='result')

# 输出结果
print('\n-------------------inference-proceeding----------------------\n')
if sess.run(result, feed_dict={input: image_data}) == 0: print("Prediction: malignant")
if sess.run(result, feed_dict={input: image_data}) == 1: print("Prediction: benign")
print('\n-------------------inference-successful----------------------', '\n')

# 结束会话
sess.close()
