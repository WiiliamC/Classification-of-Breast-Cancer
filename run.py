# 该程序实现的功能：
# 1.数据预处理，存储所有特征向量文件至CACHE_DIR目录
# 2.读取所有特征向量文件，并分为train、test数据集
# 3.训练与测试
import os
import tensorflow as tf
import time
from tensorflow.python.platform import gfile
import shutil
import numpy as np
import random
import winsound
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util
import gc

# 指定运行设备 0-GPU 1-CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 获取工作路径
current_path = os.getcwd()
# 【INPUT】图片数据文件夹。
INPUT_DATA = current_path + '\\data set'
# 【OUTPUT1】特征向量文件夹
CACHE_DIR = current_path + '\\bottlenecks'
# 【INPUT】下载的谷歌训练好的Inception-v3模型(Provided by Goolge)文件目录
MODEL_DIR = current_path + '\\inceptionV3'
# 【INPUT】下载的谷歌训练好的Inception-v3模型文件名
MODEL_FILE = 'googlenet-v3.frozen.pb'
# 图片处理batchsize
PRETRAIN_BATCH_SIZE = 256

# Inception-v3模型输出节点维度
BOTTLENECK_TENSOR_SIZE = 2048
# InceptionV3模型输出节点
BOTTLENECK_TENSOR_NAME = 'InceptionV3/Logits/AvgPool_1a_8x8/AvgPool:0'
# InceptionV3模型输入节点
JPEG_DATA_TENSOR_NAME = 'input:0'

# 四个子文件夹的文件夹名称
subfile_names = ['40x', '100x', '200x', '400x']
# 特征向量文件名中的放大倍数关键词
keywords = ['-40-', '-100-', '-200-', '-400-']
# 图片数据
image_height = 460
image_width = 700
# 神经网络结构参数
LAYER1_NODES = [1000]
# 项目内训练对象的改变所引起的调整参数
n_classes = 2
Mfactor = ['40x', '100x', '200x', '400x']
mindex = 0
ground_truth_lables = ['_B_', '_M_']
# ground_truth_lables = ['_A-','_DC-','_F-','LC','_MC-','_PC-','_PT-','_TA-']

# 训练参数
STEPS = 130000
BATCH_SIZE = 5000
VALIDATION_SIZE = 100
LEARNING_RATE_BASE = 0.01  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
LEARNING_RATE_DECAY_STEPS = 500  # 设置学习率衰减轮数
REGULARAZTION_RATE = 0.0001  # 描述模型复杂度的正则化项在损失函数中的系数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

# 是否读取pb文件进行变量初始化，1为是，0为否
read_pb_variables = 0
read_pb_var_path = r'E:\CHEN\Chen Cheng project\4.11-project\tmp_pb_model\2Cata-s60000-b5000-tmp_IDCmodel.pb'


# 将秒数间隔转换为计时器"时:分:秒"字符串
def second2time(iItv):
    iItv = int(iItv)
    h = iItv // 3600
    sUp_h = iItv - 3600 * h
    m = sUp_h // 60
    sUp_m = sUp_h - 60 * m
    s = sUp_m
    return (str(h) + '小时 ' + str(m) + '分钟 ' + str(s) + '秒')


# 该函数实现的功能：根据一个循环的开始时间，index及length值估计剩余时间，并打印进度及剩余时间
def print_rest_time(time1, index, length, name):
    print('【', name, '】:%.1f%%' % (index / length * 100))
    print('rest time:', second2time((time.time() - time1) * (length - index)))


# 该函数实现的功能：从指定目录及其子文件夹中搜索所有文件，并将其目录存储于result列表中
def return_all_files(target_path, result):
    for file in os.listdir(target_path):
        if os.path.isfile(target_path + '\\' + file):
            result.append(target_path + '\\' + file)
        else:
            return_all_files(target_path + '\\' + file, result)


# 该函数实现的功能：输入一个图片数据，返回一个列表，列表中依次存储：原始图片，左右翻转图片，上下翻转图片，对角翻转图片
#                                                            顺时针旋转90°、180°、270°，增加、减少亮度，
#                                                            增加、减少对比度，调整色度，增加、降低饱和度
def image_process(original_image, sess, generated_images, image_names, image_name):
    # 图片张量
    image_tensors = []
    # 图片增强数据集队列
    # 图片大小变化
    resized_original_image = tf.image.resize_images(original_image, size=[299, 299], method=0)
    # 0
    image_tensors.append(resized_original_image)
    # 图像翻转 1 2 3
    flip_lr_image = tf.image.flip_left_right(resized_original_image)
    image_tensors.append(flip_lr_image)
    flip_ud_image = tf.image.flip_up_down(resized_original_image)
    image_tensors.append(flip_ud_image)
    flip_qm_image = tf.image.transpose_image(resized_original_image)
    image_tensors.append(flip_qm_image)
    # 图像旋转 4 5 6
    rot_90_image = tf.image.flip_left_right(flip_qm_image)
    image_tensors.append(rot_90_image)
    rot_180_image = tf.image.flip_up_down(flip_lr_image)
    image_tensors.append(rot_180_image)
    rot_270_image = tf.image.transpose_image(flip_lr_image)
    image_tensors.append(rot_270_image)
    # 临时图片张量列表
    tmp_image_tensors = image_tensors[:]
    # 色彩调整
    for tmp_image_tensor in tmp_image_tensors:
        # 设置图片的亮度 7 8
        increase_brightness = tf.image.adjust_brightness(tmp_image_tensor, 0.14)
        decrease_brightness = tf.image.adjust_brightness(tmp_image_tensor, -0.14)
        image_tensors.append(increase_brightness)
        image_tensors.append(decrease_brightness)
        # 设置图片的对比度 9 10
        increase_contrast = tf.image.adjust_contrast(tmp_image_tensor, 1.4)
        decrease_contrast = tf.image.adjust_contrast(tmp_image_tensor, 0.5)
        image_tensors.append(increase_contrast)
        image_tensors.append(decrease_contrast)
        # 设置图片的饱和度 11 12
        increase_satu = tf.image.adjust_saturation(tmp_image_tensor, 1.65)
        decrease_satu = tf.image.adjust_saturation(tmp_image_tensor, 0.65)
        image_tensors.append(increase_satu)
        image_tensors.append(decrease_satu)

    # 运行sess获取张量值
    for index, image_tensor in enumerate(image_tensors):
        generated_images.append([sess.run(image_tensor)])
        image_names.append(image_name + '-' + str(index))


# 该函数实现的功能：进行数据预处理，将INPUT_DATA路径下的所有图片进行image_process预处理，计算特征向量
# 并将特征向量存储至CACHE_DIR目录下
def pretrain():
    # 开始
    print('--------------------------------------pretrain start--------------------------------------------')
    # 获取图片路径
    image_paths = []
    return_all_files(INPUT_DATA, image_paths)
    image_number = len(image_paths)
    calculate_times = image_number // PRETRAIN_BATCH_SIZE + 1
    for i in range(calculate_times):
        print(i, 'in', calculate_times)
        # 读取图片
        images = []
        image_names = []
        end_index = min(((i + 1) * PRETRAIN_BATCH_SIZE), len(image_paths))
        sub_image_paths = image_paths[i * PRETRAIN_BATCH_SIZE:end_index]
        for index, image_path in enumerate(sub_image_paths):
            # 获取图片名
            image_name = os.path.basename(image_path)
            # 检测是否已计算该向量
            bottleneck_path = CACHE_DIR + '\\' + image_name + '-48.txt'
            if os.path.exists(bottleneck_path):
                del image_name, bottleneck_path, index, image_path
                gc.collect()
                continue
            # 计时模块
            time1 = time.time()
            # 定义会话
            sess = tf.Session()
            # 获取图片内容
            original_image = gfile.FastGFile(image_path, 'rb').read()
            # 图片解码
            original_image = tf.image.decode_png(original_image)
            # 转码
            original_image = tf.image.convert_image_dtype(original_image, dtype=tf.float32)
            # 图片数据集增强
            image_process(original_image, sess, images, image_names, image_name)
            sess.close()
            tf.reset_default_graph()
            if index % 5 == 0:
                print(i, 'in', calculate_times)
                print_rest_time(time1, index, len(sub_image_paths), 'load images')
        # 如果images列表为空则继续
        if not len(images):
            del i, images, image_names, end_index, sub_image_paths
            gc.collect()
            continue
        # 读取pb模型及张量
        with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, return_elements=[BOTTLENECK_TENSOR_NAME,
                                                                                              JPEG_DATA_TENSOR_NAME])
        # 定义会话
        sess = tf.Session()
        # 创建特征向量文件夹
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        # 计算特征向量
        for index, image in enumerate(images):
            # 计时模块
            time1 = time.time()
            bottleneck_path = CACHE_DIR + '\\' + image_names[index] + '.txt'
            if os.path.exists(bottleneck_path):
                continue
            bottleneck_value = sess.run(bottleneck_tensor, feed_dict={jpeg_data_tensor: image})
            bottleneck_value = bottleneck_value[0][0]
            # 将计算得到的特征向量存入文件
            bottleneck_string = ','.join(str(x) for x in bottleneck_value[0])
            with open(bottleneck_path, 'w') as bottleneck_file:
                bottleneck_file.write(bottleneck_string)
            # 计时模块
            if index % 5 == 0:
                print(i, 'in', calculate_times)
                print_rest_time(time1, index, len(images), 'save images')
        # 关闭会话
        sess.close()


# 该函数实现的功能：在str字符串中查找第i个字符的所在位置并返回
def get_letter_position(str, i, letter):
    flag = 0
    for index, s in enumerate(str):
        if s == letter:
            flag = flag + 1
        if flag == i and s == letter:
            return index


# 该函数实现的功能：在source_path目录下的特征向量文件中随机抽取20名病人的数据作为测试集，
# 其余作为训练集，分别存储在工作目录下的test、train文件夹中，并将test文件夹中的数据按照
# 放大倍数分别放入40x、100x、200x、400x子文件夹中
def data_process(source_path):
    # 开始
    print('--------------------------------------dataset process start-------------------------------------')
    # 生成train,test文件夹
    train_path = os.path.join(current_path, 'train')
    test_path = os.path.join(current_path, 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    # 获取所有特征向量文件的路径
    tmpfiles_path = []
    return_all_files(source_path, tmpfiles_path)
    # 根据文件名获取病人名
    patient_names = []
    for tmpfile_path in tmpfiles_path:
        filename = os.path.basename(tmpfile_path)
        position1 = get_letter_position(filename, 2, '-')
        position2 = get_letter_position(filename, 3, '-')
        patient_name = filename[position1 + 1:position2]
        if patient_name not in patient_names:
            patient_names.append(patient_name)
        # if len(patient_names) == 82:
        #     break
    # 随机抽取20名病人放入test列表，其余病人放入train文件夹
    random.shuffle(patient_names)
    testfile_paths = []
    for tmpfile_path in tmpfiles_path:
        filename = os.path.basename(tmpfile_path)
        position1 = get_letter_position(filename, 2, '-')
        position2 = get_letter_position(filename, 3, '-')
        patient_name = filename[position1 + 1:position2]
        if patient_names.index(patient_name) < 20:
            testfile_paths.append(tmpfile_path)
        else:
            shutil.copy(tmpfile_path, train_path)
    # 将test文件按照放大倍数分别放入四个子文件夹中
    # 创建四个子文件夹
    subfile_paths = []
    for subfile_name in subfile_names:
        subfile_path = os.path.join(test_path, subfile_name)
        subfile_paths.append(subfile_path)
        if not os.path.exists(subfile_path):
            os.makedirs(subfile_path)
    for testfile_path in testfile_paths:
        testfile_name = os.path.basename(testfile_path)
        if '-0.txt' in testfile_name:
            for subfile_path, keyword in zip(subfile_paths, keywords):
                if keyword in testfile_name:
                    shutil.copy(testfile_path, subfile_path)
                    break
    # 结束
    winsound.Beep(400, 200)
    winsound.Beep(900, 200)
    print('-----------------------------------dataset process successfully---------------------------------')


# 该函数实现的功能：在目录data_path下，读取tmp_filename文件,获取其数据内容加入data列表中，制作标签加入groud_truth列表中
def append_tmpfile(data_path, tmp_filename, data, ground_truth, n_classes):
    tmp_filedirs = os.path.join(data_path, tmp_filename)
    with open(tmp_filedirs, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    bottleneck_values = [[float(x) for x in bottleneck_string.split(',')]]
    data.append(bottleneck_values[0])
    ground_truth_values = np.zeros(n_classes, dtype=np.float32)
    for index, ground_truth_lable in enumerate(ground_truth_lables):
        if ground_truth_lable in tmp_filename:
            ground_truth_values[index] = 1.0
    ground_truth.append(ground_truth_values)


# 该函数实现的功能：从train_data_path随机获取BATCH_SIZE个训练数据
def get_batch(train_data_path, BATCH_SIZE, n_classes):
    train_data = []
    groud_truth = []
    # 获取tmp文件名列表
    tmp_filenames = os.listdir(train_data_path)
    random.shuffle(tmp_filenames)
    batch_filenames = tmp_filenames[0:BATCH_SIZE]
    for batch_filename in batch_filenames:
        append_tmpfile(train_data_path, batch_filename, train_data, groud_truth, n_classes)
    return train_data, groud_truth


# 该函数实现的功能：从data_path读取所有验证或测试数据
def get_all_data(data_path, n_classes):
    tmp_data = []
    groud_truth = []
    # 获取tmp文件名列表
    tmp_filenames = os.listdir(data_path)
    for i, tmp_filename in enumerate(tmp_filenames):
        time1 = time.time()
        append_tmpfile(data_path, tmp_filename, tmp_data, groud_truth, n_classes)
        if i % 1000 == 0:
            print_rest_time(time1, i, len(tmp_filenames), 'get all data')
    return tmp_data, groud_truth


# 该函数实现的功能：从已经读取好的数据中随机抽取BATCH_SIZE个数据及其label
def get_random_data(data, label, BATCH_SIZE):
    subdata = []
    sublabel = []
    for index in range(BATCH_SIZE):
        random_number = random.randrange(0, len(data))
        subdata.append(data[random_number])
        sublabel.append(label[random_number])
    return subdata, sublabel


# 训练及测试
# 【OUTPUT】输出文件夹位于工作目录下，文件名如下
# output_file_name = 'output'
# tmp_pb_file_name = 'tmp_pb_model'
def train_and_test():
    # 获取工作路径
    current_path = os.getcwd()
    # 【INPUT】获取数据集路径
    train_data_path = os.path.join(current_path, 'train')
    test_data_path = os.path.join(current_path, 'test') + '\\' + Mfactor[mindex]
    # 读取所有训练数据与测试数据
    all_train_data, all_train_truth = get_all_data(train_data_path, n_classes)
    all_test_data, all_test_truth = get_all_data(test_data_path, n_classes)
    for LAYER1_NODE in LAYER1_NODES:
        # 【OUTPUT】输出文件夹位于工作目录下，文件名如下
        output_file_name = 'output'
        tmp_pb_file_name = 'tmp'
        # 根据read_pb_variables判断如何进行变量初始化
        with tf.Session() as sess1:
            # 若为1，读取pb文件进行变量初始化
            if read_pb_variables == 1:
                # 读取pb文件
                with gfile.FastGFile(read_pb_var_path, 'rb') as f:
                    graph_def1 = tf.GraphDef()
                    graph_def1.ParseFromString(f.read())
                    tf.import_graph_def(graph_def1, name='')
                # 存储变量
                w1 = (sess1.run('final_training_ops/Variable:0'))
                b1 = (sess1.run('final_training_ops/Variable_1:0'))
                w2 = (sess1.run('final_training_ops/Variable_2:0'))
                b2 = (sess1.run('final_training_ops/Variable_3:0'))
        # 清空图定义
        tf.reset_default_graph()
        # 定义placeholder
        bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
        ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
        # 定义神经网络结构
        with tf.name_scope('final_training_ops'):
            # 生成隐藏层的参数。
            # 根据read_pb_variables判断变量初始化方式
            # 若为1，用w、b进行初始化
            if read_pb_variables == 1:
                weights1 = tf.Variable(w1)
                biases1 = tf.Variable(b1)
                weights2 = tf.Variable(w2)
                biases2 = tf.Variable(b2)
            else:
                weights1 = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, LAYER1_NODE], stddev=0.001))
                biases1 = tf.Variable(tf.zeros([LAYER1_NODE]))
                # 生成输出层的参数。
                weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, n_classes], stddev=0.001))
                biases2 = tf.Variable(tf.zeros([n_classes]))
            # 正则化防止过拟合
            regularizer = tf.contrib.layers.l2_regularizer(0.05)
            tf.add_to_collection('losses', regularizer(weights2))
            # 定义训练轮数
            global_step = tf.Variable(0, trainable=False)
            # 设置指数衰减的学习率。
            learning_rate = tf.train.exponential_decay(
                LEARNING_RATE_BASE,
                global_step,
                LEARNING_RATE_DECAY_STEPS,
                LEARNING_RATE_DECAY,
                staircase=True)
            # 前向传播
            logits1 = tf.nn.relu(tf.matmul(bottleneck_input, weights1) + biases1)
            logits2 = tf.matmul(logits1, weights2) + biases2
            # 计算概率
            final_tensor = tf.nn.softmax(logits2)
            # 定义交叉熵损失函数
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=ground_truth_input)
            loss = tf.reduce_mean(cross_entropy)
            tf.add_to_collection('losses', loss)
            losses = tf.add_n(tf.get_collection('losses'))
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(losses, global_step=global_step)
            # 定义evaluation操作
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 开始会话
        with tf.Session() as sess:
            # 变量初始化
            tf.global_variables_initializer().run()
            # 训练过程
            for i in range(STEPS):
                # 计时模块
                time1 = time.time()
                # 获取训练数据
                train_data, train_ground_truth = get_random_data(all_train_data, all_train_truth, BATCH_SIZE)
                # time2 = time.time()
                # print('get data time cost:',time2-time1)
                # 开始训练
                sess.run(train_step,
                         feed_dict={bottleneck_input: train_data, ground_truth_input: train_ground_truth})
                # time3 = time.time()
                # print('train time cost:',time3-time2)
                # 在验证集上测试正确率。
                if i % 100 == 0 or i + 1 == STEPS:
                    # 获取验证、测试数据
                    validation_data, validation_ground_truth = get_random_data(all_test_data, all_test_truth,
                                                                               VALIDATION_SIZE)
                    validation_loss, validation_accuracy = sess.run([loss, evaluation_step], feed_dict={
                        bottleneck_input: validation_data, ground_truth_input: validation_ground_truth})
                    print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%'
                          % (i, VALIDATION_SIZE, validation_accuracy * 100))
                    # 存储临时pb文件
                    tmp_pb_path = os.path.join(current_path, tmp_pb_file_name)
                    if not os.path.exists(tmp_pb_path):
                        os.makedirs(tmp_pb_path)
                    nodes_name_list = ["evaluation/ArgMax"]
                    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                               output_node_names=nodes_name_list)
                    with tf.gfile.FastGFile(tmp_pb_path + '\\' + 'tmp-model.pb', mode='wb') as f:
                        f.write(constant_graph.SerializeToString())
                # 计时模块
                time_end = time.time()
                print('step %d in %d: calculated time left = ' % (i, STEPS), (time_end - time1) * (STEPS - i), 's')
            # 在测试数据上计算image-level正确率
            test_accuracys = []
            for mfactor in Mfactor:
                test_data_path = current_path + r'\test' + '\\' + mfactor
                all_test_data, all_test_truth = get_all_data(test_data_path, n_classes)
                test_accuracy = sess.run(evaluation_step, feed_dict={bottleneck_input: all_test_data,
                                                                     ground_truth_input: all_test_truth})
                test_accuracys.append(test_accuracy)
                print('Final test accuracy on ' + mfactor + ' image-level = %.1f%%' % (test_accuracy * 100))

            # 存储pb文件
            new_pb_path = current_path + '\\' + output_file_name
            if not os.path.exists(new_pb_path):
                os.makedirs(new_pb_path)
            nodes_name_list = ["evaluation/ArgMax"]
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                       output_node_names=nodes_name_list)
            with tf.gfile.FastGFile(new_pb_path + '\\' + 'BC-Classifier.pb', mode='wb') as f:
                f.write(constant_graph.SerializeToString())


def main(_):
    # 数据预处理，并存储所有特征向量文件
    pretrain()
    # 将数据分离至工作目录下train、test文件夹
    data_process(CACHE_DIR)
    # 训练及测试
    train_and_test()


if __name__ == '__main__':
    tf.app.run()
