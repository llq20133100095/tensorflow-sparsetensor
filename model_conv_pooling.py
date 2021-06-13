
# @create time: 20210608 17:15
# @author: leolqli
import logging
import tensorflow as tf
import time
import numpy as np
from util_event import Hparams, get_sparse_tensor
from scipy.sparse import csr_matrix, coo_matrix
from mobilenet_custom import conv_block, depthwise_conv_block, MobileNet

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args(args=[])

class LSTM_Baseline(tf.keras.Model):
    def __init__(self, rate=0.3):
        super(LSTM_Baseline, self).__init__()
        
        # t feature
        self.t_b1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,  return_sequences=False))
        # self.t_b2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,  return_sequences=False))
        self.t_d1 = tf.keras.layers.Dense(256, activation='relu') # 64
        self.t_dropout = tf.keras.layers.Dropout(rate)

    def call(self, input_t, training):

        # t
        input_t = self.t_b1(input_t)
        # input_t = self.t_b2(input_t)
        input_t = self.t_d1(input_t)                # (None, 128)
        input_t = self.t_dropout(input_t, training=training)

        return input_t

class MobileNetCus(tf.keras.Model):
    def __init__(self, frame, input_shape, rate=0.2):
        super(MobileNetCus, self).__init__()

        self.mobilenet = [MobileNet(input_shape=input_shape, weights=None, include_top=False) for _ in range(frame)]
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.max_pooling3d = tf.keras.layers.MaxPooling3D()
        self.ave3d = tf.keras.layers.GlobalAveragePooling3D()
        self.dense1 = tf.keras.layers.Dense(256, activation="relu")
        self.dropout = tf.keras.layers.Dropout(rate)
        self.frame = frame

    def call(self, inp, training):
        # each frame
        frame = inp.shape[1]
        map_net_output = []
        for f in range(self.frame):
            input_f = inp[:, f, :, :, :]
            # (None, 6, 6, 128)
            output_mobilenet = self.mobilenet[f](input_f)
            output_mobilenet = output_mobilenet[:, tf.newaxis, :, :, :]
            map_net_output.append(output_mobilenet)

        # concate
        map_net_output = self.concat(map_net_output)     # (None, frame, 6, 6, 128)
        map_net_output = self.max_pooling3d(map_net_output)
        map_net_output = self.ave3d(map_net_output)
        map_net_output = self.dropout(map_net_output, training=training)
        map_net_output = self.dense1(map_net_output)     # (None, 128)

        return map_net_output

class MobileNetMap(tf.keras.Model):
    def __init__(self, frame, input_shape_global=[None, None, 7], input_shape_mini=[None, None, 17]):
        super(MobileNetMap, self).__init__()

        self.mobilenet_global = MobileNetCus(frame, input_shape_global)
        self.mobilenet_mini = MobileNetCus(frame, input_shape_mini)
        self.lstm_seq = LSTM_Baseline()
        self.concat = tf.keras.layers.Concatenate(axis=1)
        self.frame = frame

    def call(self, input_seq_feature, input_global_map, input_mini_map, training):

        input_global_map = tf.sparse.reshape(input_global_map, [-1, self.frame, 7, 23, 23])
        input_mini_map = tf.sparse.reshape(input_mini_map, [-1, self.frame, 17, 31, 31])

        input_global_map = tf.sparse.to_dense(input_global_map)
        input_mini_map = tf.sparse.to_dense(input_mini_map)

        # transpose
        input_global_map = tf.transpose(input_global_map, perm=[0, 1, 3, 4, 2])
        input_mini_map = tf.transpose(input_mini_map, perm=[0, 1, 3, 4, 2])

        input_global_map = self.mobilenet_global(input_global_map, training=training)
        input_mini_map = self.mobilenet_mini(input_mini_map, training=training)
        input_seq_feature = self.lstm_seq(input_seq_feature, training=training)
        
        # (none, 384)
        con_output = self.concat([input_global_map, input_mini_map, input_seq_feature])

        return con_output

class MobileNetStack(tf.keras.Model):
    def __init__(self, grass_frame, skill_frame, gank_frame, input_shape_global=[None, None, 7], input_shape_mini=[None, None, 17]):
        super(MobileNetStack, self).__init__()

        self.mobilenetmap_grass = MobileNetMap(grass_frame, input_shape_global, input_shape_mini)
        self.mobilenet_skill = MobileNetMap(skill_frame, input_shape_global, input_shape_mini)
        self.mobilenet_gank = MobileNetMap(gank_frame, input_shape_global, input_shape_mini)

        # statics feature
        self.sta_d1 = tf.keras.layers.Dense(64, activation='relu')
        self.sta_d2 = tf.keras.layers.Dense(192, activation='relu')
        self.sta_d3 = tf.keras.layers.Dense(768, activation='relu')

        self.final_layers = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, grass_seq_feature, grass_global_map_feature, grass_mini_map_feature,
                    skill_seq_feature, skill_global_map_feature, skill_mini_map_feature,
                    grank_seq_feature, grank_global_map_feature, grank_mini_map_feature,
                    all_static_feature, training):

        output_grass = self.mobilenetmap_grass(grass_seq_feature, grass_global_map_feature, grass_mini_map_feature, training)
        output_skill = self.mobilenet_skill(skill_seq_feature, skill_global_map_feature, skill_mini_map_feature, training)
        output_gank = self.mobilenet_gank(grank_seq_feature, grank_global_map_feature, grank_mini_map_feature, training)

        input_sta = self.sta_d1(all_static_feature)
        input_sta = self.sta_d2(input_sta)
        input_sta = self.sta_d3(input_sta)

        con_output = output_grass + output_skill + output_gank + input_sta
        # con_output = output_skill
        con_output = self.final_layers(con_output)

        return con_output


if __name__ == "__main__":
    start = time.time()

    """ Dense """
    # input_seq_feature_grass = tf.random.normal([64, 36, 54])
    # input_global_map_grass = tf.random.normal([64, 36, 7 * 23 * 23])
    # input_mini_map_grass = tf.random.normal([64, 36, 17 * 31 * 31])

    # input_seq_feature_skill = tf.random.normal([64, 50, 54])
    # input_global_map_skill = tf.random.normal([64, 50, 7 * 23 * 23])
    # input_mini_map_skill = tf.random.normal([64, 50, 17 * 31 * 31])

    # input_seq_feature_gank = tf.random.normal([64, 50, 54])
    # input_global_map_gank = tf.random.normal([64, 50, 7 * 23 * 23])
    # input_mini_map_gank = tf.random.normal([64, 50, 17 * 31 * 31])

    # input_sta = tf.random.normal([64, 482])

    # mobilenetcus = MobileNetStack(36, 50, 50)
    # print(time.time() - start)
    # net_output = mobilenetcus(input_seq_feature_grass, input_global_map_grass, input_mini_map_grass, 
    #                             input_seq_feature_skill, input_global_map_skill, input_mini_map_skill, 
    #                             input_seq_feature_gank, input_global_map_gank, input_mini_map_gank, 
    #                             input_sta)
    # print(mobilenetcus.summary())
    # print(time.time() - start)

    """ Sparse """
    input_seq_feature_grass = np.random.normal(size=[2, 36, 54])
    input_global_map_grass = csr_matrix(np.random.normal(size=[2, 36 * 7 * 23 * 23]))
    input_mini_map_grass = csr_matrix(np.random.normal(size=[2, 36 * 17 * 31 * 31]))

    input_seq_feature_skill = np.random.normal(size=[2, 50, 54])
    input_global_map_skill = csr_matrix(np.random.normal(size=[2, 50 * 7 * 23 * 23]))
    input_mini_map_skill = csr_matrix(np.random.normal(size=[2, 50 * 17 * 31 * 31]))

    input_seq_feature_gank = np.random.normal(size=[2, 50, 54])
    input_global_map_gank = csr_matrix(np.random.normal(size=[2, 50 * 7 * 23 * 23]))
    input_mini_map_gank = csr_matrix(np.random.normal(size=[2, 50 * 17 * 31 * 31]))

    input_sta = tf.random.normal([2, 482])

    input_global_map_grass_sparse = get_sparse_tensor(input_global_map_grass)
    input_mini_map_grass_sparse = get_sparse_tensor(input_mini_map_grass)

    input_global_map_skill_sparse = get_sparse_tensor(input_global_map_skill)
    input_mini_map_skill_sparse = get_sparse_tensor(input_mini_map_skill)

    input_global_map_gank_sparse = get_sparse_tensor(input_global_map_gank)
    input_mini_map_gank_sparse = get_sparse_tensor(input_mini_map_gank)

    mobilenetcus = MobileNetStack(36, 50, 50)
    print(time.time() - start)
    net_output = mobilenetcus(input_seq_feature_grass, input_global_map_grass_sparse, input_mini_map_grass_sparse, 
                                input_seq_feature_skill, input_global_map_skill_sparse, input_mini_map_skill_sparse, 
                                input_seq_feature_gank, input_global_map_gank_sparse, input_mini_map_gank_sparse, 
                                input_sta, training=True)
    print(mobilenetcus.summary())
    print(time.time() - start)