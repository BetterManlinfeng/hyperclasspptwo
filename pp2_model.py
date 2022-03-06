
from tensorflow.keras import *
import  tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import layers, Sequential,regularizers
from tensorflow.keras.layers import Dropout
# from tensorflow.keras import *
#  定义一个3x3卷积！kernel_initializer='he_normal','glorot_normal'
from tensorflow.python.keras.layers import Concatenate


def regularized_padded_conv(*args, **kwargs):
    return layers.Conv2D(*args, **kwargs, padding='same', use_bias=False,
                         kernel_initializer='he_normal',
                         kernel_regularizer=regularizers.l2(5e-4))
############################### 通道注意力机制 ###############################
class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg= layers.GlobalAveragePooling2D()
        self.max= layers.GlobalMaxPooling2D()
        self.conv1 = layers.Conv2D(in_planes//ratio, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   use_bias=True, activation=tf.nn.relu)
        self.conv2 = layers.Conv2D(in_planes, kernel_size=1, strides=1, padding='same',
                                   kernel_regularizer=regularizers.l2(5e-4),
                                   use_bias=True)

    def call(self, inputs):
        avg = self.avg(inputs)
        max = self.max(inputs)
        avg = layers.Reshape((1, 1, avg.shape[1]))(avg)   # shape (None, 1, 1 feature)
        max = layers.Reshape((1, 1, max.shape[1]))(max)   # shape (None, 1, 1 feature)
        avg_out = self.conv2(self.conv1(avg))
        max_out = self.conv2(self.conv1(max))
        out = avg_out + max_out
        out = tf.nn.sigmoid(out)

        return out


############################### 空间注意力机制 ###############################
class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = regularized_padded_conv(1, kernel_size=kernel_size, strides=1, activation=tf.nn.sigmoid)

    def call(self, inputs):
        avg_out = tf.reduce_mean(inputs, axis=3)
        max_out = tf.reduce_max(inputs, axis=3)
        out = tf.stack([avg_out, max_out], axis=3)             # 创建一个维度,拼接到一起concat。
        out = self.conv1(out)

        return out




class BasicBlock(layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        # self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same', kernel_initializer='he_normal',kernel_regularizer=keras.regularizers.l2(5e-4))
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same',kernel_regularizer=regularizers.l2(0.0001))  #kernel_initializer='he_normal',
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same',kernel_regularizer=regularizers.l2(0.0001))
        self.bn2 = layers.BatchNormalization()

        ############################### 注意力机制 ###############################
        self.ca = ChannelAttention(filter_num)
        self.sa = SpatialAttention()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x


    def call(self, inputs, training=None):

        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        ############################### 注意力机制 ###############################
        out = self.ca(out) * out
        out = self.sa(out) * out

        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output

######################################
class build_resblock(keras.Model):
    def __init__(self, filter_num, stride):
        super(build_resblock, self).__init__()

        self.BasicBlock1 = BasicBlock(filter_num, stride)
        self.BasicBlock2 = BasicBlock(filter_num, stride=1)


    def call(self,blocks):
        res_blocks = Sequential()

        res_blocks.add(self.BasicBlock1)

        for _ in range(1, blocks):
            res_blocks.add(self.BasicBlock2)

        return res_blocks



def build_resblock(self, filter_num, blocks, stride=1):

    res_blocks = Sequential()
    # may down sample
    res_blocks.add(BasicBlock(filter_num, stride))

    for _ in range(1, blocks):
        res_blocks.add(BasicBlock(filter_num, stride=1))

    return res_blocks

######################################
class ResNet(keras.Model):


    def __init__(self, layer_dims, num_classes=16): # [2, 2, 2, 2]
        super(ResNet, self).__init__()

        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])

        self.layer1 = self.build_resblock(64,  layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=1)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=1)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=1)

        # output: [b, 512, h, w],
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes)


    def call(self, inputs, training=None):

        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # [b, c]
        x = self.avgpool(x)
        # [b, 100]
        x = self.fc(x)

        return x



    def build_resblock(self, filter_num, blocks, stride=1):

        res_blocks = Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2],num_classes=9)

def resnet34():
    return ResNet([3, 4, 6, 3],num_classes=9)


###########################  pp2主模型 ########################################
class  pp2_model(keras.Model):
    def __init__(self,filters_num,layer_dims,num_classes,dropout_rate):
        super(pp2_model, self).__init__()

        self.conv1 = layers.Conv3D(filters_num[0],kernel_size=(3,3,7),padding='same')   # filters_num = 8
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.Activation('relu')

        self.conv2 = layers.Conv3D(filters_num[1],kernel_size=(3,3,5),padding='same')   # filters_num = 16
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.Activation('relu')

        self.conv3 = layers.Conv3D(filters_num[2], kernel_size=(3, 3, 3), padding='same')   # filters_num = 32
        self.bn3 = layers.BatchNormalization()
        self.relu3 = layers.Activation('relu')

        # self.reshape = layers.Reshape()
        self.conv4 = layers.Conv2D(filters_num[3], kernel_size=(3, 3), padding='same')   # filters_num = 64
        self.bn4 = layers.BatchNormalization()
        self.relu4 = layers.Activation('relu')

        self.conv5 = layers.Conv2D(filters_num[4], kernel_size=(3, 3), padding='same')   # filters_num = **
        self.bn5 = layers.BatchNormalization()
        self.relu5 = layers.Activation('relu')
        self.dpout = layers.Dropout(dropout_rate)

        self.layer1 = self.build_resblock(filters_num[5], layer_dims[0])  # filters_num = 64
        self.layer2 = self.build_resblock(filters_num[6], layer_dims[1], stride=2)    # filters_num = 128
        self.layer3 = self.build_resblock(filters_num[7], layer_dims[2], stride=2)   # filters_num = 256
        self.layer4 = self.build_resblock(filters_num[8], layer_dims[3], stride=2)  # filters_num = 512

        # output: [b, 512, h, w],
        # self.fc1 = layers.Flatten()
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc2 = layers.Dense(filters_num[7],activation='relu')
        self.fc3 = layers.Dense(filters_num[6],activation='relu')
        self.fc4 = layers.Dense(num_classes)


    def call(self,inputs,training=None):

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        # reshape
        out = layers.Reshape((out.shape[1],out.shape[2],out.shape[3] * out.shape[4]))(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.dpout(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.dpout(out)
        out = self.relu5(out)



        x = self.layer1(out)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # [b, c]
        x = self.avgpool(x)
        # [b, 100]
        x = self.fc2(x)
        x = self.dpout(x)
        x = self.fc3(x)
        x = self.fc4(x)



        return x


    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks



class ResNet_block(keras.Model):


    def __init__(self, layer_dims,filters_num): # [2, 2, 2, 2]
        super(ResNet_block, self).__init__()
        #
        # self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
        #                         layers.BatchNormalization(),
        #                         layers.Activation('relu'),
        #                         layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        #                         ])

        self.layer1 = self.build_resblock(filters_num[0],  layer_dims[0]) # filters_num = 64
        self.layer2 = self.build_resblock(filters_num[1], layer_dims[1], stride=1)  # filters_num = 128
        self.layer3 = self.build_resblock(filters_num[2], layer_dims[2], stride=1)  # filters_num = 256
        self.layer4 = self.build_resblock(filters_num[3], layer_dims[3], stride=1)  # filters_num = 512

        # output: [b, 512, h, w],
        # self.avgpool = layers.GlobalAveragePooling2D()
        # self.fc = layers.Dense(num_classes)


    def call(self, inputs, training=None):

        # x = self.stem(inputs)

        x1 = self.layer1(inputs)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # [b, c]
        # x = self.avgpool(x)
        # [b, 100]
        # x = self.fc(x)

        return x2,x4



    def build_resblock(self, filter_num, blocks, stride=1):

        res_blocks = Sequential()
        # may down sample
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))

        return res_blocks

def network_up(input_layer_up,filters_num,dropout_rate,Block_res):
    # input_layer = Input(input_shape)
    # conv1 = layers.Conv3D(filters_num[0], kernel_size=(3, 3, 7), padding='same')(input_layer)  # filters_num = 8
    # conv1 = layers.Conv3D(filters_num[0], kernel_size=(3, 3, 3),padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001))(input_layer_up)  # filters_num = 8
    conv1 = layers.Conv3D(filters_num[0], kernel_size=(3, 3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(0.0001))(input_layer_up)  #kernel_initializer='he_normal',
    # conv_layer1m = tf.keras.layers.MaxPooling3D(pool_size=(1, 1, 1),padding='same')(conv1)
    # conv_layer1g = tf.keras.layers.GlobalMaxPooling3D()(conv1)
    conv1_bn = layers.BatchNormalization()(conv1)
    conv1_relu = layers.Activation('relu')(conv1_bn)
    # conv1_relu = Dropout(0.5)(conv1_relu)
    # conv1_relu = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(conv1_relu)

    # conv2 = layers.Conv3D(filters_num[1], kernel_size=(3, 3, 5), padding='same')(conv1_relu)  # filters_num = 16
    conv2 = layers.Conv3D(filters_num[1], kernel_size=(3, 3, 3),padding='same',kernel_regularizer=regularizers.l2(0.0001))(conv1_relu)  # filters_num = 16
    conv2_bn = layers.BatchNormalization()(conv2)
    conv2_relu = layers.Activation('relu')(conv2_bn)
    # conv2_relu = Dropout(0.5)(conv2_relu)
    # conv2_relu = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(conv2_relu)

    conv3 = layers.Conv3D(filters_num[2], kernel_size=(3, 3, 3),padding='same',kernel_regularizer=regularizers.l2(0.0001))(conv2_relu)  # filters_num = 32
    conv3_bn = layers.BatchNormalization()(conv3)
    conv3_relu = layers.Activation('relu')(conv3_bn)
    # conv3_relu = Dropout(0.5)(conv3_relu)
    # conv3_relu = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(conv3_relu)

    conv3_relu_reshape = layers.Reshape((conv3_relu.shape[1],conv3_relu.shape[2],conv3_relu.shape[3]*conv3_relu.shape[4]))(conv3_relu)
    conv3_relu_reshape = Dropout(0.5)(conv3_relu_reshape)
    ##################第二个尺度#########################
    # conv11 = layers.Conv3D(filters_num[0], kernel_size=(5, 5, 3), padding='same',
    #                       kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(input_layer_up)
    # conv11_bn = layers.BatchNormalization()(conv11)
    # conv11_relu = layers.Activation('relu')(conv11_bn)
    #
    # # conv2 = layers.Conv3D(filters_num[1], kernel_size=(3, 3, 5), padding='same')(conv1_relu)  # filters_num = 16
    # conv22 = layers.Conv3D(filters_num[1], kernel_size=(5, 5, 3), padding='same', kernel_initializer='he_normal',
    #                       kernel_regularizer=regularizers.l2(0.0001))(conv11_relu)  # filters_num = 16
    # conv22_bn = layers.BatchNormalization()(conv22)
    # conv22_relu = layers.Activation('relu')(conv22_bn)
    #
    # conv33 = layers.Conv3D(filters_num[2], kernel_size=(5, 5, 3), padding='same', kernel_initializer='he_normal',
    #                       kernel_regularizer=regularizers.l2(0.0001))(conv22_relu)  # filters_num = 32
    # conv33_bn = layers.BatchNormalization()(conv33)
    # conv33_relu = layers.Activation('relu')(conv33_bn)
    #
    # conv33_relu_reshape = layers.Reshape(
    #     (conv3_relu.shape[1], conv3_relu.shape[2], conv3_relu.shape[3] * conv3_relu.shape[4]))(conv33_relu)
    ####################################################
    # conv111 = layers.Conv3D(filters_num[0], kernel_size=(7, 7, 3), padding='same',
    #                        kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(input_layer_up)
    # conv111_bn = layers.BatchNormalization()(conv111)
    # conv111_relu = layers.Activation('relu')(conv111_bn)
    #
    # # conv2 = layers.Conv3D(filters_num[1], kernel_size=(3, 3, 5), padding='same')(conv1_relu)  # filters_num = 16
    # conv222 = layers.Conv3D(filters_num[1], kernel_size=(7, 7, 3), padding='same', kernel_initializer='he_normal',
    #                        kernel_regularizer=regularizers.l2(0.0001))(conv111_relu)  # filters_num = 16
    # conv222_bn = layers.BatchNormalization()(conv222)
    # conv222_relu = layers.Activation('relu')(conv222_bn)
    #
    # conv333 = layers.Conv3D(filters_num[2], kernel_size=(7, 7, 3), padding='same', kernel_initializer='he_normal',
    #                        kernel_regularizer=regularizers.l2(0.0001))(conv222_relu)  # filters_num = 32
    # conv333_bn = layers.BatchNormalization()(conv333)
    # conv333_relu = layers.Activation('relu')(conv333_bn)
    #
    # conv333_relu_reshape = layers.Reshape(
    #     (conv3_relu.shape[1], conv3_relu.shape[2], conv3_relu.shape[3] * conv3_relu.shape[4]))(conv333_relu)

    #################concatenate########################
    # conv33333_relu_reshape = Concatenate(axis=-1)([conv3_relu_reshape, conv33_relu_reshape])

    #########################################
    conv4 = layers.Conv2D(filters_num[3], kernel_size=(3, 3), padding='same',kernel_regularizer=regularizers.l2(0.0001))(conv3_relu_reshape)  # filters_num = 64
    conv4_bn = layers.BatchNormalization()(conv4)
    conv4_relu = layers.Activation('relu')(conv4_bn)
    # conv4_relu = Dropout(0.5)(conv4_relu)
    # conv4_relu = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv4_relu)
    # conv4_relu = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(conv4_relu)

    conv5 = layers.Conv2D(filters_num[4], kernel_size=(3, 3), padding='same',kernel_regularizer=regularizers.l2(0.0001))(conv4_relu)  # filters_num = **
    conv5_bn = layers.BatchNormalization()(conv5)
    conv5_relu = layers.Activation('relu')(conv5_bn)
    # conv5_relu = Dropout(0.5)(conv5_relu)
    # conv5_relu = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv5_relu)
    # conv5_relu = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(conv5_relu)
    # conv5_dpout = layers.Dropout(dropout_rate)(conv5)

    # conv5_reshape = layers.Reshape((conv5_dpout.shape[1],conv5_dpout.shape[2],conv5_dpout.shape[3]))(conv5_dpout)
    outputs2,outputs4 = Block_res(conv5_relu)

    return conv5,outputs2,outputs4



    # layer1 = build_resblock(filters_num[5], layer_dims[0])  # filters_num = 64
    # layer2 = build_resblock(filters_num[6], layer_dims[1], stride=2)    # filters_num = 128
    # layer3 = build_resblock(filters_num[7], layer_dims[2], stride=2)   # filters_num = 256
    # layer4 = build_resblock(filters_num[8], layer_dims[3], stride=2)  # filters_num = 512


































