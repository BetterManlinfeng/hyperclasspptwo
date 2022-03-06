import tensorflow as tf
from tensorflow.keras import layers,regularizers,Sequential
from tensorflow.keras.layers import Dropout,Conv2D




def network_up(input_layer_up,filters_num,filters_num_resnet,layer_dims, stride=1):
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
    # outputs2,outputs4 = Block_res(conv5_relu)
    #
    # return conv5,outputs2,outputs4
    out2,out4 = ResNet_block(conv5_relu, filters_num_resnet, layer_dims, stride=stride)

    return conv5_relu,out2,out4


# filter_num
def ChannelAttention(inputs,filter_num,ratio=8):
    avg = layers.GlobalAveragePooling2D()(inputs)
    max = layers.GlobalMaxPooling2D()(inputs)
    avg = layers.Reshape((1, 1, avg.shape[1]))(avg)   # shape (None, 1, 1 feature)
    max = layers.Reshape((1, 1, max.shape[1]))(max)   # shape (None, 1, 1 feature)
    avg_out1 = Conv2D(filter_num//ratio, kernel_size=1, strides=1, padding='same',
                               kernel_regularizer=regularizers.l2(5e-4),
                               use_bias=True, activation=tf.nn.relu)(avg)
    avg_out = layers.Conv2D(filter_num, kernel_size=1, strides=1, padding='same',
                  kernel_regularizer=regularizers.l2(5e-4),
                  use_bias=True)(avg_out1)

    max_out1 = Conv2D(filter_num // ratio, kernel_size=1, strides=1, padding='same',
                      kernel_regularizer=regularizers.l2(5e-4),
                      use_bias=True, activation=tf.nn.relu)(max)
    max_out = layers.Conv2D(filter_num, kernel_size=1, strides=1, padding='same',
                            kernel_regularizer=regularizers.l2(5e-4),
                            use_bias=True)(max_out1)
    out = avg_out + max_out
    out = tf.nn.sigmoid(out)

    return out


def SpatialAttention(inputs,kernel_size=7):
    avg_out = tf.reduce_mean(inputs, axis=3)
    max_out = tf.reduce_max(inputs, axis=3)
    out = tf.stack([avg_out, max_out], axis=3) # 创建一个维度,拼接到一起concat。
    out = layers.Conv2D(1, kernel_size=kernel_size, strides=1, activation=tf.nn.sigmoid, padding='same', use_bias=False,
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(5e-4))(out)

    return out

def BasicBlock(inputs,filters_num_resnet,stride=1):
    conv1 = layers.Conv2D(filters_num_resnet, (3, 3), strides=stride, padding='same',
                               kernel_regularizer=regularizers.l2(0.0001))(inputs)  # kernel_initializer='he_normal',
    bn1 = layers.BatchNormalization()(conv1)
    relu1 = layers.Activation('relu')(bn1)

    conv2 = layers.Conv2D(filters_num_resnet, (3, 3), strides=1, padding='same',
                               kernel_regularizer=regularizers.l2(0.0001))(relu1)
    out = layers.BatchNormalization()(conv2)

    out = ChannelAttention(out,filters_num_resnet) * out
    out = SpatialAttention(out) * out

    if stride != 1:
        downsample = Sequential()
        downsample.add(layers.Conv2D(filters_num_resnet, (1, 1), strides=stride))
    else:
        downsample = lambda x: x

    identity = downsample(inputs)

    output = layers.add([out, identity])
    output = tf.nn.relu(output)


    return output


def build_resblock(inputs,filters_num_resnet, layer_dims, stride=1):

    res_blocks = Sequential()
    # may down sample
    res_blocks = BasicBlock(inputs,filters_num_resnet, stride)

    for _ in range(1, layer_dims):
        res_blocks = BasicBlock(res_blocks,filters_num_resnet, stride=1)

    return res_blocks

def ResNet_block(inputs,filters_num_resnet,layer_dims,stride=1):

    x1 = build_resblock(inputs,filters_num_resnet[0],layer_dims[0])
    x2 = build_resblock(x1,filters_num_resnet[1],  layer_dims[1], stride=stride)
    x3 = build_resblock(x2,filters_num_resnet[2],  layer_dims[2], stride=stride)
    x4 = build_resblock(x3,filters_num_resnet[3],  layer_dims[3], stride=stride)

    return x2,x4

















