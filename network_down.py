from tensorflow.keras.layers import Input,Conv3D,Reshape,Conv2D,Flatten,Dense,Dropout
from tensorflow.keras import regularizers,layers
import tensorflow as tf
from tensorflow.python.keras.layers import Concatenate


def CNN3D_2D(S,L,output_units,Xtrain,Ytrain,Xtest,Ytest,epochs):
    input_layer = Input((S,S,L,1))
    conv_layer1 = Conv3D(filters=8,kernel_size=(3,3,7),activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=16,kernel_size=(3,3,5),activation='relu')(conv_layer1)
    conv_layer3 = Conv3D(filters=32,kernel_size=(3,3,3),activation='relu')(conv_layer2)
    print(conv_layer3.shape)
    conv3d_shape = conv_layer3.shape
    # 维度变换 => 二维卷积
    conv_layer4 = Reshape((conv3d_shape[1],conv3d_shape[2],conv3d_shape[3]*conv3d_shape[4]))(conv_layer3)
    conv_layer5 = Conv2D(filters=64,kernel_size=(3,3),activation='relu')(conv_layer4)
    flatten_layer = Flatten()(conv_layer5)
    # 全卷积层
    dense_layer1 = Dense(units=256,activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=128,activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=output_units,activation='softmax')(dense_layer2)
    # model = Model(inputs=input_layer,outputs = output_layer)
    # model.summary()
    # adam = Adam(lr=0.001,decay=1e-06)
    # model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    # weight_path = r".\best_model.hdf5"
    # checkpoint = ModelCheckpoint(weight_path,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
    # # callbacks_list = [checkpoint]
    # csv_logger = CSVLogger('./LOG.csv', separator=';', append=True)
    # history = model.fit(x=Xtrain,y=Ytrain,batch_size=256,epochs=epochs,callbacks=[checkpoint,csv_logger],
    #                     validation_data=(Xtest, Ytest))
    # 4、下载训练好的权重
    # model.load_weights(weight_path)
    # model = load_model(weight_path)

    # return model

# def network_down(input_layer_down):
#     # inputs = Input(input_shape)
#
#     # conv_layer1 = Conv3D(filters=8,kernel_size=(3,3,7),activation='relu')(inputs)
#     conv_layer1 = Conv3D(filters=64,kernel_size=(3,3,7),strides=(2, 2, 2),activation='relu')(input_layer_down)
#     # conv_layer2 = Conv3D(filters=16,kernel_size=(3,3,5),activation='relu')(conv_layer1)
#     conv_layer2 = Conv3D(filters=128,kernel_size=(3,3,5),strides=(2, 2, 2),activation='relu')(conv_layer1)
#     conv_layer3 = Conv3D(filters=256,kernel_size=(3,3,3),strides=(1, 1, 1),activation='relu')(conv_layer2)
#     print(conv_layer3.shape)
#     outputs_down = Reshape((conv_layer3.shape[1],conv_layer3.shape[2],conv_layer3.shape[3]*conv_layer3.shape[4]))(conv_layer3)
#     print(outputs_down.shape)
#
#     return outputs_down


# def network_down(input_layer_down):
#     # inputs = Input(input_shape)
#
#     # conv_layer1 = Conv3D(filters=8,kernel_size=(3,3,7),activation='relu')(inputs)
#     conv_layer1 = Conv3D(filters=64,kernel_size=(3,3,3),strides=(2, 2, 2),activation='relu')(input_layer_down)
#     conv_layer1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1))(conv_layer1)
#     # conv_layer1 = tf.keras.layers.GlobalMaxPooling3D()(conv_layer1)
#
#     # conv_layer2 = Conv3D(filters=16,kernel_size=(3,3,5),activation='relu')(conv_layer1)
#     conv_layer2 = Conv3D(filters=128,kernel_size=(3,3,3),strides=(2, 2, 1),activation='relu')(conv_layer1)
#     conv_layer2 = tf.keras.layers.GlobalMaxPooling3D()(conv_layer2)
#     conv_layer3 = Conv3D(filters=256,kernel_size=(3,3,3),strides=(2, 2, 1),activation='relu')(conv_layer2)
#     # conv_layer3 = Conv3D(filters=256,kernel_size=(3,3,3),strides=(1, 1, 1),activation='relu',padding='same')(conv_layer2)
#     # conv_layer3 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(2, 2, 1), activation='relu')(conv_layer2)
#     print(conv_layer3.shape)
#     outputs_down = Reshape((conv_layer3.shape[1],conv_layer3.shape[2],conv_layer3.shape[3]*conv_layer3.shape[4]))(conv_layer3)
#     print(outputs_down.shape)
#
#     return outputs_down

# def network_down(input_layer_down):
#     # inputs = Input(input_shape)
#
#     # conv_layer1 = Conv3D(filters=8,kernel_size=(3,3,7),activation='relu')(inputs)
#     conv_layer1 = Conv3D(filters=8,kernel_size=(3,3,7),activation='relu',padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001))(input_layer_down)
#     # conv1_bn = layers.BatchNormalization()(conv_layer1)
#     # conv_layer1 = layers.Activation('relu')(conv1_bn)
#     # conv_layer1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1),padding='same')(conv_layer1)
#     # conv_layer1 = tf.keras.layers.GlobalMaxPooling3D()(conv_layer1)
#
#     # conv_layer2 = Conv3D(filters=16,kernel_size=(3,3,5),activation='relu')(conv_layer1)
#     conv_layer2 = Conv3D(filters=16,kernel_size=(3,3,5),activation='relu',padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001))(conv_layer1)
#     # conv_layer2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1),padding='same')(conv_layer2)
#     # conv_layer2 = tf.keras.layers.GlobalMaxPooling3D()(conv_layer2)
#     conv_layer3 = Conv3D(filters=32,kernel_size=(3,3,3),activation='relu',padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.001))(conv_layer2)
#     # conv_layer3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1), padding='same')(conv_layer3)
#     # conv_layer3 = Conv3D(filters=256,kernel_size=(3,3,3),strides=(1, 1, 1),activation='relu',padding='same')(conv_layer2)
#     # conv_layer3 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(2, 2, 1), activation='relu')(conv_layer2)
#     print(conv_layer3.shape)
#     outputs_down = Reshape((conv_layer3.shape[1],conv_layer3.shape[2],conv_layer3.shape[3]*conv_layer3.shape[4]))(conv_layer3)
#     print(outputs_down.shape)
#
#     return outputs_down


# def network_down(input_layer_down):
#     # inputs = Input(input_shape)
#
#     # conv_layer1 = Conv3D(filters=8,kernel_size=(3,3,7),activation='relu')(inputs)
#     conv_layer1 = Conv3D(filters=64,kernel_size=(3,3,3),padding='same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001))(input_layer_down)
#     # conv1_bn_down = layers.BatchNormalization()(conv_layer1)
#     # conv_layer1 = layers.Activation('relu')(conv1_bn_down)
#
#     # conv_layer1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1),padding='same')(conv_layer1)
#     # conv_layer1 = tf.keras.layers.GlobalMaxPooling3D()(conv_layer1)
#
#     # conv_layer2 = Conv3D(filters=16,kernel_size=(3,3,5),activation='relu')(conv_layer1)
#     conv_layer2 = Conv3D(filters=64,kernel_size=(3,3,3),padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001))(conv_layer1)
#     conv2_bn_down = layers.BatchNormalization()(conv_layer2)
#     conv_layer2 = layers.Activation('relu')(conv2_bn_down)
#
#     # conv_layer2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1),padding='same')(conv_layer2)
#     # conv_layer2 = tf.keras.layers.GlobalMaxPooling3D()(conv_layer2)
#     conv_layer3 = Conv3D(filters=64,kernel_size=(3,3,3),padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001))(conv_layer2)
#     conv3_bn_down = layers.BatchNormalization()(conv_layer3)
#     conv_layer3 = layers.Activation('relu')(conv3_bn_down)
#     # conv_layer3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1), padding='same')(conv_layer3)
#     # conv_layer3 = Conv3D(filters=256,kernel_size=(3,3,3),strides=(1, 1, 1),activation='relu',padding='same')(conv_layer2)
#     # conv_layer3 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(2, 2, 1), activation='relu')(conv_layer2)
#     print(conv_layer3.shape)
#     outputs_down = Reshape((conv_layer3.shape[1],conv_layer3.shape[2],conv_layer3.shape[3]*conv_layer3.shape[4]))(conv_layer3)
#     print(outputs_down.shape)
#
#     conv_layer4_down = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',activation='relu',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001)
#                           )(outputs_down)  # filters_num = 64
#     conv4_bn_down = layers.BatchNormalization()(conv_layer4_down)
#     outputs_down1 = layers.Activation('relu')(conv4_bn_down)
#
#     return outputs_down


##################################################################
def network_down(input_layer_down):
    # inputs = Input(input_shape)
    #######################第一次##########################
    # conv_layer1 = Conv3D(filters=8,kernel_size=(3,3,7),activation='relu')(inputs)
    # conv_layer1 = Conv3D(filters=64,kernel_size=(3,3,7),padding='same',kernel_initializer='he_normal',kernel_regularizer=regularizers.l2(0.0001))(input_layer_down)
    conv_layer1 = Conv3D(filters=64,kernel_size=(3,3,3),padding='same',kernel_regularizer=regularizers.l2(0.0001))(input_layer_down)
    conv1_bn_down = layers.BatchNormalization()(conv_layer1)
    conv_layer1 = layers.Activation('relu')(conv1_bn_down)
    # conv_layer1 = Dropout(0.5)(conv_layer1)
    # conv_layer1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(conv_layer1)

    # conv_layer1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1),padding='same')(conv_layer1)
    # conv_layer1 = tf.keras.layers.GlobalMaxPooling3D()(conv_layer1)

    # conv_layer2 = Conv3D(filters=16,kernel_size=(3,3,5),activation='relu')(conv_layer1)
    conv_layer2 = Conv3D(filters=128,kernel_size=(3,3,3),padding='same',kernel_regularizer=regularizers.l2(0.0001))(conv_layer1)
    conv2_bn_down = layers.BatchNormalization()(conv_layer2)
    conv_layer2 = layers.Activation('relu')(conv2_bn_down)
    # conv_layer2 = Dropout(0.5)(conv_layer2)
    # conv_layer2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(conv_layer2)

    # conv_layer2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1),padding='same')(conv_layer2)
    # conv_layer2 = tf.keras.layers.GlobalMaxPooling3D()(conv_layer2)
    conv_layer3 = Conv3D(filters=128,kernel_size=(3,3,3),padding='same',kernel_regularizer=regularizers.l2(0.0001))(conv_layer2)
    conv3_bn_down = layers.BatchNormalization()(conv_layer3)
    conv_layer3 = layers.Activation('relu')(conv3_bn_down)
    # conv_layer3 = Dropout(0.5)(conv_layer3)
    # conv_layer3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(1, 1, 1), padding='same')(conv_layer3)
    # conv_layer3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1), padding='same')(conv_layer3)
    # conv_layer3 = Conv3D(filters=256,kernel_size=(3,3,3),strides=(1, 1, 1),activation='relu',padding='same')(conv_layer2)
    # conv_layer3 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(2, 2, 1), activation='relu')(conv_layer2)
    print(conv_layer3.shape)
    outputs_down = Reshape((conv_layer3.shape[1],conv_layer3.shape[2],conv_layer3.shape[3]*conv_layer3.shape[4]))(conv_layer3)
    outputs_down = Dropout(0.5)(outputs_down)
    print(outputs_down.shape)

    ########################第二次#################################
    # conv_layer11 = Conv3D(filters=64, kernel_size=(3, 3, 9), padding='same', activation='relu',
    #                      kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0001))(input_layer_down)
    # conv11_bn_down = layers.BatchNormalization()(conv_layer11)
    # conv_layer11 = layers.Activation('relu')(conv11_bn_down)
    #
    # # conv_layer1 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1),padding='same')(conv_layer1)
    # # conv_layer1 = tf.keras.layers.GlobalMaxPooling3D()(conv_layer1)
    #
    # # conv_layer2 = Conv3D(filters=16,kernel_size=(3,3,5),activation='relu')(conv_layer1)
    # conv_layer22 = Conv3D(filters=64, kernel_size=(3, 3, 9), padding='same', kernel_initializer='he_normal',
    #                      kernel_regularizer=regularizers.l2(0.0001))(conv_layer11)
    # conv22_bn_down = layers.BatchNormalization()(conv_layer22)
    # conv_layer22 = layers.Activation('relu')(conv22_bn_down)
    #
    # # conv_layer2 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1),padding='same')(conv_layer2)
    # # conv_layer2 = tf.keras.layers.GlobalMaxPooling3D()(conv_layer2)
    # conv_layer33 = Conv3D(filters=64, kernel_size=(3, 3, 9), padding='same', kernel_initializer='he_normal',
    #                      kernel_regularizer=regularizers.l2(0.0001))(conv_layer22)
    # conv33_bn_down = layers.BatchNormalization()(conv_layer33)
    # conv_layer33 = layers.Activation('relu')(conv33_bn_down)
    # conv_layer3 = tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 1), padding='same')(conv_layer3)
    # conv_layer3 = Conv3D(filters=256,kernel_size=(3,3,3),strides=(1, 1, 1),activation='relu',padding='same')(conv_layer2)
    # conv_layer3 = Conv3D(filters=256, kernel_size=(3, 3, 3), strides=(2, 2, 1), activation='relu')(conv_layer2)
    # print(conv_layer33.shape)
    # outputs_down1 = Reshape((conv_layer33.shape[1], conv_layer33.shape[2], conv_layer33.shape[3] * conv_layer33.shape[4]))(
    #     conv_layer3)
    # print(outputs_down.shape)
    #####################合并###################################
    # outputs_down_con = Concatenate(axis=-1)([outputs_down, outputs_down1])


    #########################################################
    # activation='relu',

    conv_layer4_down = layers.Conv2D(filters=256, kernel_size=(7, 7), padding='same',kernel_regularizer=regularizers.l2(0.0001) #kernel_initializer='he_normal',
                          )(outputs_down)  # filters_num = 64
    conv4_bn_down = layers.BatchNormalization()(conv_layer4_down)
    outputs_down_mo = layers.Activation('relu')(conv4_bn_down)
    # outputs_down_mo = Dropout(0.5)(outputs_down_mo)
    # outputs_down_mo = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(outputs_down_mo)

    return outputs_down_mo
















