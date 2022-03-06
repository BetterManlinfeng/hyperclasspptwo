import os
import scipy.io as sio
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam,SGD
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,cohen_kappa_score
from operator import truediv
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import *
import spectral
import gdal
import math
import random
import time
# from pp2_model import network_up,ResNet_block
from up_model import *
from network_down import network_down
from slic import slic_seg

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

np.random.seed(345)
# tf.set_random_seed(-1)
# 1 读影像
def loadData(name):
    # 获取当前目录下文件
    # data_path = os.path.join(os.getcwd(), 'Indian Pines')
    # 获取指定路径下文件
    # data_path = os.path.join(r'E:\2021-04\Hyperspectral Remote Sensing Scenes', 'Indian Pines')
    if name == 'IP':
        # 145*145*200;类别数目：16
        data_path = os.path.join(r'E:\2021-04\Hyperspectral Remote Sensing Scenes', 'Indian Pines')
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
        band_num = 15
        output_numclass = 16
    elif name == 'SA':
        # 512*217*204 ; 类别数目：16
        data_path = os.path.join(r'E:\2021-04\Hyperspectral Remote Sensing Scenes', 'Salinas')
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
        band_num = 15
        output_numclass = 16
    elif name == 'SSA':
        # 83*86*204 ; 类别数目:14或者6
        data_path = os.path.join(r'E:\2021-04\Hyperspectral Remote Sensing Scenes', 'Salinas-A')
        data = sio.loadmat(os.path.join(data_path, 'SalinasA_corrected.mat'))['salinasA_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'SalinasA_gt.mat'))['salinasA_gt']
        band_num = 30
        output_numclass = 6
    elif name == 'PU':
        # 610*340*103 ; 类别数目：9
        data_path = os.path.join(r'E:\2021-04\Hyperspectral Remote Sensing Scenes', 'Pavia University')
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
        band_num = 15
        output_numclass = 9
    elif name == 'PC':
        # 1096*715*102 ; 类别数目：9
        data_path = os.path.join(r'E:\2021-04\Hyperspectral Remote Sensing Scenes', 'Pavia Centre')
        data = sio.loadmat(os.path.join(data_path, 'Pavia.mat'))['pavia']
        labels = sio.loadmat(os.path.join(data_path, 'Pavia_gt.mat'))['pavia_gt']
        band_num = 15
        output_numclass = 9
    elif name == 'KSC':
        # 512*614*176 ；类别数目：13
        data_path = os.path.join(r'E:\2021-04\Hyperspectral Remote Sensing Scenes', 'Kennedy Space Center')
        data = sio.loadmat(os.path.join(data_path, 'KSC.mat'))['KSC']
        labels = sio.loadmat(os.path.join(data_path, 'KSC_gt.mat'))['KSC_gt']
        band_num = 30
        output_numclass = 13
    elif name == 'BWA':
        # 1476*256*145 ； 类别数目：14
        data_path = os.path.join(r'E:\2021-04\Hyperspectral Remote Sensing Scenes', 'Botswana')
        data = sio.loadmat(os.path.join(data_path, 'Botswana.mat'))['Botswana']
        labels = sio.loadmat(os.path.join(data_path, 'Botswana_gt.mat'))['Botswana_gt']
        band_num = 30
        output_numclass = 14

    return data, labels,band_num,output_numclass

def applyPCA(X,numComponents=75):
    newX = np.reshape(X,[-1,X.shape[2]])
    pca = PCA(n_components=numComponents,whiten=True)
    newX = pca.fit_transform((newX))
    newX = np.reshape(newX,(X.shape[0],X.shape[1],numComponents))
    return newX,pca

def normalize(data):  # 对数据标准化处理
    data_mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
    data_std = np.std(data, axis=(0, 1, 2), keepdims=True)
    normal_data = (data - data_mean) / data_std
    return normal_data, data_mean, data_std

def padWithZeros(X,margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:x_offset + X.shape[0], y_offset:y_offset + X.shape[1], :] = X
    return newX


def createImageCubes(x,y,windowSize=5,removeZerosLabels = True):
    margin = int((windowSize-1)/2)
    zerosPaddedX = padWithZeros(x,margin=margin)
    patchesData = np.zeros((x.shape[0] * x.shape[1], windowSize, windowSize, x.shape[2]),dtype=np.float32)
    patchesLabels = np.zeros(x.shape[0] * x.shape[1])
    patchIndex = 0
    for r in range(margin,zerosPaddedX.shape[0] - margin):
        for c in range(margin,zerosPaddedX.shape[1] - margin):
            patch = zerosPaddedX[r - margin: r + margin + 1,c - margin:c + margin+ 1]
            patchesData[patchIndex,:,:,:] = patch
            patchesLabels[patchIndex] = y[r-margin,c-margin]
            patchIndex += 1
    if removeZerosLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData,patchesLabels



# 2、训练数据，测试数据
def splitTrainTestSet(X,Y,testRatio,randomState=345):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=testRatio,random_state=randomState,
                     stratify=Y)
    return X_train,X_test,Y_train,Y_test


def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix,axis=1)
    each_acc = np.nan_to_num(truediv(list_diag,list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc,average_acc



def reports(model,Xtest,Ytest,test_x2, test_y2,name):
    Y_pred = model.predict([Xtest,test_x2])
    Y_pred = np.argmax(Y_pred,axis=1)
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
                        ,'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == 'SA':
        target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']
        # 'Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
        #                 'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
        #                 'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
        #                 'Vinyard_untrained','Vinyard_vertical_trellis']
    elif name == 'PU':
        target_names = ['Asphalt','Meadows','Gravel','Trees', 'Painted metal sheets','Bare Soil','Bitumen',
                        'Self-Blocking Bricks','Shadows']
    elif name == 'PC':
        target_names = ['Water', 'Trees', 'Asphalt', 'Self-Blocking Bricks', 'Bitumen', 'Tiles', 'Shadows',
                        'Meadows', 'Bare Soil']
    classification = classification_report(np.argmax(Ytest,axis=1),Y_pred,target_names=target_names)
    oa = accuracy_score(np.argmax(Ytest,axis=1),Y_pred)
    confusion = confusion_matrix(np.argmax(Ytest,axis=1),Y_pred)
    each_acc,aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(np.argmax(Ytest,axis=1),Y_pred)
    #########################
    # # adam = Adam(lr=0.001, decay=1e-06)
    # model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    sgd = SGD(learning_rate=0.01, decay=1e-06, clipvalue=1.)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    ########################
    score = model.evaluate((Xtest,test_x2),Ytest,batch_size=128)
    Test_loss = score[0]*100
    Test_accuracy = score[1] * 100

    return classification,confusion,Test_loss,Test_accuracy,oa*100,each_acc*100,aa*100,kappa*100

def Patch(data,height_index,width_index,patch_size):
    height_slice = slice(height_index,height_index + patch_size)
    width_slice = slice(width_index,width_index + patch_size)
    patch = data[height_slice,width_slice,:]
    return patch

# 预测取邻域块
def create_kernel(w):
    # 创建核函数，w为核函数的大小，如7*7
    kernel = np.zeros((w, w, 1, w * w),
                      np.float32)  # 做卷积运算的核函数必须是四维的，shape为 [filter_height, filter_width, in_channel, out_channels]；
    for i in range(w):
        for j in range(w):
            k_2d = np.zeros((w, w), np.float32)
            k_2d[i, j] = 1
            kernel[:, :, 0, i * w + j] = k_2d
    return kernel


def conv_kernel(img_band, kernel, w):  # 取一幅影像所有点的邻域，如7*7；
    pad_size = int((w - 1) / 2)  # 二维图像边缘要扩充的大小
    k = kernel.shape[1]
    img_band = img_band[np.newaxis, :, :, np.newaxis]  # 原影像单个波段来说，是二维矩阵，卷积操作中，原影像维度必须是4维，因为conv2d的参数都是四维的
    img_band = np.pad(img_band, ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='symmetric')
    # kernel = tf.convert_to_tensor(kernel)
    # img_band: object = tf.convert_to_tensor(img_band)  # 该函数将各种类型的Python对象转换为张量对象,将python的数据类型（列表和矩阵）转换成TensorFlow可用的tensor数据类型
    img_band = tf.nn.conv2d(img_band, kernel, strides=[1, 1, 1, 1], padding="VALID")  # 利用卷积功能获取所有像元的w*w的邻域；
    img_band = tf.reshape(tf.squeeze(img_band), [-1, k * k])  # tf.squeeze()函数用于从张量形状中移除大小为1的维度
    img_band = tf.reshape(img_band, [-1, k, k])
    return img_band.numpy()  #将tensor类型转成numpy类型


def gen_batch_data(img, w_size):
    '''函数功能：获取要分类的原始影像每个波段的w*w大小的邻域块值；'''
    img = img.astype(np.float32).transpose([2,0,1])
    kernel = create_kernel(w_size)  # 首先创建核函数，也就是filter滤波器
    all_batch_img = np.zeros([img.shape[0], img.shape[1] * img.shape[2], w_size, w_size], dtype=np.float32)
    for i in range(img.shape[0]):  # 遍历要分类影像的每个波段，对每个波段进行卷积，即取每个点的w*w邻域块值;
        all_batch_img[i] = conv_kernel(img[i, :, :], kernel, w_size)
    return all_batch_img.transpose([1, 2, 3, 0])  # 转换维度顺序


# hyppredict(model,X_db,windowSize,X_slic,windowSize_down,Y_labels)
def hyppredict(model,X_db,windowSize,img_slic,windowSize_down,Y_labels,data_mean_X, data_std_X,data_mean_Xslic, data_std_Xslic):
    """
    功能：对要分类的影像进行分类预测
    :param origin_ds: 要分类的影像数组
    :param out_ds: 分类后的影像标签
    :param model:  训练好的模型
    :param block_size: 分块大小；因为要对整副影像分块读取；
    :param train_mean: 训练样本的均值；
    :param train_std: 训练样本的方差；
    :param w: 邻域框的大小，如7*7；
    :return:
    """
    # Cols = img_dt.shape[0]  # 要分类影像的行列数目，栅格矩阵的列数，宽度方向，水平
    # Rows = img_dt.shape[1]  # 栅格矩阵的行数，高度上，竖直
    # x = int(Cols / block_size)  # 水平方向分了多少块
    # y = int(Rows / block_size)  # 竖直方向分了多少块
    image = X_db.copy()
    image_SLIC = img_slic.copy()

    image = image.astype(np.float32)
    image_SLIC = image_SLIC.astype(np.float32)
    all_batch_img = gen_batch_data(image, windowSize)  # 对整副要分类的影像取每个点的w*w邻域
    all_batch_img_slic = gen_batch_data(image_SLIC, windowSize_down)  # 对整副要分类的影像取每个点的w*w邻域
    all_batch_img = (all_batch_img - data_mean_X) / data_std_X  # 标准化
    all_batch_img_slic = (all_batch_img_slic - data_mean_Xslic) / data_std_Xslic  # 标准化
    predict_lb = model.predict((all_batch_img,all_batch_img_slic), batch_size=128)  # 预测
    predict_lb = np.argmax(predict_lb, axis=1)  # 返回的是索引值
    predict_lb = np.reshape(predict_lb, [image.shape[0], image.shape[1]])
    predict_lb[Y_labels == 0] = 0
    predict_lb[Y_labels != 0] += 1
    # scipy.misc.imsave('outfile.jpg', all_lb)
    return  predict_lb


def tif_gene(img_dt, predict_lb, save_path):
    spectral.save_rgb('IP_classpp2D3.jpg', predict_lb.astype(int), colors=spectral.spy_colors)
    origin_ds_RasterXSize = img_dt.shape[0]
    origin_ds_RasterYSize = img_dt.shape[1]
    # 5、为生成的图像做预备工作
    driver = gdal.GetDriverByName('GTiff')
    # origin_ds = gdal.Open(img_path)
    origin_ds = img_dt
    # origin_ds = origin_ds.ReadAsArray()
    # origin_ds = origin_ds.astype(np.float32)
    # out_ds = driver.Create(save_path, origin_ds_RasterXSize, origin_ds_RasterYSize, 1, gdal.GDT_Int32)
    out_ds = driver.Create(save_path, origin_ds_RasterYSize,origin_ds_RasterXSize, 1, gdal.GDT_Int32)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(predict_lb)
    out_band.FlushCache()
    out_band.ComputeStatistics(False)
    # make_raster(out_ds, all_lb)

    del out_ds

def make_raster(out_ds, block_data):
    # out_ds.SetProjection(origin_ds.GetProjection())
    # out_ds.SetGeoTransform(origin_ds.GetGeoTransform())
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(block_data)
    out_band.FlushCache()
    out_band.ComputeStatistics(False)

# def epoch_lr(epoch):
#     if epoch < 10:
#         lr = 1e-3 * 1.4  #
#     elif epoch < 20:
#         lr = 1e-3 * 1.3   # 0.7
#     elif epoch < 30:
#         lr = 1e-3 * 1.2
#     elif epoch < 40:
#         lr = 1e-3 * 1.1   # 0.7
#     elif epoch < 50:
#         lr = 1e-3 * 1.1
#     elif epoch < 60:
#         lr = 1e-3 * 1.0  # 0.2
#     elif epoch < 70:
#         lr = 1e-3 * 1.0  # 0.7
#     elif epoch < 80:
#         lr = 1e-3 * 0.9
#     elif epoch < 90:
#         lr = 1e-3 * 0.85  # 0.2
#     else:
#         lr = 1e-3 * 0.8
#
#     return lr


# def epoch_lr(epoch):
#     if epoch < 10:
#         lr = 1e-3 * 1.4  #
#     elif epoch < 20:
#         lr = 1e-3 * 1.3   # 0.7
#     elif epoch < 30:
#         lr = 1e-3 * 1.2
#     elif epoch < 40:
#         lr = 1e-3 * 1.1   # 0.7
#     elif epoch < 50:
#         lr = 1e-3 * 1.0
#     elif epoch < 60:
#         lr = 1e-3 * 0.9  # 0.2
#     elif epoch < 70:
#         lr = 1e-3 * 0.85  # 0.7
#     elif epoch < 80:
#         lr = 1e-3 * 0.75
#     elif epoch < 90:
#         lr = 1e-3 * 0.65  # 0.2
#     else:
#         lr = 1e-3 * 0.55
#
#     return lr

# def epoch_lr(epoch):
#     if epoch < 10:
#         lr = 1e-3 * 1.2  #
#     elif epoch < 20:
#         lr = 1e-3 * 1.1   # 0.7
#     elif epoch < 30:
#         lr = 1e-3 * 1.1
#     elif epoch < 40:
#         lr = 1e-3 * 1.0   # 0.7
#     elif epoch < 50:
#         lr = 1e-3 * 1.0
#     elif epoch < 60:
#         lr = 1e-3 * 0.9  # 0.2
#     elif epoch < 70:
#         lr = 1e-3 * 0.8  # 0.7
#     elif epoch < 80:
#         lr = 1e-3 * 0.7
#     elif epoch < 90:
#         lr = 1e-3 * 0.6  # 0.2
#     else:
#         lr = 1e-3 * 0.5
#
#     return lr

# def epoch_lr(epoch):
#     if epoch < 10:
#         lr = 1e-3 * 1.2
#     elif epoch < 20:
#         lr = 1e-3 * 1.15   # 0.7
#     elif epoch < 30:
#         lr = 1e-3 * 1.14
#     elif epoch < 40:
#         lr = 1e-3 * 1.13   # 0.7
#     elif epoch < 50:
#         lr = 1e-3 * 1.12
#     elif epoch < 60:
#         lr = 1e-3 * 1.11  # 0.2
#     elif epoch < 70:
#         lr = 1e-3 * 1.10  # 0.7
#     elif epoch < 80:
#         lr = 1e-3 * 1.09
#     elif epoch < 90:
#         lr = 1e-3 * 1.08  # 0.2
#     else:
#         lr = 1e-3 * 1.07
#
#     return lr


############对的###对SA数据集
# def epoch_lr(epoch):
#     if epoch < 20:
#         lr = 1e-4
#     elif epoch < 40:
#         lr = 1e-4   # 0.7
#     elif epoch < 60:
#         lr = 1e-4    # 0.7
#     elif epoch < 80:
#         lr = 1e-4
#     elif epoch < 90:
#         lr = 1e-4   # 0.2
#     else:
#         lr = 1e-4
#     return lr
###########################
def epoch_lr(epoch):
    if epoch < 15:
        lr = 1e-4
    elif epoch < 30:
        lr = 1e-4  # 0.7
    elif epoch < 45:
        lr = 1e-4    # 0.7
    elif epoch < 60:
        lr = 1e-4
    elif epoch < 75:
        lr = 1e-4   # 0.2
    else:
        lr = 1e-4 *0.5
    return lr



# def epoch_lr(epoch):
#     if epoch < 20:
#         lr = 1e-3
#     elif epoch < 70:
#         lr = 1e-3 * 0.9   # 0.7
#     elif epoch < 80:
#         lr = 1e-3 * 0.8
#     elif epoch < 90:
#         lr = 1e-3 * 0.7
#     elif epoch < 70:
#         lr = 1e-3 * 0.9   # 0.7
#     elif epoch < 80:
#         lr = 1e-3 * 0.8
#     elif epoch < 90:
#         lr = 1e-3 * 0.7  # 0.2
#     else:
#         lr = 0.0003
#
#     return lr




# def epoch_lr(epoch):
#     if epoch < 10:
#         lr = 1e-3
#     elif epoch < 20:
#         lr = 1e-3 * 0.95   # 0.9
#     elif epoch < 30:
#         lr = 1e-3 * 0.90  # 0.8
#     elif epoch < 40:
#         lr = 1e-3 * 0.85  # 0.7
#     elif epoch < 50:
#         lr = 1e-3 * 0.80   # 0.6
#     elif epoch < 60:
#         lr = 1e-3 * 0.75  #0.5
#     elif epoch < 70:
#         lr = 1e-3 * 0.70  # 0.4
#     elif epoch < 80:
#         lr = 1e-3 * 0.65  #0.3
#     elif epoch < 90:
#         lr = 1e-3 * 0.60  # 0.2
#     else:
#         lr = 1e-3 * 0.5  #0.1
#
#     return lr


def getSampleData(x1,x2,y,trainSample_ratio):
    train_num = x1.shape[0]
    lable_value = np.unique(y)
    train_x = np.zeros([1,x1.shape[1],x1.shape[2],x1.shape[3]],dtype=np.float)
    test_x = np.zeros([1,x1.shape[1],x1.shape[2],x1.shape[3]],dtype=np.float)
    train_y = np.ones([1],dtype=np.int)
    test_y = np.ones([1],dtype=np.int)

    train_x2 = np.zeros([1, x2.shape[1], x2.shape[2], x2.shape[3]], dtype=np.float)
    test_x2 = np.zeros([1, x2.shape[1], x2.shape[2], x2.shape[3]], dtype=np.float)
    train_y2 = np.ones([1], dtype=np.int)
    test_y2 = np.ones([1], dtype=np.int)
    for i in lable_value:
        lable_i_sum = np.sum(y == i)  # 第i类样本的个数
        x_i = x1[y == i]              # 把第i类样本的x取出来
        y_i = y[y == i]
        x2_i = x2[y == i]
        rdm_num = np.arange(0, lable_i_sum, dtype=np.int32)  # 生成0-train_num的向量
        np.random.shuffle(rdm_num)  # 打乱样本顺序

        train_xi = x_i[rdm_num][0:int(lable_i_sum * trainSample_ratio)]
        test_xi = x_i[rdm_num][int(lable_i_sum * trainSample_ratio):]
        train_x2i = x2_i[rdm_num][0:int(lable_i_sum * trainSample_ratio)]
        test_x2i = x2_i[rdm_num][int(lable_i_sum * trainSample_ratio):]
        # 合并第一幅
        train_x = np.concatenate([train_x,train_xi],axis=0)
        test_x = np.concatenate([test_x,test_xi],axis=0)
        # slic第二幅
        train_x2 = np.concatenate([train_x2, train_x2i], axis=0)
        test_x2 = np.concatenate([test_x2, test_x2i], axis=0)
        #第一幅图像生成标签
        train_i = np.ones([int(lable_i_sum * trainSample_ratio)],dtype=np.int) * int(i)
        train_y = np.concatenate([train_y,train_i],axis=0)
        test_i = np.ones([int(lable_i_sum - int(lable_i_sum * trainSample_ratio))],dtype=np.int) * int(i)
        test_y = np.concatenate([test_y,test_i],axis=0)
        # 第二幅图像生成标签
        # train_y2 = train_y.copy()
        # test_y2 = test_y.copy()
    # 第一幅
    train_x = train_x[1:]
    train_y = train_y[1:]
    test_x = test_x[1:]
    test_y = test_y[1:]

    # 第二幅slic
    train_x2 = train_x2[1:]
    train_y2 = train_y.copy()
    test_x2 = test_x2[1:]
    test_y2 = test_y.copy()


    return train_x,train_y,test_x,test_y,train_x2,train_y2,test_x2,test_y2




def hsidata_pre(dataset,windowSize,windowSize_down,trainSample_ratio):
    # 1、下载数据
    X_data, Y_data, band_num, output_numclass = loadData(dataset)
    Y_labels = Y_data.copy()
    print(X_data.shape, Y_data.shape)
    # 2、执行PCA变换
    X_pca, pca = applyPCA(X_data, numComponents = band_num)
    X_slic, _ = applyPCA(X_data, numComponents=3)
    X_db = X_pca.copy()
    print(X_data.shape)
    ############执行SLIC分割##############
    img_slic = slic_seg(X_slic, seg_scale)
    ##########################
    # 3、对影像数组取邻域块
    X, Y = createImageCubes(X_pca, Y_data, windowSize = windowSize)
    print(X.shape, Y.shape)
    # X, data_mean_X, data_std_X = normalize(X)
    image_slic, Y_slic = createImageCubes(img_slic, Y_data, windowSize=windowSize_down)
    print(image_slic.shape, Y_slic.shape)
    # image_slic, data_mean_slic, data_std_slic = normalize(image_slic)
    ###################################
    # train_num = X.shape[0]
    # rdm_num = np.arange(0, train_num, dtype=np.int32)  # 生成0-train_num的向量
    # np.random.shuffle(rdm_num)  # 打乱样本顺序
    # X = X[rdm_num]
    # Y = Y[rdm_num]
    # image_slic = image_slic[rdm_num]
    #################
    train_x,train_y,test_x,test_y,train_x2,train_y2,test_x2,test_y2 = getSampleData(X,image_slic, Y,trainSample_ratio)
    # 归一化
    train_x, data_mean_X, data_std_X = normalize(train_x)
    test_x = (test_x - data_mean_X) / data_std_X
    train_x2, data_mean_Xslic, data_std_Xslic = normalize(train_x2)
    test_x2 = (test_x2 - data_mean_Xslic) / data_std_Xslic
    #
    train_x = train_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    train_x2 = train_x2.astype(np.float32)
    test_x2 = test_x2.astype(np.float32)
    # 4、训练集和测试集
    # Xtrain, Xtest, Ytrain, Ytest = splitTrainTestSet(X, Y, testRatio)
    # print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
    # # 5、训练集维度转换，ont_hot编码
    # # Xtrain = Xtrain.reshape(-1, windowSize, windowSize, K, 1)
    # print('训练集Xtrain：', Xtrain.shape)
    # # Ytrain = to_categorical(Ytrain)
    # print('训练集Ytrain：', Ytrain.shape)
    # # Xtest = Xtest.reshape(-1, windowSize, windowSize, K, 1)
    # print('测试集Xtest：', Xtest.shape)
    # # Ytest = to_categorical(Ytest)
    # print('测试集Ytest：', Ytest.shape)
    # 标准化处理
    # Xtrain, data_mean, data_std = normalize(Xtrain)
    # Xtest = (Xtest - data_mean) / data_std
    # Ytrain = tf.cast(Ytrain,dtype=tf.int32)
    # Ytest = tf.cast(Ytest,dtype=tf.int32)
    # Xtest_indexacc = Xtest.copy()
    # Ytest_indexacc = Ytest.copy()

    # 训练测试数据组合
    # train_db = (Xtrain, Ytrain)
    # test_db = (Xtest, Ytest)


    # sample = next(iter(train_db))
    # print('sample:', sample[0].shape, sample[1].shape,
    #       tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))
    return train_x,train_y,test_x,test_y,train_x2,train_y2,test_x2,test_y2,band_num,output_numclass,X_db,img_slic,Y_labels,data_mean_X, data_std_X,data_mean_Xslic, data_std_Xslic

    # return Xtrain, Ytrain, Xtest, Ytest,band_num,output_numclass,tif_Y,data_mean, data_std

def db_tt(Xtrain, Ytrain, Xtest, Ytest,batch_size):

    Ytrain = tf.cast(Ytrain,dtype=tf.int32)
    Ytest = tf.cast(Ytest,dtype=tf.int32)

    train_db = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
    train_db = train_db.shuffle(100).batch(batch_size)
    test_db = tf.data.Dataset.from_tensor_slices((Xtest, Ytest))
    test_db = test_db.batch(batch_size)


    return train_db,test_db



def train_model_zeros(model,weight_path, Xtrain_up, Ytrain, Xtest_up, Ytest,train_x2_up, train_y2, test_x2_up, test_y2, batch_size, epoch=10):
    # (x_train, y_train)= train_db
    # (X_val, Y_val) = test_db
    # x_train = Xtrain
    # X_val = Xtest
    # Ytrain = tf.one_hot(Ytrain,depth=output_numclass)
    # Ytest = tf.one_hot(Ytest,depth=output_numclass)

    Ytrain = to_categorical(Ytrain)
    Ytest = to_categorical(Ytest)
    train_y2 = to_categorical(train_y2)
    test_y2 = to_categorical(test_y2)

    # y_train = tf.cast(y_train,dtype=tf.int32)
    # Y_val = tf.cast(Y_val,dtype=tf.int32)

    # model = resnet18()
    # model.build(input_shape=(None, windowSize, windowSize, band_num))
    # model.summary()
    # # optimizer = optimizers.Adam(lr=1e-3)
    # # adam = Adam(lr=1e-4,decay=1e-06)
    ################################
    # adam = Adam(lr=0.001,decay=1e-06,clipvalue=15.)
    # model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    # sgd = SGD(learning_rate=0.001, decay=1e-06, clipvalue=15.,momentum=0.9)
    sgd = SGD(learning_rate=0.01, decay=1e-06, clipvalue=15., momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    ##############################

    checkpoint = ModelCheckpoint(weight_path, monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger('./LOG.csv', separator=';', append=True)
    model.fit(x=[Xtrain_up,train_x2_up],y=Ytrain, epochs=epoch, batch_size= batch_size,verbose=2,
              callbacks=[checkpoint, csv_logger],
              validation_data=([Xtest_up,test_x2_up], Ytest))

    #  model.fit()中的batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。

#   train_model(model, Xtrain, Ytrain, Xtest, Ytest,train_x2, train_y2, test_x2, test_y2, output_numclass, epochs,batch_size)
def train_model(model,Xtrain, Ytrain, Xtest, Ytest, train_x2, train_y2, test_x2, test_y2,output_numclass,epochs,batch_size):
    train_db,test_db = db_tt(Xtrain, Ytrain, Xtest, Ytest, batch_size)
    train_db2, test_db2 = db_tt(train_x2, train_y2, test_x2, test_y2, batch_size)
    train_step_num = math.ceil(1.0 * Xtrain.shape[0] / batch_size)
    test_step_num = math.ceil(1.0* Xtest.shape[0] / batch_size)
    # 构建模型
    # model = resnet18()
    # model.build(input_shape=(None, windowSize, windowSize, band_num))
    # model.summary()
    ##########
    # optimizer = Adam(lr=1e-3)

    train_acc = np.zeros([epochs,1],dtype=np.float32)
    train_loss = np.zeros([epochs,1],dtype=np.float32)
    test_acc = np.zeros([epochs,1],dtype=np.float32)
    test_loss = np.zeros([epochs,1],dtype=np.float32)

    flag_acc = 0

    for epoch in range(epochs):
        lr = epoch_lr(epoch)
        ######################
        # optimizer = Adam(lr=lr)
        optimizer = SGD(learning_rate=0.001,decay=1.e-6,momentum=0.9,clipvalue=1.)
        # optimizer.clipvalue()
        ########################

        train_step = 0
        train_num = 0
        train_loss_epoch = 0
        train_correct_temp = 0
        train_acc_tmp = 0
        train_correct_epoch = 0

        for step, ((x,y),(x2,y2)) in enumerate(zip(train_db,train_db2)):

            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] => [b, 100]
                logits = model([x,x2])
                # [b] => [b, 100]
                y_onehot = tf.one_hot(y, depth = output_numclass)  # 或者下一句进行 one_hot
                # y_onehot = to_categorical(y)
                # compute loss
                # tf.losses.
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

                # 训练阶段：精度计算
                pred = tf.cast(tf.argmax(tf.nn.softmax(logits,axis=1),axis=1),dtype=tf.int32)
                train_correct_step = tf.reduce_sum(tf.cast(tf.equal(pred,y),dtype=tf.int32))

            train_num += x.shape[0]
            train_correct_temp += train_correct_step

            train_acc_epoch = train_correct_temp / train_num
            # train_acc_tmp += train_acc_epoch

            train_loss_epoch += loss  # 每个step累加的损失，不除以step次数嘛


            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 50 == 0:
                print(epoch, step, 'loss:', float(loss))
            # print('train_step',train_step)
            train_step += 1

        train_loss[epoch] = train_loss_epoch.numpy() / train_step_num
        train_acc[epoch] = train_acc_epoch.numpy()
        test_step = 0

        total_num = 0
        total_correct = 0
        test_loss_tmp = 0
        for (x, y),(x2,y2) in zip(test_db,test_db2):
            logits = model([x,x2])
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)
            # 测试阶段精度计算结束

            #测试阶段损失计算开始：
            y_onehot = tf.one_hot(y,depth=output_numclass)
            test_loss_epoch = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True))
            # 测试损失累加
            test_loss_tmp += test_loss_epoch

            # print('test_step',test_step)
            test_step += 1


        #测试精度前面已经累加过，此处进行计算即可；
        test_acc_tmp = total_correct / total_num

        # test_loss_tmp += test_loss_epoch

        test_loss[epoch] = test_loss_tmp.numpy() / test_step_num  # 应该除以step个数吧
        test_acc[epoch] = test_acc_tmp

        print(epoch, 'test_acc:', test_acc_tmp)
        if flag_acc < (test_acc_tmp / test_step_num):
            flag_acc = (test_acc_tmp / test_step_num)
            model.save_weights(weight_path)

    result = np.concatenate([train_acc,train_loss,test_acc,test_loss],axis=1)
    np.savetxt(r'result.csv',result,delimiter=',')




    #训练和验证的精度、损失曲线数据；混淆矩阵，


# def slic_seg(image,seg_scale):
# 	# seg_num = len(seg_scale)
# 	img_slic = np.zeros((1, image.shape[0], image.shape[1]), dtype=np.float)
# 	for numSegments in seg_scale:
# 		# 应用slic算法并获取分割结果
# 		segments = slic(image, n_segments=numSegments, sigma=5)
# 		# img_slic[seg_num] = segments
# 		segments = segments[np.newaxis,:]
# 		img_slic = np.concatenate([img_slic, segments],axis=0)
# 	img_slic = img_slic[1:]
# 	return img_slic


def Index_cal(dataset, model, Xtest, Ytest, test_x2, test_y2):
    # (Xtest, Ytest) = test_db
    # 训练集 训练模型
    # model = CNN3D_2D(S,L,output_units,Xtrain,Ytrain,Xtest,Ytest,epochs)
    # 测试集调用模型
    Y_pred_test = model.predict([Xtest,test_x2])
    Y_pred_test = np.argmax(Y_pred_test,axis=1)
    Ytest = to_categorical(Ytest)
    # 测试结果报告
    classification = classification_report(np.argmax(Ytest,axis=1),Y_pred_test)
    print(classification)
    classification, confusion, Test_loss, Test_accuracy, oa, each_acc , aa , kappa  = reports(model,Xtest,Ytest,test_x2, test_y2,dataset)
    classification = str(classification)
    confusion = str(confusion)
    file_name = 'classification_report.txt'
    with open(file_name,'w') as f:
        f.write('{} Test loss (%)'.format(Test_loss))
        f.write('\n')
        f.write('{} Test_accuracy(%)'.format(Test_accuracy))
        f.write('\n')
        f.write('\n')
        f.write('{} Kappa accuracy'.format(kappa))
        f.write('\n')
        f.write('{} Overall acccuracy (%)'.format(oa))
        f.write('\n')
        f.write('\n')
        f.write('{} Average accuracy (%)'.format(aa))
        f.write('\n')
        f.write('{} '.format(classification))
        f.write('\n')
        f.write('{} '.format(confusion))
        f.write('\n')
        f.write('{} Average accuracy (%)'.format(each_acc))
    # 对整幅影像进行预测
    # predict(dataset, model, windowSize, K)

def AA_andEachClassAccuracy(confusion_matrix):
    counter = confusion_matrix.shape[0]
    list_diag = np.diag(confusion_matrix)  # 输出对角线上的元素
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))  # list_diag/list_raw_sum C内核
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def reports_two(y_test,y_pred):
    y_test = np.reshape(y_test,[y_test.shape[0] * y_test.shape[1]])
    y_pred = np.reshape(y_pred, [y_pred.shape[0] * y_pred.shape[1]])

    y_test = tf.cast(y_test, dtype=tf.int32)
    y_pred = tf.cast(y_pred, dtype=tf.int32)

    # y_test.reshape([y_test.shape[0] * y_test.shape[1]], dtype=np.int64)
    # y_pred.reshape([y_pred.shape[0] * y_pred.shape[1]], dtype=np.int64)
    classification = classification_report(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    evaluate = [oa, aa, kappa]
    each_acc = list(np.round(each_acc * 100, 5))  # 百分制，取两位有效
    return classification, confusion, evaluate, each_acc


# 构建双分支复合模型
# def dualBranchNetwork(outputs1,outputs2,cls_num):
#     # outputs1 = network_up(inputs_layer1)
#     # outputs2 = network_down(inputs_layer2)
#     x = Concatenate(axis=-1)([outputs1,outputs2])
#     x_f = Flatten()(x)
#     f1 = Dense(128, activation='relu')(x_f)
#     f2 = Dense(64, activation='relu')(f1)
#     f3 = Dense(32, activation='relu')(f2)
#     outputs = Dense(cls_num, activation='softmax')(f3)
#     model = Model(inputs=[outputs1,outputs2],outputs=outputs)
#
#     return model

def build_resblock(filter_num, blocks, layer_res1,layer_res2):
    res_blocks = Sequential()
    # may down sample
    res_blocks.add(layer_res1)

    for _ in range(1, blocks):
        res_blocks.add(layer_res2)

    return res_blocks


def main(dataset,windowSize,windowSize_down,batch_size,weight_path,save_path,seg_scale,trainSample_ratio,epochs=10):
    Xtrain, Ytrain, Xtest, Ytest,train_x2, train_y2, test_x2, test_y2,band_num,output_numclass,X_db,img_slic,Y_labels,data_mean_X, data_std_X,data_mean_Xslic, data_std_Xslic = hsidata_pre(dataset, windowSize,windowSize_down,trainSample_ratio)
    # train_x, train_y, test_x, test_y, train_x2, train_y2, test_x2, test_y2
    # model = train_model(train_db, test_db, windowSize, band_num, output_numclass, epoch)
    ############执行SLIC分割##############
    # image_slic = slic_seg(X_slic, seg_scale)
    ##########################
    ###########样本维度转换成3Dcnn的输入维度
    # Xtrain_up = Xtrain.reshape(-1, windowSize, windowSize, band_num, 1)
    # print('训练集Xtrain：', Xtrain_up.shape)
    # Xtest_up = Xtest.reshape(-1, windowSize, windowSize, band_num, 1)
    # print('测试集Xtest：', Xtest_up.shape)
    # # 第二分支样本维度转换
    # train_x2_up = train_x2.reshape(-1, windowSize_down, windowSize_down, train_x2.shape[3], 1)
    # print('训练集Xtrain：', train_x2_up.shape)
    # test_x2_up = test_x2.reshape(-1, windowSize_down, windowSize_down, test_x2.shape[3], 1)
    # print('测试集Xtest：', test_x2_up.shape)
    ###########模型实例化
    # 双分支上模型
    # filters_num_conv = [8, 16, 32, 64, 128, 64, 128, 256, 512]
    dropout_rate = 0.2
    filters_num = [64,64,64,128,256]   #8,16,32,64,128   #[8,16,32,64,256]
    filters_num_resnet = [256,256,256,256]   #128,128,256,512  #[256,256,256,256]
    layer_dims= [2,2,2,2]
    input_shape_up = (windowSize, windowSize, band_num, 1)
    input_layer_up = Input(input_shape_up)
    # Block_res = ResNet_block(layer_dims,filters_num_resnet)
    # conv5,outputs2,outputs4 = network_up(input_layer_up, filters_num, Block_res)
    conv5,outputs2,outputs4 = network_up(input_layer_up, filters_num, filters_num_resnet, layer_dims, stride=1)
    # outputs_up.summary()
    # network_up = pp2_model(filters_num,layer_dims,output_numclass)
    # model_up.summary()
    # 双分支下模型
    input_shape_down = (windowSize_down, windowSize_down, len(seg_scale),1)
    # network_down = UNet(input_shape_down, output_numclass)
    input_layer_down = Input(input_shape_down)
    outputs_down = network_down(input_layer_down)
    # outputs_down.summary()
    # model_down.summary()
    #####网络模型组合######
    x = Concatenate(axis=-1)([conv5,outputs2,outputs4,outputs_down])
    x_f = Flatten()(x)
    x_f = Dropout(0.5)(x_f)
    f1 = Dense(1024, activation='relu')(x_f) # 128;1024
    # f1 = Dropout(0.2)(f1)
    # f2 = Dense(256, activation='relu')(f1)     # 64;512
    f1 = Dense(512, activation='relu')(f1)     # 32;128
    # drop1 = Dropout(0.2)(f1)
    outputs = Dense(output_numclass, activation='softmax')(f1)
    model = Model(inputs=[input_layer_up, input_layer_down], outputs=outputs)



    # model = dualBranchNetwork(input_shape_up, input_shape_down, outputs_up, outputs_down,output_numclass)
    model.summary()

    ##############
    # model = resnet34()
    # model.build(input_shape=(None, windowSize, windowSize, band_num))
    # model.summary()
    # optimizer = optimizers.Adam(lr=1e-3)
    # adam = Adam(lr=1e-4,decay=1e-06)
    # adam = optimizers.Adam(lr=0.001,decay=1e-06)
    # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    ####pp2模型#
    train_t1 = time.time()
    # train_model(model, Xtrain, Ytrain, Xtest, Ytest,train_x2, train_y2, test_x2, test_y2, output_numclass, epochs,batch_size)
    train_model_zeros(model, weight_path, Xtrain, Ytrain, Xtest, Ytest, train_x2, train_y2, test_x2, test_y2,
                      batch_size, epoch=epochs)
    train_t2 = time.time()
    train_time = train_t2 - train_t1
    print('训练所用时间：', train_time)
    ####
    # train_model_zeros(model,weight_path, Xtrain, Ytrain, Xtest, Ytest,train_x2, train_y2, test_x2, test_y2,batch_size, epoch=epochs)
    ############


    # from functools import partial
    # new_fun = partial(calloss,z=9,k=11)
    # new_fun.__name__ =  'calloss'
    # 4、下载训练好的权重
    model.load_weights(weight_path)
    # model = load_model(weight_path)
    Index_cal(dataset, model, Xtest, Ytest,test_x2, test_y2)
    test_t1 = time.time()
    predict_lb = hyppredict(model,X_db,windowSize,img_slic,windowSize_down,Y_labels,data_mean_X, data_std_X,data_mean_Xslic, data_std_Xslic)
    test_t2 = time.time()
    test_time = test_t2 - test_t1
    print('预测所用时间：', test_time)
    with open("timeuse.txt", 'a') as ft:
        ft.write('{} train time (%)'.format(train_time))
        ft.write('\n')
        ft.write('{} test time (%)'.format(test_time))
    classification, confusion, evaluate, each_acc = reports_two(Y_labels,predict_lb)
    with open("acc_oa_aa.txt", 'a') as ft:
        ft.write('{} classification (%)'.format(classification))
        ft.write('\n')
        ft.write('{} confusion (%)'.format(confusion))
        ft.write('\n')
        ft.write('{} evaluate:oa, aa, kappa (%)'.format(evaluate))
        ft.write('\n')
        ft.write('{} each_acc (%)'.format(each_acc))
    tif_gene(X_db, predict_lb, save_path)


def calloss(x,y,z=0,k=0):
    pass

if __name__ == '__main__':
    dataset = 'IP'   # 数据集名称
    windowSize = 11  # 邻域窗口的大小
    windowSize_down = 11  # 邻域窗口的大小
    trainSample_ratio = 0.3
    # testRatio = 0.7  # 测试样本的比例
    batch_size = 32 # 每批次样本个数
    epochs = 200    # 数据集迭代次数
    weight_path = "IPbest_modelPP2D3.h5"
    save_path = r'IP_classPP2D3.tif'  # 预测标签生成的路径
    #########IPIPIP#######
    # IP数据集用
    # seg_scale = (50, 80, 110, 140, 170, 200, 230, 260, 290, 220, 250, 280, 310, 340, 370, 400)
    # seg_scale = (15,25,35,45,55,65,75,85,95,115,135,155,175)  # IP数据集用
    # seg_scale = (50, 100, 150, 200, 250, 300, 350,  400)  ##200以下不太行，最小要设置成200，最大400差不多
    seg_scale = (200,250,300,350,400,450,500)
    #### SA数据集用########
    # seg_scale = [10,40,70,100,130,160,190,220,250,280,310] #  不合适
    # seg_scale = [100, 160, 220, 280, 240, 300, 360, 420,480,540,600] # 第一次错了
    # seg_scale = [100, 160, 220, 280, 340, 400, 460, 520, 580, 640, 700] ##第三次还行
    # seg_scale = [340, 400, 460, 520, 580, 640, 700]
    # seg_scale = [300, 400, 500, 600, 700,800,900,1000,1100]
    ######### seg_scale = [500, 600, 700, 800, 900, 1000, 1100] 用用用
    # seg_scale = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]  # 第二次

    main(dataset,windowSize,windowSize_down,batch_size,weight_path,save_path,seg_scale,trainSample_ratio,epochs=epochs)



