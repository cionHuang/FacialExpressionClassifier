# 导入程序所需要的库和包
import numpy as np # numpy库 
import argparse # 命令行工具
import matplotlib.pyplot as plt # 绘图工具
import cv2 # opencv
from tensorflow.keras.models import Sequential # tensorflow的神经网络工具
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import os # 本地文件操作

# 新建一个存放识别结果的文件夹result
result_folder = 'result'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

# 命令行操作 
ap = argparse.ArgumentParser() # 创建一个解析命令行参数的对象
ap.add_argument("--mode",help="train/display") # 添加一个命令行参数
mode = ap.parse_args().mode # 解析命令行参数，并获取参数的值

# 绘制模型训练的准确度和损失函数曲线
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # 绘制准确度曲线
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1, len(model_history.history['accuracy']) / 10))
    axs[0].legend(['train', 'val'], loc='best')
    # 绘制损失函数曲线
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1, len(model_history.history['loss']) / 10))
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()

# 定义模型训练的数据格式与基本参数
train_dir = 'data/train' # 训练集地址
val_dir = 'data/test' # 测试集地址

num_train = 28709 # 训练集数量
num_val = 7178 # 测试集数量
batch_size = 256 # 每批次的样本数量
num_epoch = 100 # 迭代次数

train_datagen = ImageDataGenerator(rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest') 
val_datagen = ImageDataGenerator(rescale=1./255) # 图像预处理，将原始灰度图转化为0-1之间的值

# 训练集的初始化
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48), # 像素大小
        batch_size=batch_size, # 批次样本数量
        color_mode="grayscale", # 灰度图模式
        class_mode='categorical') # 分类标签

# 测试集的初始化
validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical') 

# 卷积神经网络的实例化
model = Sequential() # Keras中的Sequential神经网络模型

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1))) # 输入层与第一个卷积层，输入为48x48的灰度图，32个3x3大小的卷积核，激活函数为relu
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # 第二个卷积层，64个3x3的卷积核，激活函数为relu
model.add(MaxPooling2D(pool_size=(2, 2))) # 最大池化层，2x2的池化窗口，减小特征图大小
model.add(Dropout(0.25)) # 随机停止一定数量的卷积，提高泛化能力，丢弃率为25%

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu')) # 第三卷积层，128个3x3大小的卷积核，激活函数relu
model.add(MaxPooling2D(pool_size=(2, 2))) # 最大池化层，2x2的池化窗口，减小特征图大小
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu')) # 第三卷积层，128个3x3大小的卷积核，激活函数relu
model.add(MaxPooling2D(pool_size=(2, 2))) # 最大池化层，2x2的池化窗口，减小特征图大小
model.add(Dropout(0.25)) # 随机停止一定数量的卷积，提高泛化能力，丢弃率为25%

model.add(Flatten()) # 将卷积层的输出拉平为一维向量，为全连接层做准备
model.add(Dense(1024, activation='relu')) # 1024 个神经元，激活函数使用 ReLU
model.add(Dropout(0.5)) # 使用 50% 的丢弃率
model.add(Dense(7, activation='softmax')) # 7 个神经元，对应于数据集中的 7 个情感类别，激活函数使用 softmax

# 命令行对应操作
# 新模型训练选项
if mode == "train":
    initial_learning_rate = 0.0001 # 初始学习率
    lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True) # 创建一个指数衰减学习率的调度器。这个调度器会在每个 decay_steps 步之后将学习率乘以 decay_rate
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr_schedule), metrics=['accuracy']) # 编译模型，配置训练过程。在这里，使用分类交叉熵作为损失函数，Adam 优化器，并使用前面创建的学习率调度器
   
   # 定义 EarlyStopping 回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 训练
    model_info = model.fit_generator(
            train_generator, # 训练集图像
            steps_per_epoch=num_train // batch_size, # 每次迭代的步数
            epochs=num_epoch, # 迭代次数
            validation_data=validation_generator, # 测试集图像
            validation_steps=num_val // batch_size, # 每次迭代步数
            
            callbacks=[early_stopping])  # 将 EarlyStopping 回调加入 callbacks 列表中

    plot_model_history(model_info) # 存储每次训练的准确度与损失函数值，方便绘图
    model.save_weights('model.h5') # 保存模型的权重
# 使用模型进行实时预测
elif mode == "display":
    model.load_weights('model.h5') # 加载预测模型

    # 存储表情预期对应编码的字典
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # 视频输入
    cap = cv2.VideoCapture(0)
    frame_count = 0
    while True:
        ret, frame = cap.read() # 帧读取
        inputframe = frame.copy() # 保存一个原始帧
        if not ret:
            break
        # 创建一个人脸检测器对象，使用的是哈尔级联分类器（Haar Cascade Classifier），用于检测图像中的人脸
        facecasc = cv2.CascadeClassifier(
            'haarcascade_frontalface_alt.xml') 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 将视频流转化为灰度图
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=5) # 调用之前的人脸检测对象，压缩率设置为1.1，检测阈值为5，并将人脸的像素坐标与大小信息保存在faces变量中

        for (x, y, w, h) in faces: # 开始遍历之前存储的面部信息
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2) # 为脸部画框
            roi_gray = gray[y:y + h, x:x + w] # 取出脸部区域，作为ROI
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0) # 将ROI转化为48x48大小，与训练集像素大小匹配
            prediction = model.predict(cropped_img) # 使用预测的模型进行表情预测，并返回一个包含有各组表情匹配度的数组，保存在变量prediction中
            # print(prediction) # 此变量的保存格式为[[0. 0. 0. 1. 0. 0. 0.]]，代表对happy的匹配程度为100%
            maxindex = int(np.argmax(prediction)) # 取出预测数组中概率最大的那一个项的序号，即预测到的表情
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y+h-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # 读取表情字典里对应的表情，使用opencv的putText函数在之前的面部框上标明表情

        cv2.imshow('Result',frame) # 显示含有预测结果的视频流
        key = cv2.waitKey(1) & 0xFF # 保持视频流输出，并检测键盘操作
        if key == ord('s'): # 当键盘按下s开始截图
            frame_count += 1 # 截图序号
            save_path1 = f'result/result{frame_count:04d}.png' # 保存预测帧的文件格式
            save_path2 = f'result/input{frame_count:04d}.png' # 保存原始帧的文件格式
            cv2.imwrite(save_path1, frame) # 保存预测帧
            cv2.imwrite(save_path2, inputframe) # 保存原始帧
            
        # 如果按下 'q' 键则退出循环
        elif key == ord('q'):
            break

    cap.release() # 释放摄像头
    cv2.destroyAllWindows() # 销毁所有窗口