import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils      #独热处理
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D    #平坦层，卷积层，池化层


def model_train(x_Train4D_normalize, y_TrainOneHot):
    model = Sequential()
    #建立卷积层1
    model.add(Conv2D(filters=16,                  #建立16个滤镜
                    kernel_size=(5, 5),          #滤镜大小5*5
                    padding="same",               #卷积运算产生的图片大小不变
                    input_shape=(28, 28, 1),      #前两个数值为像素大小，后一数值为单色
                    activation="relu"))
    #建立池化层1
    model.add(MaxPooling2D(pool_size=(2, 2)))     #4-->1
    #建立卷积层2
    model.add(Conv2D(filters=36,
                    kernel_size=(5, 5),
                    padding="same",
                    activation="relu"))
    #建立池化层2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #加入DropOut减少过拟合
    model.add(Dropout(0.25))                     #随机放弃25%的神经元
    #建立平坦层
    model.add(Flatten())
    #建立隐藏层   隐藏层中有128个神经元
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))                     #随机放弃50%的神经元
    #建立输出层
    model.add(Dense(10, activation="softmax"))
    model.compile(loss = "categorical_crossentropy",
                 optimizer="adam",
                 metrics=["accuracy"])
    train_history = model.fit(x=x_Train4D_normalize,
                             y=y_TrainOneHot,
                             verbose=2,
                             epochs=10,
                             validation_split=0.2,
                             batch_size=300)
    return model, train_history


def  show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


def save_result(label, prediction, path, name):
    result = pd.DataFrame({"label":label, "prediction":prediction})
    full_path = path+name+".csv"
    result.to_csv(full_path, index=False)


if __name__ == "__main__":
    #数据预处理
    (x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()
    x_Train_4D = x_Train.reshape(60000, 28, 28, 1).astype(np.float32)
    x_Test_4D = x_Test.reshape(10000, 28, 28, 1).astype(np.float32)
    x_Train4D_normalize = x_Train_4D / 255
    x_Test4D_normalize = x_Test_4D / 255
    y_TrainOneHot = np_utils.to_categorical(y_Train)
    y_TestOneHot = np_utils.to_categorical(y_Test)

    model, train_history = model_train(x_Train4D_normalize, y_TrainOneHot)
    scores = model.evaluate(x_Test4D_normalize, y_TestOneHot)
    print("scores =", scores[1])

    #预测结果
    prediction = model.predict_classes(x_Test4D_normalize)

    #训练过程正确率可视化
    show_train_history(train_history, "acc", "val_acc")

    #混淆矩阵
    print(pd.crosstab(y_Test, prediction, rownames=["label"], colnames=["prediction"]))

    #保存训练数据
    save_result(y_Test, prediction, "C:/my python/project_mnist/", "mnist_CNN_prediction")


