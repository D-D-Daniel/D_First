import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential     #线性堆叠模型
from keras.layers import Dense         #全连接模式
from keras.layers import Dropout


def model_train(x_Train_normalize, y_TrainOneHot):
    model = Sequential()
    model.add(Dense(units=1000,
                    input_dim=784,
                    kernel_initializer="normal",
                    activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=1000,
                    kernel_initializer="normal",
                    activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(units=10,
                    kernel_initializer="normal",
                    activation="softmax"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    train_history = model.fit(x=x_Train_normalize,
                              y=y_TrainOneHot,
                              validation_split=0.2,
                              epochs=10,
                              batch_size=200,
                              verbose=2)
    return model, train_history


def show_train_history(train_history, train, validation):
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
    (X_train_image, y_train_label), (X_test_image, y_test_label) = mnist.load_data()
    x_Train = X_train_image.reshape(60000, 28 * 28).astype(np.float32)
    x_Test = X_test_image.reshape(10000, 28 * 28).astype(np.float32)
    x_Train_normalize = x_Train / 255
    x_Test_normalize = x_Test / 255
    y_TrainOneHot = np_utils.to_categorical(y_train_label)
    y_TestOneHot = np_utils.to_categorical(y_test_label)

    model, train_history = model_train(x_Train_normalize, y_TrainOneHot)
    scores = model.evaluate(x_Test_normalize, y_TestOneHot)
    print("scores =", scores[1])

    # 预测结果
    prediction = model.predict_classes(x_Test_normalize)

    # 训练过程正确率可视化
    show_train_history(train_history, "acc", "val_acc")

    # 混淆矩阵
    print(pd.crosstab(y_test_label, prediction, rownames=["label"], colnames=["prediction"]))

    # 保存训练数据
    save_result(y_test_label, prediction, "C:/my python/project_mnist/", "mnist_MLP_prediction")
