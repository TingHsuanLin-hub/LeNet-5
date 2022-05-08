import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD
from keras.datasets import mnist
from keras.metrics import SparseCategoricalAccuracy, SparseCategoricalCrossentropy
import time
import os
import numpy as np
from keras_flops import get_flops

def Lenet5():
    model = tf.keras.Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5,5), padding='valid', input_shape=(32,32,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=16, kernel_size=(5,5),padding='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(120,activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model
optimizer = SGD(1e-3,momentum=0.9)
crossloss = keras.losses.SparseCategoricalCrossentropy()

writer_static = tf.summary.create_file_writer('log/tensorflow_static')
writer_dynamic = tf.summary.create_file_writer('log/tensorflow_dynamic')
training_acc_metric = SparseCategoricalAccuracy()
val_acc_metric = SparseCategoricalAccuracy()
training_loss_metric = SparseCategoricalCrossentropy()
val_loss_metric = SparseCategoricalCrossentropy()

if not os.path.isdir('checkpoint/tensorflow_static'):
    os.mkdir('checkpoint/tensorflow_static')
if not os.path.isdir('checkpoint/tensorflow_dynamic'):
    os.mkdir('checkpoint/tensorflow_dynamic')
#hyper-parameter setting
epoch = 500
batch_size = 64


@tf.function
def training_static(x,y):
    with tf.GradientTape() as tape:
        outputs = model_static(x, training=True)
        loss = crossloss(y, outputs)
    grads  = tape.gradient(loss, model_static.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_static.trainable_weights))
    training_acc_metric.update_state(y, outputs)
    training_loss_metric.update_state(y, outputs)
    return outputs, loss


def training_dynamic(x,y):
    with tf.GradientTape() as tape:
        outputs = model_dynamic(x, training=True)
        loss = crossloss(y, outputs)
    grads  = tape.gradient(loss, model_dynamic.trainable_weights)
    optimizer.apply_gradients(zip(grads, model_dynamic.trainable_weights))
    training_acc_metric.update_state(y, outputs)
    training_loss_metric.update_state(y, outputs)
    return outputs, loss


@tf.function
def val_static(x,y):
    outputs = model_static(x, training=False)
    loss = crossloss(y, outputs)
    val_acc_metric.update_state(y,outputs)
    val_loss_metric.update_state(y, outputs)
    return outputs, loss


def val_dynamic(x,y):
    outputs = model_dynamic(x, training=False)
    loss = crossloss(y, outputs)
    val_acc_metric.update_state(y,outputs)
    val_loss_metric.update_state(y, outputs)
    return outputs, loss

@tf.function
def testing_static(x,pretrained_model):
    model = Lenet5()
    model.load_weights(pretrained_model)
    start_time = time.time()
    outputs = model(x, training=False)
    predict = outputs.argmax(dim=1)
    infer_time = time.time()-start_time
    infer_time = infer_time/x.size(0)
    print("Average Inference Time:{:.3f}".format(infer_time))
    return predict


def testing_dynamic(x, pretrained_model):
    model = Lenet5()
    model.load_weights(pretrained_model)
    start_time = time.time()
    outputs = model(x, training=False)
    predict = outputs.argmax(dim=1)
    infer_time = time.time()-start_time
    infer_time = infer_time/x.size(0)
    print("Average Inference Time:{:.3f}, ".format(infer_time))
    return predict


if __name__ == "__main__":
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train ,x_test = tf.image.resize(x_train[..., np.newaxis]/255.0,[32,32]), tf.image.resize(x_test[..., np.newaxis]/255.0,[32,32])
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_loader = train_loader.shuffle(buffer_size=1024).batch(batch_size)
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_loader = test_loader.batch(batch_size)
    model_static = Lenet5()
    model_dynamic = Lenet5()
    # Model Static
    print("\nStart of Static Graph")
    best_acc = 0.0
    for epo in range(epoch):
        start_time = time.time()
        for batch_idx,(x_train, label_train) in enumerate(train_loader):
            outputs, loss = training_static(x_train, label_train)
        with writer_static.as_default():
            tf.summary.scalar('Training_loss', training_loss_metric.result(), step=epo)
            tf.summary.scalar("Training_accuracy", training_acc_metric.result(), step=epo)
        
        training_time = time.time()-start_time
        print("Average Training Time:{:.3f} \t Train Epoch {} \t , Training loss: {:.3f}, Training accuracy: {:.3f}%".format(training_time, epo, training_loss_metric.result(), 100*training_acc_metric.result()))

        if epo % 5 == 0:
            start_time = time.time()
            for batch_idx, (x_test, label_test) in enumerate(test_loader):
                outputs, loss = val_static(x_test, label_test)
            with writer_static.as_default():
                tf.summary.scalar("Validation_loss", val_loss_metric.result(), step=epo)
                tf.summary.scalar("Validation_accuracy", val_acc_metric.result(), step=epo)
            val_time = time.time()-start_time
            print("Average Testing Time:{:.3f} \t Test Epoch {} \t , Testing loss: {:.3f}, Testing accuracy: {:.3f}%".format(val_time, epo, val_loss_metric.result(), 100*val_acc_metric.result()))

            if best_acc < val_acc_metric.result():
                best_acc = val_acc_metric.result()
                model_static.save_weights('checkpoint/tensorflow_static/epoch_%d_acc_%.3f.pth' %(epo, val_acc_metric.result()))

        if epo % 20 == 0 and epo > 1:
            model_static.save_weights('checkpoint/tensorflow_static/epoch_%d_acc_%.3f.pth' %(epo, val_acc_metric.result()))
        
        val_loss_metric.reset_states()
        val_acc_metric.reset_states()

        training_acc_metric.reset_states()
        training_loss_metric.reset_states()


    # Model Dynamic
    print("\nStart of Dynamic Graph")
    best_acc = 0.0
    for epo in range(epoch):
        start_time = time.time()
        for batch_idx,(x_train, label_train) in enumerate(train_loader):
            outputs, loss = training_dynamic(x_train, label_train)
        with writer_dynamic.as_default():
            tf.summary.scalar('Training_loss', training_loss_metric.result(), step=epo)
            tf.summary.scalar("Training_accuracy", training_acc_metric.result(), step=epo)
        
        training_time = time.time()-start_time
        print("Average Training Time:{:.3f} \t Train Epoch {} \t , Training loss: {:.3f}, Training accuracy: {:.3f}%".format(training_time, epo, training_loss_metric.result(), 100*training_acc_metric.result()))

        if epo % 5 == 0:
            start_time = time.time()
            for batch_idx, (x_test, label_test) in enumerate(test_loader):
                outputs, loss = val_dynamic(x_test, label_test)
            with writer_dynamic.as_default():
                tf.summary.scalar("Validation_loss", val_loss_metric.result(), step=epo)
                tf.summary.scalar("Validation_accuracy", val_acc_metric.result(), step=epo)
            val_time = time.time()-start_time
            print("Average Testing Time:{:.3f} \t Test Epoch {} \t , Testing loss: {:.3f}, Testing accuracy: {:.3f}%".format(val_time, epo, val_loss_metric.result(), 100*val_acc_metric.result()))
            
            if best_acc < val_acc_metric.result():
                best_acc = val_acc_metric.result()
                model_dynamic.save_weights('checkpoint/tensorflow_dynamic/epoch_%d_acc_%.3f.pth' %(epo, val_acc_metric.result()))

        if epo % 20 == 0 and epo > 1:
            model_dynamic.save_weights('checkpoint/tensorflow_dynamic/epoch_%d_acc_%.3f.pth' %(epo, val_acc_metric.result()))
        
        val_loss_metric.reset_states()
        val_acc_metric.reset_states()

        training_acc_metric.reset_states()
        training_loss_metric.reset_states()
    # model = Lenet5()
    # flops = get_flops(model, batch_size=1)
    # print(f"FLOPS: {flops / 10 ** 3:.03} K")