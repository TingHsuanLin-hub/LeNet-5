import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
from keras.datasets import mnist
import lenet5_tensorflow
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


def tester(tflite_model, test_images):
    # Initialize TFLite interpreter using the model.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_tensor_index = interpreter.get_input_details()[0]["index"]
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for test_image in test_images:
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_tensor_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    return prediction_digits


if __name__ == "__main__":
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train ,x_test = tf.image.resize(x_train[..., np.newaxis]/255.0,[32,32]), tf.image.resize(x_test[..., np.newaxis]/255.0,[32,32])
    model = Lenet5()

    model.load_weights("./checkpoint/tensorflow_dynamic/epoch_480_acc_0.989.pth")
    # print("Original Model Size of tensorflow version in KBs:", len(model) / 1024)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_float_model = converter.convert()
    print("Original model size = %dKBs." % (len(tflite_float_model)/1024))
    f = open('tflite_float32.tflite', 'wb')
    f.write(tflite_float_model)
    f.close()

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_float_model = converter.convert()
    print("Float model size = %dKBs." % (len(tflite_float_model)/1024))
    f = open('tflite_float16.tflite', 'wb')
    f.write(tflite_float_model)
    f.close()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    images = x_train
    ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
    def representative_data_gen():
        for input_value in ds.take(100):
            # Model has only one input so each data point has one element.
            yield [input_value]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_quantized_model = converter.convert()
    print('Quantized model size = %dKBs.' % (len(tflite_quantized_model)/1024))
    f = open('tflite_uint8.tflite', 'wb')
    f.write(tflite_quantized_model)
    f.close()
    