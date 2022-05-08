# LeNet-5
## How to train a model
You can simply run the files names "lenet5_mine.py", "lenet5_tensorflow.py", "lenet5_pytorch.py", "lenet5_tflite.py" to train or convert a model.

## Evaluate performance 
You need to execute the correspondence method with "testing" appended in each file mentioned above to evaluate the performance.

### Reference:
[LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
* Tensorflow
https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
https://datahacker.rs/lenet-5-implementation-tensorflow-2-0/

* PyTorch
https://hackmd.io/@lido2370/SJMPbNnKN?type=view
https://iter01.com/648766.html
https://www.cnblogs.com/gshang/p/13099170.html

* Handcrafted
https://www.linkedin.com/pulse/forward-back-propagation-over-cnn-code-from-scratch-coy-ulloa
https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1
https://www.796t.com/content/1546281127.html
[不同layer的forward和backward](https://datascience-enthusiast.com/DL/Convolution_model_Step_by_Stepv2.html)
[activation, optimizer的forward和backward](https://nthu-datalab.github.io/ml/labs/10_TensorFlow101/10_NN-from-Scratch.html)
[更多layer的forward和backward](https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/fc_layer)
[softmax_crossentropy的backward](https://medium.com/hoskiss-stand/backpropagation-with-softmax-cross-entropy-d60983b7b245)

* TFlite
https://developer.android.com/codelabs/digit-classifier-tflite?hl=zh-tw#0

[圖片和tensor之間的轉換(包含Tensorflow和PyTorch)](https://towardsdatascience.com/convert-images-to-tensors-in-pytorch-and-tensorflow-f0ab01383a03)
