from libs_rysy_python.rysy import *

#load dataset
dataset_path = "/home/michal/dataset/mnist/"
dataset = DatasetMnist(dataset_path + "train-images-idx3-ubyte",
                            dataset_path + "train-labels-idx1-ubyte",
                            dataset_path + "t10k-images-idx3-ubyte",
                            dataset_path + "t10k-labels-idx1-ubyte")

dataset.normalise_input()


'''
create modern CNN, only 3x3 kernels, deeper network, no FC layers, dropout
layer1 : convolution size 3x3, 16 kernels, relu activation
layer2 : max pooling with size 2x2

layer3 : convolution size 3x3, 32 kernels, relu activation
layer4 : convolution size 3x3, 32 kernels, relu activation
layer5 : max pooling with size 2x2

layer6 : convolution size 3x3, 64 kernels, relu activation

layer7 : dropout

layer8 : 10 output neurons

learning rate = 0.001
accuracy afeter 1 epoch = 98.15%
'''

cnn = CNN(dataset.get_input_shape(), dataset.get_output_shape(), 0.001)

cnn.add_layer("convolution", Shape(3, 3, 16))
cnn.add_layer("elu")
cnn.add_layer("max_pooling", Shape(2, 2))

cnn.add_layer("convolution", Shape(3, 3, 32))
cnn.add_layer("elu")
cnn.add_layer("convolution", Shape(3, 3, 32))
cnn.add_layer("elu")
cnn.add_layer("max_pooling", Shape(2, 2))

cnn.add_layer("convolution", Shape(3, 3, 64))
cnn.add_layer("elu")

cnn.add_layer("dropout")

cnn.add_layer("output")

cnn._print()

#train network, 1 epoch count
cnn.train(dataset.get_training_output_all(), dataset.get_training_input_all(), 1)


#test network response on whole testing dataset items

compare = ClassificationCompare(dataset.get_classes_count())

nn_output = VectorFloat(dataset.get_classes_count())

#for all testing items
for item_idx in range(0, dataset.get_testing_count()):
    #get network response
    cnn.forward(nn_output, dataset.get_testing_input(item_idx))

    #compare with testing dataset
    compare.add(dataset.get_testing_output(item_idx), nn_output)

    if compare.is_nan_error():
        print("NaN error")

#process computing and print results
compare.compute()
print(compare.asString())
