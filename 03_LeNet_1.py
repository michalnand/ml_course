from libs_rysy_python.rysy import *

#load dataset
dataset_path = "/home/michal/dataset/mnist/"
dataset = DatasetMnist(dataset_path + "train-images-idx3-ubyte",
                            dataset_path + "train-labels-idx1-ubyte",
                            dataset_path + "t10k-images-idx3-ubyte",
                            dataset_path + "t10k-labels-idx1-ubyte")

'''
create LeNet 1 -> first convolutional neural network
layer1 : convolution size 5x5, 4 kernels, relu activation
layer2 : average pooling with size 2x2

layer3 : convolution size 5x5, 8 kernels, relu activation
layer4 : average pooling with size 2x2

layer3 : 10 output neurons

learning rate = 0.001
accuracy afeter 1 epoch = 95.5%
accuracy afeter 5 epoch = 96.95%
'''

cnn = CNN(dataset.get_input_shape(), dataset.get_output_shape(), 0.001)

cnn.add_layer("convolution", Shape(5, 5, 4))
cnn.add_layer("relu")
cnn.add_layer("average_pooling", Shape(2, 2))

cnn.add_layer("convolution", Shape(5, 5, 8))
cnn.add_layer("relu")
cnn.add_layer("average_pooling", Shape(2, 2))

cnn.add_layer("output")

cnn._print()

#train network, 1 epoch count
cnn.train(dataset.get_training_output_all(), dataset.get_training_input_all(), 5)


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
