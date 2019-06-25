from libs_rysy_python.rysy import *

#load dataset
dataset = DatasetImages("dataset_path.json")

'''
create modern CNN, only 3x3 kernels, deeper network, no FC layers, dropout
layer1 : convolution size 3x3, 8 kernels, relu activation
layer2 : max pooling with size 2x2

layer3 : convolution size 3x3, 8 kernels, relu activation
layer4 : max pooling with size 2x2

layer5 : convolution size 3x3, 8 kernels, relu activation
layer6 : max pooling with size 2x2

layer7 : dropout

layer8 : 10 output neurons

learning rate = 0.001
accuracy after 20 epoch = ?%
'''

cnn = CNN(dataset.get_input_shape(), dataset.get_output_shape(), 0.0005)

cnn.add_layer("convolution", Shape(3, 3, 8))
cnn.add_layer("relu")
cnn.add_layer("max_pooling", Shape(2, 2))

cnn.add_layer("convolution", Shape(3, 3, 8))
cnn.add_layer("relu")
cnn.add_layer("max_pooling", Shape(2, 2))

cnn.add_layer("convolution", Shape(3, 3, 8))
cnn.add_layer("relu")
cnn.add_layer("max_pooling", Shape(2, 2))

 
cnn.add_layer("dropout")

cnn.add_layer("output")

cnn._print()
cnn.save("path_network/")

#train network, 1 epoch count
cnn.train(dataset.get_training_output_all(), dataset.get_training_input_all(), 20)
cnn.save("path_network/")


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
