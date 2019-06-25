from libs_rysy_python.rysy import *

#load dataset
dataset_path = "/home/michal/dataset/mnist/"
dataset = DatasetMnist(dataset_path + "train-images-idx3-ubyte",
                            dataset_path + "train-labels-idx1-ubyte",
                            dataset_path + "t10k-images-idx3-ubyte",
                            dataset_path + "t10k-labels-idx1-ubyte")

'''
create example network - single layer (linear classifier)
layer1 : 10 output neurons

learning rate = 0.001
accuracy = 85.2%
'''
cnn = CNN(dataset.get_input_shape(), dataset.get_output_shape(), 0.001)

cnn.add_layer("output")
#the some result with
#cnn.add_layer("fc", Shape(1, 1, 10))


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
