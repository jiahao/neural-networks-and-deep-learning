include("network.jl")
include("mnist_loader.jl")

training_data, validation_data, test_data = load_data_wrapper()
net = Network([784, 30, 10])
SGD!(net, training_data, 30, 10, 3.0f0, test_data)

