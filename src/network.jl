"""
network.jl
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

import Printf: @printf
import Random: shuffle!
using LinearAlgebra

struct Network{T}
    num_layers :: Int
    sizes ::Vector{Int}
    biases ::Vector{Vector{T}}
    weights::Vector{Matrix{T}}
end

"""The list ``sizes`` contains the number of neurons in the
respective layers of the network.  For example, if the list
was [2, 3, 1] then it would be a three-layer network, with the
first layer containing 2 neurons, the second layer 3 neurons,
and the third layer 1 neuron.  The biases and weights for the
network are initialized randomly, using a Gaussian
distribution with mean 0, and variance 1.  Note that the first
layer is assumed to be an input layer, and by convention we
won't set any biases for those neurons, since biases are only
ever used in computing the outputs from later layers."""
Network(sizes) = Network(Float32, sizes)
function Network(T, sizes)
    biases = [randn(T, y) for y in sizes[2:end]]
    weights = [randn(T, y, x)
                    for (x, y) in zip(sizes[1:end-1], sizes[2:end])]

    Network{T}(length(sizes), sizes, biases, weights)
end

"""Return the output of the network if ``a`` is input."""
function feedforward(
    N::Network{T},
    a::AbstractVector{T}) where T

    S = typeof(a)
    for (b, w) in zip(N.biases, N.weights)
        a::S = sigmoid.(w*a+b)
    end
    return a
end

"""Train the neural network using mini-batch stochastic
gradient descent.  The ``training_data`` is a list of tuples
``(x, y)`` representing the training inputs and the desired
outputs.  The other non-optional parameters are
self-explanatory.  If ``test_data`` is provided then the
network will be evaluated against the test data after each
epoch, and partial progress printed out.  This is useful for
tracking progress, but slows things down substantially."""
function SGD!(N::Network{T},
              training_data::Vector{Tuple{Vector{T},Vector{T}}},
              epochs::Int, mini_batch_size::Int, η::T,
              test_data=nothing) where T

    have_test_data = test_data != nothing
    if have_test_data
        n_test = length(test_data)
    end

    n = length(training_data)
    for j in 1:epochs
        shuffle!(training_data)

        for k in 1:mini_batch_size:n-mini_batch_size
            mini_batch = training_data[k:k+mini_batch_size]
            update_mini_batch!(N, mini_batch, η)
        end

        if have_test_data
            @printf("Epoch %d: %d / %d\n", j, evaluate(N, test_data), n_test)
        else
            @printf("Epoch %d complete\n", j)
        end
    end
end

"""Update the network's weights and biases by applying
gradient descent using backpropagation to a single mini batch.
The ``mini_batch`` is a list of tuples ``(x, y)``, and ``η``
is the learning rate."""
function update_mini_batch!(N::Network, mini_batch, η)
    ∇b = [zeros(size(b)) for b in N.biases]
    ∇w = [zeros(size(w)) for w in N.weights]
    for (x, y) in mini_batch
        δ_∇b, δ_∇w = backprop(N, x, y)
        ∇b = [∇b+δ∇b for (∇b, δ∇b) in zip(∇b, δ_∇b)]
        ∇w = [∇w+δ∇w for (∇w, δ∇w) in zip(∇w, δ_∇w)]
    end
    c = η/length(mini_batch)
    N.weights[:] = [w-c*δw for (w, δw) in zip(N.weights, ∇w)]
    N.biases[:]  = [b-c*δb for (b, δb) in zip(N.biases , ∇b)]
end

"""Return a tuple ``(∇b, ∇w)`` representing the
gradient for the cost function C_x.  ``∇b`` and
``∇w`` are layer-by-layer lists of arrays, similar
to ``N.biases`` and ``N.weights``."""
function backprop(N::Network{T}, x::AbstractVecOrMat{T}, y) where T
    ∇b = [zeros(T, size(b)) for b in N.biases]
    ∇w = [zeros(T, size(w)) for w in N.weights]
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for (b, w) in zip(N.biases, N.weights)
        z = w*activation+b
        push!(zs, z)
        activation = sigmoid.(z)
        push!(activations, activation)
    end
    # backward pass
    δ = cost_derivative(activation, y).*sigmoid_prime.(zs[end])
    ∇b[end] = δ
    ∇w[end] = δ*activations[end-1]'
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in 2:N.num_layers-1
        z = zs[end+1-l]
        σ′ = sigmoid_prime.(z)
        δ = (N.weights[end-l+2]'*δ) .* σ′
        ∇b[end+1-l] = δ
        ∇w[end+1-l] = δ*activations[end-l]'
    end
    return (∇b, ∇w)
end

"""Return the number of test inputs for which the neural
network outputs the correct result. Note that the neural
network's output is assumed to be the index of whichever
neuron in the final layer has the highest activation."""
function evaluate(N::Network, test_data)
    cnt = 0
    for (x, y) in test_data
        test_result = argmax(feedforward(N, x)) - 1
        cnt += (test_result == y)
    end
    return cnt
end

#doc"""Return the vector of partial derivatives \partial C_x /
#\partial a for the output activations."""
cost_derivative(x, y) = x-y

#### Miscellaneous functions
"""The sigmoid function."""
sigmoid(z) = 1/(1+exp(-z))

"""Derivative of the sigmoid function."""
sigmoid_prime(z) = sigmoid(z)*(1-sigmoid(z))
