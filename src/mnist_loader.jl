"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

using JLD2, FileIO
using Serialization

"""Return the MNIST data as a tuple containing the training data,
the validation data, and the test data.

The ``training_data`` is returned as a tuple with two entries.
The first entry contains the actual training images.  This is a
numpy ndarray with 50,000 entries.  Each entry is, in turn, a
numpy ndarray with 784 values, representing the 28 * 28 = 784
pixels in a single MNIST image.

The second entry in the ``training_data`` tuple is a numpy ndarray
containing 50,000 entries.  Those entries are just the digit
values (0...9) for the corresponding images contained in the first
entry of the tuple.

The ``validation_data`` and ``test_data`` are similar, except
each contains only 10,000 images.

This is a nice data format, but for use in neural networks it's
helpful to modify the format of the ``training_data`` a little.
That's done in the wrapper function ``load_data_wrapper()``, see
below.
"""
function load_data()
    @load "../data/mnist.jld2"
    return ((train_digits, train_labels),
        (valid_digits, valid_labels),
        (test_digits, test_labels))
end

"""Return a tuple containing ``(training_data, validation_data,
test_data)``. Based on ``load_data``, but the format is more
convenient for use in our implementation of neural networks.

In particular, ``training_data`` is a list containing 50,000
2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
containing the input image.  ``y`` is a 10-dimensional
numpy.ndarray representing the unit vector corresponding to the
correct digit for ``x``.

``validation_data`` and ``test_data`` are lists containing 10,000
2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
numpy.ndarry containing the input image, and ``y`` is the
corresponding classification, i.e., the digit values (integers)
corresponding to ``x``.

Obviously, this means we're using slightly different formats for
the training data and the validation / test data.  These formats
turn out to be the most convenient for use in our neural network
code."""
function load_data_wrapper()
    ((tr_d, tr_l), (va_d, va_l), (te_d, te_l)) = load_data()
    ntr = size(tr_d, 1)
    nva = size(va_d, 1)
    nte = size(te_d, 1)
    training_inputs = [reshape(tr_d[i, :], (784,)) for i in 1:ntr]
    training_results = [vectorized_result(tr_l[i]) for i in 1:ntr]
    training_data = [(training_inputs[i], training_results[i]) for i in 1:ntr]
    validation_inputs = [reshape(va_d[i, :], (784,)) for i in 1:nva]
    validation_data = [(validation_inputs[i], va_l[i]) for i in 1:nva]
    test_inputs = [reshape(te_d[i, :], (784,)) for i in 1:nte]
    test_data = [(test_inputs[i], te_l[i]) for i in 1:nte]
    return (training_data, validation_data, test_data)
end

vectorized_result(j::Integer) = vectorized_result(Float32, j)
function vectorized_result(T, j::Integer)
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    #e = OffsetArray{Float32}(undef, 0:9)
    #e[j] = 1.0
    e = zeros(T, 10)
    e[j+1] = 1.0
    return e
end
