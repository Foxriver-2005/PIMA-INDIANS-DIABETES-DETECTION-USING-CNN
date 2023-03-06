# PIMA-INDIANS-DIABETES-DETECTION-USING-CNN

CNN
    • Learns the filter automatically to extract the right features from the data.
    • It captures spatial features (arrangement of pixels) whereas Artificial neural network can’t (ANN).
    • It also follow parameter sharing like RNN, applies single filter in different parts of a single image.
    • It doesn’t have recurrent connection like RNN instead it has convolution type of hidden layers.
    • Convolution and pooling are used as activation functions.
    • CONVOLUTION – Input image and other as filter on input image (kernel) produces input image.
    • POOLING – Picking maximum value from selected region  is max poling and vice verse.

SIMPLEX SIMPLEX CLASSIFICATION
Softmax (L(n)) = e^Ln / absolute of e^L

L = X.W + b

Y = softmax (X.W + b) where: X – images, W – weights, b – bases.

In TensorFlow
Y = tf.nn.softmax (tf.matmul(X,W) + b)

Cross entropy: Σyi.log(Yi)

Softmax Function
    • Softmax activations function will be applied in the last layer of neural networks, instead of ReLU, tanh, sigmoid.
    • It is used to map the non-normalized output of a network to a probability over predicted output class. That is, it convert output of last layer into a probability distribution.

Implementation of Neural Network
    a) Feed forward – set up input features and random weights. Weights will be optimized by back propagation.
    b) Back propagation – Calculating error between predicted output and target output and use gradient descent method to update weights.
    c) Gradient descent – It operates iteratively to find the optimal features for its parameters. User-defined leading rate and initial parameters vales.

Vanishing & exploding Gradient
    • It is very common problem in every neural network, which is associated with back propagation.
    • Weights of network are updated through back propagation by finding gradients.
    • When the number of hidden layer is high, then the gradient vanishes or explodes as it propagates backward. It leads instability in network, unable to learn from training.
    • The explosion occurs through exponential growth by repeatedly multiplying gradient through the network layers that have values layers than 1.0.
    • It can be fixed by redesigning the network, using long short term memory network gradient clipping etc.

Keras Basic Syntax
Adding layers
model.add(Dense(12,input_dim=8,int = ‘uniform’, activation = ‘relu’))
model.add(Dense(8,activation = ‘relu’))

Compile model
model.compile(loss= ‘binary_crossentropy’, optimizer = ‘adam’, metrics = [‘accuracy’]

    • Optimizer – String identifier of an existing optimizer.
    • Loss function – This is the objective that the model will try to minimize.
    • Limit of metrics – for any classification problem you will want to set this to metrics = [‘accuracy’]

Batch Vs Epoch
    • Training occurs over epochs and each is split into batches.
    • Epoch – one pass through all of the rows in the training dataset.
    • Batch – one or more samples considered by the model within an epoch before weights are updated.
