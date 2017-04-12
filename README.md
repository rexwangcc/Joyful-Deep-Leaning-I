Joyful Deep Learning - I
=========================

This repository is a collection of my useful resources of Deep Learning in Tensorflow, in 5 sections:

1. [Machine Learning and Deep Learning Basics in Math and Numpy](#sec1)
2. [Deep Learning Basics in Math, Numpy and Scikit-Learn](#sec2)
3. [Deep Learning Basics in Tensoflow](#sec3)
4. [Deep Learning Advanced  in Tensoflow](#sec4)
5. [Idle](#sec5)

It would be really grateful to contribute to/clone this repository, commercial uses are not welcomed. Thanks for the help of [Prof.Brian Kulis](http://www.bu.edu/eng/profile/brian-kulis/) and [Prof.Kate Saenko](http://vision.cs.uml.edu/ksaenko.html) and TFs of CS591-S2 (Deep Learning) at Boston University. Of course also thanks to Google's Open Source Tensorflow!

All results in Jupyter-Notebook are trained under GTX 1070, training CPUs may cost much more time.


<a name="sec1"></a>
[## Section 1 Content - Machine Learning and Deep Learning Basics in Math and Numpy (click to view full notebook)](https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/Section1/Section1.ipynb)

* **Coding requirements:**

```python
# Python 3.5+
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import matplotlib.cm as cm

```

* **Closed-Form Maximum Likelihood mathematical derivation:**

    + $P(x \ | \ \theta) = \theta e^{-\theta x}$ for $x \geq 0$

    + $P(x \ | \ \theta) = \frac{1}{\theta}$ for $ 0 \leq x \leq \theta$

* **Gradient for Maximum Likelihood Estimation mathematical derivation:**
    
    + Gradients for log-likelihood of the following model:

        - we have $X \in \mathbf R^{n \times k}$ - constant data matrix, $\mathbf x_i$ - vector corresponding to a single data point

        - $\theta$ is a $k$-dimensional (unknown) weight vector

        - $\varepsilon \sim \text{Student}(v)$ is a $n$-dimensional (unknown) noise vector

        - and we observe vector $\mathbf y = X\theta + \varepsilon$

        - $$ P(y_i \ | \ \mathbf x_i, \theta, v) = \frac{1}{Z(v)} \Big(1 + \frac{(\theta^T \mathbf x_i - y_i) ^2}{v}\Big)^{-\frac{v+1}{2}}$$

    + Stochastic Gradient Descent Implementation

* **Matrix Derivatives mathematical derivation:**
    
    + Multivariate Gaussian:

        - $ \frac{\partial \mathcal L(\Sigma)}{\partial \Sigma} = -\frac12 \left( \frac{1}{|\Sigma|} |\Sigma| \Sigma^{-T}  - \Sigma^{-T} (x- \bar \mu)(x-\bar \mu)^T\Sigma^{-T} \right) =  -\frac12 \left(\Sigma^{-T}  - \Sigma^{-T} (x- \bar \mu)(x-\bar \mu)^T\Sigma^{-T} \right)$

    + Multi-target Linear Regression model:

        - we have $X \in \mathbf R^{n \times k}$ is a constant data matrix

        - $\theta$ is a $k \times m$-dimentional weight matrix

        - $\varepsilon_{ij} \sim \mathcal N(0, \sigma_\epsilon)$ is a normal noise ($i \in [0; n], j \in [0;m]$)

        - and we observe a matrix $Y = X\theta + \varepsilon \in \mathbf R^{n \times m}$

        - $$\varepsilon = Y - X\theta \sim \mathcal N_n(0, \sigma_\epsilon I)$$

        - $$\mathcal L(\theta) = \log P(Y - X\theta \ | \ \theta) = \log \mathcal N_n(Y - X\theta \ | \ 0, \sigma_\epsilon I)$$

        - $$\theta_{MLE} = \arg \max_{\theta} \mathcal L(\theta) = \arg \min_{\theta} \text{loss}(\theta) = \arg \min_{\theta} \big( ||Y-X\theta||^2_F \big)$$

        - Deriavation: $\frac{\partial\text{loss}(\theta)}{\partial \theta} = -2X^T (Y-X\theta)$

        - Deriavation: 
        $\theta_{MLE} = (X^T X)^{-1} X^T Y$

* **Logistic Regression mathematical derivation**


* **Logistic Regression implementation**

<a name="sec2"></a>
[## Section 2 Content - Deep Learning Basics in Math, Numpy and Scikit-Learn (click to view full notebook)](https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/Section2/Section2.ipynb)

* **Coding requirements:**

```python
# Python 3.5+
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from sklearn.datasets import fetch_mldata

```

* **Cross-Entropy and Softmax mathematical derivation:**
    
    + Minimizing the multiclass cross-entropy loss function to obtain the maximum likelihood estimate of the parameters $\theta$:

        - $L(\theta)= - \frac{1}{N}\sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log(h_k(x_i,\theta))$ where $N$ is the number of examples $\{x_i,y_i\}$

* **Simple Regularization Methods:**

    + *L2* regularization

    + *L1* regularization

* **Backprop in a simple MLP - Multi-layer perceptron's mathematical derivation:**

<p align="center">
  <img src="https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/imgs/s2_mlp.png?raw=true" width="70%"/>
</p>

* **XOR problem - A Neural network to solve the XOR problem:** (*This is a really good example to help us understand the essence of neural networks*)

<p align="center">
  <img src="https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/imgs/s2_xor.png?raw=true"/>
</p>

* **Implementing a simple MLP - Implement a MLP by hand in numpy and scipy**

    + Common useful activation functions implementation escaped from numerical accuracy problems:

        - softplus function:

                import numpy as np

                    def softplus(x):
                        return np.logaddexp(0, x)

                    def derivative_softplus(x):
                        return np.exp(-np.logaddexp(0,-z))
                        
        - sigmoid function:

                import numpy as np

                    def sigmoid(x):
                        return np.exp(-np.logaddexp(0, -x))

                    def derivative_sigmoid(x):
                        return np.multiply(np.exp(-np.logaddexp(0, -x)), (1.-np.exp(-np.logaddexp(0, -x))))

        - relu function:

                import numpy as np

                    def relu(x):
                        return np.maximum(0, x)

                    def derivative_relu(x):
                        for i in range(0, len(x)):
                            for k in range(len(x[i])):
                                if x[i][k] > 0:
                                    x[i][k] = 1
                                else:
                                    x[i][k] = 0
                        return x

    + Forward pass implementation

    + Backward pass implementation

    + Test MLP on MNIST dataset and its visualization

<a name="sec3"></a>
[## Section 3 Content - Deep Learning Basics in Tensoflow (click to view full notebook)](https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/Section3/Section3.ipynb)

* **Coding requirements:**

```python
# Python 3.5+
import numpy as np

# tensorflow-gpu==1.0.1 or tensorflow==1.0.1
import tensorflow as tf

from matplotlib import pyplot as plt

# Scikit-learn's TSNE is relatively slow, use BHTSNE as a faster alternative:
# https://github.com/dominiek/python-bhtsne
from sklearn.manifold import TSNE
```

* **MNIST Softmax Classifier Demo in TensorFlow**

* **Building Neural Networks with the power of `Variable Scope` - MLP in TensorFlow:**

    With the power of **variable scope**, we can implement a very flexible MLP in tensorflow without hard-code the layers and weights:

```python
def mlp(x, hidden_sizes, activation_fn=tf.nn.relu):
    
    '''
    Inputs:
        x: an input tensor of the images in the current batch [batch_size, 28x28]
        hidden_sizes: a list of the number of hidden units per layer. For example: [5,2] means 5 hidden units in the first layer, and 2 hidden units in the second (output) layer. (Note: for MNIST, we need hidden_sizes[-1]==10 since it has 10 classes.)
        activation_fn: the activation function to be applied

    Output:
        a tensor of shape [batch_size, hidden_sizes[-1]].
    '''
    if not isinstance(hidden_sizes, (list, tuple)):
        raise ValueError("hidden_sizes must be a list or a tuple")
        
    # Number of layers
    L = len(hidden_sizes)

    for l in range(L):

        with tf.variable_scope("layer"+str(l)):
            
            # Create variable named "weights".
            if l == 0:
                weights = tf.get_variable("weights", shape= [x.shape[1], hidden_sizes[l]], dtype=tf.float32, initializer=None)
            else:
                weights = tf.get_variable("weights", shape= [hidden_sizes[l-1], hidden_sizes[l]], dtype=tf.float32, initializer=None)

            # Create variable named "biases".
            biases = tf.get_variable("biases", shape=[hidden_sizes[l]], dtype=tf.float32, initializer=None)

            # Pre-Actiation Layer
            if l == 0:
                pre_activation = tf.add(tf.matmul(x, weights), biases)
            else:
                pre_activation = tf.add(tf.matmul(activated_layer, weights), biases)

            # Activated Layer
            if l == L-1:
                activated_layer = pre_activation
            else:
                activated_layer = activation_fn(pre_activation)
    return activated_layer
```

* **Siamese Network in TensorFlow**

<p align="center">
  <img src="https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/imgs/s3_siamese.png?raw=true" width="60%" height="50%"/>
</p>

* **Visualize learned features of Siamese Network with T-SNE**

<p align="center">
  <img src="https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/imgs/s3_tsne.png?raw=true"/>
</p>


<a name="sec4"></a>
[## Section 4 Content - Deep Learning Advanced  in Tensoflow (click to view full notebook)](https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/Section4/Section4.ipynb)

* **Coding requirements:**

```python
# Python 3.5+
import numpy as np
import scipy
import scipy.io

# tensorflow-gpu==1.0.1 or tensorflow==1.0.1
import tensorflow as tf

from matplotlib import pyplot as plt

# Scikit-learn's TSNE is relatively slow, use BHTSNE as a faster alternative:
# https://github.com/dominiek/python-bhtsne
from sklearn.manifold import TSNE
```

* **Building and training a convolutional network in Tensorflow with `tf.layers/tf.contrib`**

* **Building and training a convolutional network by hand in Tensorflow with `tf.nn`**

* **Saving and Reloading Model Weights in Tensorflow**

* **Fine-tuning a pre-trained network**

* **Visualizations using Tensorboard:**

    + Visualize Filters/Kernels

    + Visualize Loss

    + Visualize Accuracy


<a name="sec5"></a>
[## Section 5 Content (click to view full notebook)]()

