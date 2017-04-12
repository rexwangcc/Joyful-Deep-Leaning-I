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

This readme file is supported by [readme2tex](https://github.com/leegao/readme2tex)

<br>
<br>
<br>
<br>



<a name="sec1"></a>
<h2><a href="https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/Section1/Section1.ipynb">Section 1 Content - Machine Learning and Deep Learning Basics in Math and Numpy (click to view full notebook)</a></h2>


* **Coding requirements:**

```python
# Python 3.5+
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import matplotlib.cm as cm

```

* **Closed-Form Maximum Likelihood mathematical derivation:**

    + <img alt="$P(x \ | \ \theta) = \theta e^{-\theta x}$" src="svgs/e569bf2f55e8050e6d778a5424437294.png?invert_in_darkmode" align=middle width="120.44191500000001pt" height="27.852989999999977pt"/> for <img alt="$x \geq 0$" src="svgs/92106cc04e070ebc1780e6da06cd51b6.png?invert_in_darkmode" align=middle width="39.418335pt" height="21.10812pt"/>

    + <img alt="$P(x \ | \ \theta) = \frac{1}{\theta}$" src="svgs/e67248bfc0ca777acca8a8424eb0cc37.png?invert_in_darkmode" align=middle width="89.005455pt" height="27.720329999999983pt"/> for <img alt="$ 0 \leq x \leq \theta$" src="svgs/261d4cde744b07a079970d5243f18a7d.png?invert_in_darkmode" align=middle width="69.431175pt" height="22.745910000000016pt"/>

* **Gradient for Maximum Likelihood Estimation mathematical derivation:**
    
    + Gradients for log-likelihood of the following model:

        - we have <img alt="$X \in \mathbf R^{n \times k}$" src="svgs/31fcff5b0e78f373e638ada8861be796.png?invert_in_darkmode" align=middle width="74.603265pt" height="27.852989999999977pt"/> - constant data matrix, <img alt="$\mathbf x_i$" src="svgs/eb9d118bc54ee2366c55493ee8aabe04.png?invert_in_darkmode" align=middle width="14.573460000000003pt" height="14.55728999999999pt"/> - vector corresponding to a single data point

        - <img alt="$\theta$" src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.png?invert_in_darkmode" align=middle width="8.143030500000002pt" height="22.745910000000016pt"/> is a <img alt="$k$" src="svgs/63bb9849783d01d91403bc9a5fea12a2.png?invert_in_darkmode" align=middle width="9.041505pt" height="22.745910000000016pt"/>-dimensional (unknown) weight vector

        - <img alt="$\varepsilon \sim \text{Student}(v)$" src="svgs/7066e6b06265b54d66e0e86af9540a85.png?invert_in_darkmode" align=middle width="106.936995pt" height="24.56552999999997pt"/> is a <img alt="$n$" src="svgs/55a049b8f161ae7cfeb0197d75aff967.png?invert_in_darkmode" align=middle width="9.830040000000002pt" height="14.102549999999994pt"/>-dimensional (unknown) noise vector

        - and we observe vector <img alt="$\mathbf y = X\theta + \varepsilon$" src="svgs/d6ac5774c6ea62f6121ee0f02fd47e4b.png?invert_in_darkmode" align=middle width="82.75509pt" height="22.745910000000016pt"/>

        - <p align="center"><img alt="$$ P(y_i \ | \ \mathbf x_i, \theta, v) = \frac{1}{Z(v)} \Big(1 + \frac{(\theta^T \mathbf x_i - y_i) ^2}{v}\Big)^{-\frac{v+1}{2}}$$" src="svgs/6cf71cdcab445dace4fe24c91db57650.png?invert_in_darkmode" align=middle width="332.8182pt" height="40.754504999999995pt"/></p>

    + Stochastic Gradient Descent Implementation

* **Matrix Derivatives mathematical derivation:**
    
    + Multivariate Gaussian:

        - <img alt="$ \frac{\partial \mathcal L(\Sigma)}{\partial \Sigma} = -\frac12 \left( \frac{1}{|\Sigma|} |\Sigma| \Sigma^{-T}  - \Sigma^{-T} (x- \bar \mu)(x-\bar \mu)^T\Sigma^{-T} \right) =  -\frac12 \left(\Sigma^{-T}  - \Sigma^{-T} (x- \bar \mu)(x-\bar \mu)^T\Sigma^{-T} \right)$" src="svgs/e53b5bde0af9d03f0a3cf84561e9c432.png?invert_in_darkmode" align=middle width="675.14337pt" height="37.803480000000015pt"/>

    + Multi-target Linear Regression model:

        - we have <img alt="$X \in \mathbf R^{n \times k}$" src="svgs/31fcff5b0e78f373e638ada8861be796.png?invert_in_darkmode" align=middle width="74.603265pt" height="27.852989999999977pt"/> is a constant data matrix

        - <img alt="$\theta$" src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.png?invert_in_darkmode" align=middle width="8.143030500000002pt" height="22.745910000000016pt"/> is a <img alt="$k \times m$" src="svgs/36290fecb73181e2ffc443f92d17d1bb.png?invert_in_darkmode" align=middle width="43.466115pt" height="22.745910000000016pt"/>-dimentional weight matrix

        - <img alt="$\varepsilon_{ij} \sim \mathcal N(0, \sigma_\epsilon)$" src="svgs/f6fcf8a0c6c05d839a1b5055209d7997.png?invert_in_darkmode" align=middle width="100.81730999999999pt" height="24.56552999999997pt"/> is a normal noise (<img alt="$i \in [0; n], j \in [0;m]$" src="svgs/b44013e0c84e47c8475cc4bc078107a1.png?invert_in_darkmode" align=middle width="134.07702pt" height="24.56552999999997pt"/>)

        - and we observe a matrix <img alt="$Y = X\theta + \varepsilon \in \mathbf R^{n \times m}$" src="svgs/bf6feeef5a1a3ac6a62fb23d2603cb69.png?invert_in_darkmode" align=middle width="149.841285pt" height="26.124119999999984pt"/>

        - <p align="center"><img alt="$$\varepsilon = Y - X\theta \sim \mathcal N_n(0, \sigma_\epsilon I)$$" src="svgs/68a4bfa409aad95a73cb3ef68cb7520c.png?invert_in_darkmode" align=middle width="182.3316pt" height="16.376943pt"/></p>

        - <p align="center"><img alt="$$\mathcal L(\theta) = \log P(Y - X\theta \ | \ \theta) = \log \mathcal N_n(Y - X\theta \ | \ 0, \sigma_\epsilon I)$$" src="svgs/a0793fe6d8a7f414ab981012609eedb1.png?invert_in_darkmode" align=middle width="375.71985pt" height="16.376943pt"/></p>

        - <p align="center"><img alt="$$\theta_{MLE} = \arg \max_{\theta} \mathcal L(\theta) = \arg \min_{\theta} \text{loss}(\theta) = \arg \min_{\theta} \big( ||Y-X\theta||^2_F \big)$$" src="svgs/5968614cfb104b54ad4490800f53095b.png?invert_in_darkmode" align=middle width="457.51695pt" height="24.905594999999998pt"/></p>

        - Deriavation: <img alt="$\frac{\partial\text{loss}(\theta)}{\partial \theta} = -2X^T (Y-X\theta)$" src="svgs/63348dde1425d4b5baba614715f782a2.png?invert_in_darkmode" align=middle width="184.23372pt" height="33.14091000000001pt"/>

        - Deriavation: <img alt="$\theta_{MLE} = (X^T X)^{-1} X^T Y$" src="svgs/ca3e9320514eecdd426cae0a7c94edb2.png?invert_in_darkmode" align=middle width="172.268745pt" height="27.598230000000008pt"/>

* **Logistic Regression mathematical derivation**


* **Logistic Regression implementation**

<a name="sec2"></a>
<h2><a href="https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/Section2/Section2.ipynb">Section 2 Content - Deep Learning Basics in Math, Numpy and Scikit-Learn (click to view full notebook)</a></h2>

* **Coding requirements:**

```python
# Python 3.5+
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from sklearn.datasets import fetch_mldata

```

* **Cross-Entropy and Softmax mathematical derivation:**
    
    + Minimizing the multiclass cross-entropy loss function to obtain the maximum likelihood estimate of the parameters <img alt="$\theta$" src="svgs/27e556cf3caa0673ac49a8f0de3c73ca.png?invert_in_darkmode" align=middle width="8.143030500000002pt" height="22.745910000000016pt"/>:

        - <img alt="$L(\theta)= - \frac{1}{N}\sum_{i=1}^{N} \sum_{k=1}^{K} y_{ik} \log(h_k(x_i,\theta))$" src="svgs/e8f0fe3e582f1f3ffe16e13dd8e386ec.png?invert_in_darkmode" align=middle width="290.002845pt" height="32.19743999999999pt"/> where <img alt="$N$" src="svgs/f9c4988898e7f532b9f826a75014ed3c.png?invert_in_darkmode" align=middle width="14.944050000000002pt" height="22.381919999999983pt"/> is the number of examples <img alt="$\{x_i,y_i\}$" src="svgs/e187a9ac17643462c5faecf4772fc9ba.png?invert_in_darkmode" align=middle width="52.00073999999999pt" height="24.56552999999997pt"/>

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
<h2><a href="https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/Section3/Section3.ipynb">Section 3 Content - Deep Learning Basics in Tensoflow (click to view full notebook)</a></h2>

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
<h2><a href="https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/Section4/Section4.ipynb">Section 4 Content - Deep Learning Advanced  in Tensoflow (click to view full notebook)</a></h2>

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
<h2><a href="https://github.com/rexwangcc/Joyful-Deep-Leaning/blob/master/Section5/Section5.ipynb">Section 5 Content (click to view full notebook)</a></h2>
