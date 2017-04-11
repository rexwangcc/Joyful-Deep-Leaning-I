<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>
<script type="text/javascript" async src="path-to-mathjax/MathJax.js?config=TeX-AMS_CHTML"></script>
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-MML-AM_CHTML">
</script>




Joyful Deep Learning
=========================

This repository is a collection of my useful resources of Deep Learning in Tensorflow, in 5 sections.

# Section 1 Content - Machine Learning and Deep Learning Basics in Math and Numpy

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

# Section 2 Content - Deep Learning Basics in Math, Numpy and Scikit-Learn

* **Coding requirements:**

```python
# Python 3.5+
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from sklearn.datasets import fetch_mldata

```
# Section 3 Content 

# Section 4 Content 

# Section 5 Content 