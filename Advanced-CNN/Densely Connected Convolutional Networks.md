### Densely Connected Convolutional Networks

As CNNs become increasingly deep, a new research problem emerges: as information about the input or gradient passes through many layers, it can vanish and “wash out” by the time it reaches the end (or beginning) of the network.

The key characteristic to solve this is to create short paths from early layers to later layers.

Instead of drawing representational power from extremely deep or wide architectures, **DenseNets exploit the potential of the network through feature reuse**, yielding condensed models that are easy to train and highly parameter-efficient

The architechture of traditional CNN is $x_l = H_l(x_{l-1})$, and the architechture of ResNet is $x_l = H_l(x_{l-1}) + x_{l-1}$. Hence, an advantage of ResNet is that the gradient can flow directly through the identity function from later layers to the earlier layers. That of DenseNet is $x_l = H_l([x_0, x_1, ..., x_{l-1}])$.

However, the concatenation operation used in DenseNet is not viable when the size of feature-maps changes. However, an essential part of convolutional networks is down-sampling layers that change the size of feature-maps.

---

原因

* 最直观: 特征图越多, 模型的表示能力越强
* 其次, down-sampling代表某种特征选择或者转换的过程, 类似于机器学习里的特征工程. 也就是说, 深度学习的特征工程自动化, 是将机器学习的特征工程分为了两步, 对输入(输出)的处理, 特征层的增多

---

We refer to layers between blocks as transition layers, which do convolution and pooling. The transition
layers used in our experiments consist of a batch normalization layer and an 1×1 convolutional layer followed by a
2×2 average pooling layer. These layers change the size of feature-maps