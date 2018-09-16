## ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices

### Motivation

The state-of-the-art basic architectures such as Xception and ResNeXt become less efficient in extremely small networks because of the costly dense $1\times 1$ convolutions. That is, the depthwise convolution has a computational cost $D_k\cdot D_k \cdot M \cdot D_F \cdot D_F$, and the pointwise convolution has a computational cost $M\cdot N \cdot D_F \cdot D_F$. It seems that the depthwise convolution has more parameters, but usually, the number of output feature map $N$ is much larger that the kernel size $D_k \cdot D_k$.

We propose using **pointwise group convolutions** instead to reduce computation complexity. To overcome the side effects brought by pointwise group convolutions, we come up with a novel **channel shuffle** operation to help the information flowing across feature channels.

### Approach

Xception -> depthwise separable convolutions; ResNeXt -> group convolutions