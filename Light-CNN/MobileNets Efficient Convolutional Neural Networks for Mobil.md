## MobileNets Efficient Convolutional Neural Networks for Mobile Vision Applications

### Motivation

Introduce two simple global hyper-parameters that efficiently trade off between latency and accuracy. 

MobileNets primarily focus on optimizing for latency but also yield small networks. Many papers on small networks focus only on size but do not consider speed.

### Methods for small network

**Network level**

* Flattened networks
* Factorized networks
* Xception networks: scale up depthwise separable filters
* Squeezenet: uses a botteleneck approach

**Shrinking, factorizing or compressing pretrained networks**

* Product quantization
* Hashing
* Pruning, vector quantization and Huffman coding

**Low bit networks**

### MobileNet Architecture

#### Depthwise Separable Convolution

The MobileNet model is based on depthwise separable convolutions which is a form of factorized convolutions which factorize a standard convolution into a depthwise convolution and a 1$\times$1 convolution called a pointwise convolution.

In detail, the depthwise convolution applies a single filter to each input channel. The pointwise convolution then applies a $1\times 1$ convolution to combine the outputs the depthwise convolution. A standard convolution both filters and combines inputs into a new set of outputs in one step. **The depthwise separable convolution splits this into two layers, a separable layer for filtering and a separate layer for combining**. This factorization has the effect of drastically reducing computation and model size.

![](assets/mobilenet_depthwise.png)

A standard convolutional layer takes as input a $D_F\times D_F \times M$ feature map F and produces a $D_F\times D_F \times N$ feature map G (assuming that the output feature map has the same spatial dimensions as the input and both feature maps are square) where M is the number of input channels, N is the number of output channels. And the standard convolutional layer is parameterized by convolution kernel K of size $D_k \times D_k \times M \times N$ where $D_k$ is the spatial dimension of the kernel assumed to be square.

The output feature map for standard convolution assuming stride one and padding is computing as 
$$
G_{k, l, n} = \sum_{i, j, m}K_{i, j, m, n} \cdot F_{k+i-1, l+j-1, m}
$$
Standard convolutions have the computational cost of:
$$
D_k \cdot D_k \cdot M \cdot N \cdot D_F \cdot D_F
$$
where the computational cost depends multiplicatively on the number of input channels M, the number of output channels N, the kernel size $D_k \times D_k$ and the feature map size $D_F \times D_F$. MobileNet models address each of these terms and their interactions. First it uses depthwise separable convolutions to break the interaction between the number of output channels and the size of the kernel.

Depthwise separable convolution are made up of two layers: depthwise convolutions and pointwise convolutions. We use depthwise convolutions to apply a single filter per each input channel. Pointwise convolution, a simple $1\times 1$ convolution, is then used to create a linear combination of the output of the depthwise layer. MobileNets use both batchnorm and ReLU nonlinearities for both layers.

Depthwise convolution with one filter per input channel can be written as 
$$
\hat G_{k, l, m} = \sum_{i, j}\hat K_{i, j, m} \cdot F_{k+i-1, l+j-1, m}
$$
where $\hat K$ is the depthwise convolutional kernel of size $D_k \times D_k \times M$ where the $m_{th}$ filter in $\hat K$ is applied to the $m_{th}$ channel in F to produce the $m_{th}$ channel of the filtered output feature map $\hat G$

Depthwise convolution has a computational cost of 
$$
D_k\cdot D_k \cdot M \cdot D_F \cdot D_F
$$
Then we should concat the kernel with depth 1 into depth N by $1\times 1$ convolution, which is called pointwise convolution. And the total cost is 
$$
D_k \cdot D_k \cdot M \cdot D_F \cdot D_F + M\cdot N \cdot D_F \cdot D_F
$$
And the reduction in computation is 
$$
\frac {D_k \cdot D_k \cdot M \cdot D_F \cdot D_F + M\cdot N \cdot D_F \cdot D_F} {D_k \cdot D_k \cdot M \cdot N \cdot D_F \cdot D_F} = \frac 1 N + \frac 1 {D^2_K}
$$
Or $N+D^2_K : ND^2_K$

#### Network Structure and Training



![](assets/mobilenet_basic_module.png)

![](assets/mobilenet_body.png)

1. No label smoothing
2. Very small or no weight decay on the depthwise filters
3. RMSprop with asynchronous gradient descent similar to Inception V3

#### Width Multiplier: Thinner Models

The role of the width multiplier $\alpha$ is to thin a network uniformly at each layer. For a given layer and width multiplier $\alpha$, the number of input channels M becomes $\alpha M$ and the number of output channels N becomes $\alpha N$.
$$
D_k \cdot D_k \cdot \alpha M \cdot D_F \cdot D_F + \alpha M \cdot \alpha N \cdot D_F \cdot D_F
$$
where $\alpha \in (0, 1]$ with typical settings of 1, 0.75, 0.5, and 0.25. Width multiplier has the effect of reducing computational cost and the number of parameters quadratically by roughly $\alpha^2$. **Width multiplier can be applied to any model structure to define a new smaller model with a reasonable accuracy, latency and size trade off**

**Q: Why $\alpha^2$**

| Type            | Mult-Adds | Parameters |
| --------------- | --------- | ---------- |
| Conv 1 1        | 94.86%    | 74.59%     |
| Conv DW 3 3     | 3.06%     | 1.06%      |
| Conv 3 3        | 1.19%     | 0.02%      |
| Fully Connected | 0.18%     | 24.33%     |

Conv 1 1 ocupies almost 95% mult-adds, so the reduction of computational cost is roughly $\alpha^2$

#### Resolution Multiplier: Reduced Representation

This is applied to the input image and the internal representation of every layer is subsequently reduced by the same multiplier. In practice we implicitly set $\rho$ by setting the input resolution.

Then the computational cost is 
$$
D_k \cdot D_k \cdot \alpha M \cdot \rho D_F \cdot \rho D_F + \alpha M \cdot \alpha N \cdot \rho D_F \cdot \rho D_F
$$
And the input resolution of the network is 224, 192, 160, 128

---

#### Asynchronous gradient descent

#### Synchronous gradient descent

