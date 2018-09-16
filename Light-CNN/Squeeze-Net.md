## Squeeze-Net

### Motivation

A CNN architecture with fewer parameters has several advantages:

* More efficient distributed training. **For distributed data-parallel training, communication overhead is directly proportional to the number of parameters in the model**
* Less overhead when exporting new models to clients. For autonomous driving, companies such as Tesla periodically copy new models from their servers to customers' cars. This practice is often referred to **over-the-air** update. However, over-the-air updates of today's typical CNN/DNN models can require large data transfers.
* Feasible FPGA and embedded deployment. FPGAs often has less than **10MB of on-chip memory and no off-chip memory or storage**. 

### Related Work

#### Model compression

Network Pruning, e.g sparse CNN, Deep Compression with Huffman encoding and quantization

#### CNN Micro-architecture

Modify the filter from large to small $5\times 5 -\to 3\times 3 \to 1\times 1$ 

#### CNN Macro-architecture

While the CNN micro-architecture refers to individual layers and modules, we define the CNN macro-architecture as the system-level organization of multiple modules into an end-to-end CNN architecture.

The depth of CNN, or the connections across multiple layers or modules, like Res-Net

### Squeeze-Net

#### Architectural design strategies

**Replace 3 3 filters with 1 1 filters**

**Decrease the number of input channels to 3 3 filters**: The total quantity of parameters in this layer is (number of input channels) * (number of filters) * (3*3). We decrease the number of input channels to 3 3 filters using **squeeze layer**

**Down-sample late in the network so that convolution layers have large activation maps**.  Most commonly, down-sampling is engineered into CNN architectures by setting the $stride\ge 2$ in some of the convolution or pooling layers. If early layers in the network have large strides, then most layers will have small activation maps. Conversely, **If late layers have large strides, then most layers will have large activation maps, and then the accuracy would be higher**

Strategy 1 and 2 are about judiciously decreasing the quantity of parameters in a CNN while attempting to preserve accuracy. Strategy 3 is about maximizing accuracy on a limited budget of parameters.

#### The Fire Module

A Fire module is comprised of: a **squeeze convolution layer**, feeding into an **expand** layer that has a mix of 1 1 and 3 3 convolution filters. Then we have three hyper-parameters $s_{1x1}, e_{1x1}, e_{3x3}$. $s_{1x1}$ is the number of filters in the squeeze layer, $e_{1x1}$ is the number of 1 1 filters in the expand layer, and $e_{3x3}$ is the number of  3 3 filters in the expand layer. The squeeze layer helps to limit the number of input channels to the 3 3 filters, if $s_{1x1} < e_{1x1} + e_{3x3}$

#### Squeeze-Net

![](assets/squeezenet.png)

Details

1. Activation is ReLU
2. The output activations from 1 1 and 3 3 filters have the same height and width
3. Dropout with a ratio of 50% is applied after the fire9 module
4. Using global average pooling instead of FC

Results

The Squeeze-Net with simple bypass has the highest accuracy and the same model size as vanilla Squeeze-Net. 