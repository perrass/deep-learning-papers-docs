### Systematic evaluation of CNN advances on the ImageNet and ResNet

#### Baseline

91.070%

#### Bias

* True
* False

#### Batch Normalization and ReLU

* Before ReLU
* Before PReLU
* After PReLU

#### ReLU

* ReLU
* ELU: $f(x) = max(0, x) + min(0, alpha * (exp(x)-1))$
* SELU: $f(x) = scale * max(0, x) + min(0, alpha * (exp(x)-1))$, where $scale=1.0507009873554804934193349852946, alpha=1.6732632423543772848170429916717$
* PReLU: $f(x) = max(0, x) + a * min(0, x)$
  * num_parameters is the num of a for sharing, usually be 1 or nChannels. **The difference of channel-shared and channel-wise is not quiet obserse**
  * weight decay should not be used when learning “a” for good performance, due to **a weight decay tends to push $a$ to zero, and thus biases PReLU to ReLU**
* LReLU: $f(x) = max(0, x) + a * min(0, x)$, but with no num_parameters, and $a$ is fixed

#### Pooling

* MaxPooling 3 3 2
* MaxPooling 2 2 2
* MaxPooling + Average Pooling

#### Learning rate decay policy

* step
* square
* square root
* linear

#### Architectures

* ResNet34
* ResNet50
* ResNet101
* ResNet152