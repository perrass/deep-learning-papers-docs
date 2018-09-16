### Delving Deep into Rectifiers Surpassing Human-Level Performance on ImageNet Classification

PReLU improves model fitting with nearly zero extra computational cost and little overfitting risk

First, the first conv layer (conv1) has coefficients (0.681 and 0.596) significantly greater than 0. As the fil-
ters of conv1 are mostly Gabor-like filters such as edge or texture detectors, the learned results show that both positive and negative responses of the filters are respected. We believe that this is a more economical way of exploiting low-
level information, given the limited number of filters (e.g.,64) (你只能理解到, 确实是exploiting更多的信息, 但是为什么是low-level information ?)

Second, for the channel-wise version, the deeper convlayers in general have smaller coefficients. This implies that
the activations gradually become “more nonlinear” at increasing depths. In other words, the learned model tends to
keep more information in earlier stages and becomes more discriminative in deeper stages. (说明激活函数越来越非线性, 且, 越高的卷积层保持的信息越少. **How to understand information**)

Xavier initialization is based on the assumption that the activations are linear. This assumption is invalid for ReLU and PReLU