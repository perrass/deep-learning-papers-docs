### YOLO9000: Better, Faster, Stronger

**Motivation**: improve recall and localization while maintaining classification accuracy

* Error analysis of YOLO compared to Fast R-CNN shows that YOLO makes a significant number of localization errors.
* YOLO has relatively low recall compared to region proposal-based methods

#### Better

* Batch Normalization(2%), add bn on all convolutional layers, and remove dropout from the model without overfitting
* High Resolution Classifier(4%) when training network, 448/448 for 10 epochs on ImageNet, and then fine-turning the classifier
* Convolutional with Anchor Boxes. YOLO predicts the coordinates of bounding boxes directly using fully convolutional layers on top of the convolutional feature extracto. While, Faster R-CNN uses the regional proposal network to predict offsets and confidence scores for anchor boxes. **Since the prediction layer is convolutional, the RPN predicts these offsets at every location in a feature map**. Predicting offsets instead of coordinates simplifies the problem and makes it easier for the network to learn. (可能是因为全连接层失去了spatial information)
  * Eliminate one pooling layer to make the output of the network's convolutional layers high resolution
  * Shrink the input images to 416/416 to ensure the output size is odd (13/13). This leads to there exsits a centre pixels in the outputs. In object detection, this pixel or block is important because most of objects are on there.
  * Decouple the class prediction from spatial location and use class and objectness prediction to prediction **anchor boxes**(only about anchor boxes). The objectness prediction still predicts the IOU of the ground truth and the proposed box and the class predictions predict the conditional probability of that class given that there is an object.
* Dimension Clusters. The issure of  anchor box is how to set an good prior of bounding box, or a good dimension. YOLO uses k-means to clusters pixels and get a good bounding box prior.
  * K is set to 5 or 9, YOLOv2 is 5 and YOLOv3 is 9
  * Distance metric is customized, because the Euclidean distance would generate more error in larger boxes compared with small ones. **What we really want are priors that lead to good IOU scores**, hence, the metric is $d(box, centroid) = 1 - IOU(box, centroid)$
* Direct location prediction(5% with anchor boxes and k-means). The other issue of anchor box is model instability
  * $b_x = \sigma(t_x) + c_x$
  * $b_y = \sigma(t_y) + c_y$
  * $b_w = p_we^{t_w}​$
  * $b_h = p_he^{t^h}$
  * $Pr(object) * IOU(b, object) = o(t_o)$
  * where, $t_x, t_y, t_w, t_h$ are predictions from the network for each bounding box
  * $c_x, c_y$ are the top left corner of the image
  * $p_w, p_h$ are the width and height of the bounding box priors
  * $\sigma()$ is logistic activation
* Fine-Grained Features (1%), concatenates the higher resolution features with lower resolution layers. That is, the route layer in offical configuration file
* Multi-scale Training. Every 10 batches our network randomly chooses a new image dimension size. This controls the trade-off between accuracy and speed.

#### Faster

* Darknet19
* DNN is replaced by CNN

#### Stronger

#### YOLO V3

YOLOv3 predicts an objectness score for each bounding box using logistic regression, 每个ground truth只配一个框. 因为对有很多重复类别的数据集(比如people和woman), softmax并不适用, 因为softmax假设每个类别都是独立的

9个而非5个anchor bbox prior

3个detection, 13/13, 26/26, 52/52

Darknet-19到Darknet-53

