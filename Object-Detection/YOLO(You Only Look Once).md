### YOLO(You Only Look Once)

#### Benefits

* Fast: frame detection as a regression problem
* YOLO sees the entire image during training and test time so it  implicitly encodes contextual information about classes as well as their appearance. Fast R-CNN, a top detection method, mistakes back-ground patches in an image for objects because it can't see the larger context.
* This leads to YOLO learns generalizable representations of objects
* However, it struggles to precisely localize some objects, especially small ones\

#### Unified Detection

1. Divide the input image into an $S \times S$ grid. If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object
2. Each grid cell predicts B bounding boxes and confidence scores for those boxes.
   1. **Confidence score** reflects how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts. $Pr(Object) \times IOU_{pred}^{truth}$
   2. Each bounding box consists of 5 predictions: $x, y, w, h, confidence$
      1. $x, y$ coordinates represent the center of box relative to the bounds of the grid cell
      2. w, h are the width and height predicted relative to the whole image
      3. confidence prediction represents the IOU between the predicted box and any ground truth box
   3. Each grid cell also predicts C conditional class probabilities, $Pr(Class_i|Object)$. These probabilities are conditioned on the grid cell containing an object. **We only predict one set of class probabilities per grid cell**
3. At test time, we get class-specific confidence scores for each box $Pr(Class_i|Object) \times Pr(object)\times IOU_{pred}^{truth} = Pr(Class_i) \times IOU^{truth}_{pred}$

**It divides the image into an $S\times S$ grid and for each grid cell predicts $B$ bounding boxes, confidence for those boxes, and $C$ class probablities. These predictions are encoded as an $S\times S\times (B\times 5  +C)$ tensor**. E.g., in PASCAL VOC(20 labels), we use s = 7, b = 2, and the final prediction is a $7\times 7\times30$ tensor

#### Architecture



#### Training

* Using pretrained model from ImageNet 2012 and fine turning
* Double the input size
* Normalize the bounding box width and height by the image wide and height
* Parametrize the bounding box x and y coordinates to be offsets of a particular grid cell location
* Leaky ReLu
* Using sum-squared error for the sake of optimization, but it is not perfectly align with our goal of maximizing average precision. 
  * However, for grid cells that do not contain any object. This pushes the confidence socre of those cells towards zero, often overpowering the gradient from cells that do contain objects. Hence, two parameters $\lambda_{coord}(5), \lambda_{noobj})(0.5)$ are added.
  * Sum-squared error also equally weights errors in large boxes and small boxes. Our error metric should reflect that small deviations in large boxes matter less than in small boxes. To partially address this we predict the square root of the bounding box width and height instead of the width and height directly
  * loss function
* Bs = 64, momentum = 0.9, decay = 0.0005, dp = 0.5 after the first connected layer
* random scaling and translations of up to 20% of the original image size, randomly adjust the exposure and saturation of the image by up to a factor of 1.5 in the HSV color space

#### Limitations

* Struggles with small objects that appear in groups
* struggles to generalize to objects in new or unusual aspect ratios or configurations
* A small error in a large box is generally benign but a small error in a small box has a much greater effect on IOU. Our main source of error is incorrect localizations