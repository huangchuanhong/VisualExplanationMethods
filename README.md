# VisualExplanationMethods
Some visual explanation methods. 
## Include: 
### gradient_based
#### Backpropagation:
**Description:**
  Only Do backward
**url:**
 https://github.com/huangchuanhong/VisualExplanationMethods/blob/master/grad-cam-pytorch/grad_cam.py &nbsp;  class BackPropation <br/>
**origin github:**
 https://github.com/kazuto1011/grad-cam-pytorch
#### Deconv
**Description:**
 When Do backward, exchange Relu backward with Relu.
**url:**
 1.https://github.com/huangchuanhong/VisualExplanationMethods/tree/master/VisualizingCNN <br/>
 2.https://github.com/huangchuanhong/VisualExplanationMethods/blob/master/grad-cam-pytorch/grad_cam.py &nbsp;  class Deconvnet <br/>
**paper**
 2014 ECCV paper "Visualizing and understanding convolutional networks" <br/>
 paper url: https://arxiv.org/abs/1311.2901
**origin github**
 1.https://github.com/huybery/VisualizingCNN <br/>
 2.https://github.com/kazuto1011/grad-cam-pytorch
#### GuidedBackPropagation
**Description:**
 When Do backward, add a Relu after each Relu backward.
**url:**
 https://github.com/huangchuanhong/VisualExplanationMethods/blob/master/grad-cam-pytorch/grad_cam.py &nbsp;  class GuidedBackPropatation <br/>
**paper:**
 "Striving for simplicity: The All Convolutional net" <br/>
 paper url: https://arxiv.org/abs/1412.6806
**origin github:**
 https://github.com/kazuto1011/grad-cam-pytorch
#### CAM(Class Activation Map)
 **url:** https://github.com/huangchuanhong/VisualExplanationMethods/tree/master/CAM <br/>
 2016 CVPR paper "Learning Deep Features for Discriminative Localization" <br/>
 paper url: https://arxiv.org/abs/1512.04150
 origin github: https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
#### Grad-CAM 
 url: https://github.com/huangchuanhong/VisualExplanationMethods/tree/master/grad-cam-pytorch
 paper: Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
 paper url: https://arxiv.org/abs/1610.02391
 origin github: https://github.com/kazuto1011/grad-cam-pytorch
### perturbation_base
#### Lime
 url: https://github.com/huangchuanhong/VisualExplanationMethods/tree/master/Lime
 paper: 
#### MP 
### SAM
