# VisualExplanationMethods
Some visual explanation methods. 
## Include: 
### gradient_based
#### Backpropagation:
**Description:**
  Only Do backward <br/>
**url:**
 https://github.com/huangchuanhong/VisualExplanationMethods/blob/master/grad-cam-pytorch/grad_cam.py &nbsp;  class BackPropation <br/>
**origin github:**
 https://github.com/kazuto1011/grad-cam-pytorch
#### Deconv
**Description:**
 When Do backward, exchange Relu backward with Relu. <br/>
**url:**
 1.https://github.com/huangchuanhong/VisualExplanationMethods/tree/master/VisualizingCNN <br/>
 2.https://github.com/huangchuanhong/VisualExplanationMethods/blob/master/grad-cam-pytorch/grad_cam.py &nbsp;  class Deconvnet <br/>
**paper:**
 2014 ECCV paper "Visualizing and understanding convolutional networks" <br/>
 paper url: https://arxiv.org/abs/1311.2901 <br/>
**origin github**
 1.https://github.com/huybery/VisualizingCNN <br/>
 2.https://github.com/kazuto1011/grad-cam-pytorch
#### GuidedBackPropagation
**Description:**
 When Do backward, add a Relu after each Relu backward. <br/>
**url:**
 https://github.com/huangchuanhong/VisualExplanationMethods/blob/master/grad-cam-pytorch/grad_cam.py &nbsp;  class GuidedBackPropatation <br/>
**paper:**
 "Striving for simplicity: The All Convolutional net" <br/>
 paper url: https://arxiv.org/abs/1412.6806 <br/>
**origin github:**
 https://github.com/kazuto1011/grad-cam-pytorch
#### CAM(Class Activation Map)
 **url:** 
 https://github.com/huangchuanhong/VisualExplanationMethods/tree/master/CAM <br/>
 **paper:** 
 2016 CVPR paper "Learning Deep Features for Discriminative Localization" <br/>
 paper url: https://arxiv.org/abs/1512.04150 <br/>
 **origin github:**
 https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
#### Grad-CAM 
 **url:** 
  https://github.com/huangchuanhong/VisualExplanationMethods/tree/master/grad-cam-pytorch/grad_cam.py &nbsp; class GradCAM <br/>
 **paper:**
  Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization <br/>
  paper url: https://arxiv.org/abs/1610.02391 <br/>
 **origin github:**
  https://github.com/kazuto1011/grad-cam-pytorch
#### GI(Gradient\*input)、IG(Integrated Gradient)、SG(SmoothGrad)
 **url:**
  https://github.com/huangchuanhong/VisualExplanationMethods/blob/master/sam <br/>
 **paper:**
  "SAM: The Sensitivity of Attribution Methods to Hyperparameters" <br/>
  paper url: https://arxiv.org/abs/2003.08754 <br/>
 **origin github:**
  https://github.com/anguyen8/sam
### perturbation_base
#### Lime
 **url:**
  1.https://github.com/huangchuanhong/VisualExplanationMethods/tree/master/lime_example <br/>
  2.https://github.com/huangchuanhong/VisualExplanationMethods/blob/master/sam/LIME_Madry.py <br/>
 **paper:**
  "Why should I trust you? Explaining the Predictions of Any Classifier" <br/>
  paper url: https://arxiv.org/abs/1602.04938 <br/>
 **origin github:**
  https://github.com/anguyen8/sam
#### MP 
 **url:**
  https://github.com/huangchuanhong/VisualExplanationMethods/blob/master/sam/MP_MADRY.py <br/>
 **paper:**
  "Interpretable Explanations of Black Boxes by Meaningful Perturbation" <br/>
  paper url: https://arxiv.org/abs/1704.03296 <br/>
 **origin github:**
  https://github.com/anguyen8/sam
#### SP(Sliding Path, Occlusion)
 **url:**
  https://github.com/huangchuanhong/VisualExplanationMethods/blob/master/sam <br/>
 **paper:**
  "SAM: The Sensitivity of Attribution Methods to Hyperparameters" <br/>
  paper url: https://arxiv.org/abs/2003.08754 <br/>
 **origin github:**
  https://github.com/anguyen8/sam
### SAM
   **url:**
  https://github.com/huangchuanhong/VisualExplanationMethods/blob/master/sam <br/>
 **paper:**
  "SAM: The Sensitivity of Attribution Methods to Hyperparameters" <br/>
  paper url: https://arxiv.org/abs/2003.08754 <br/>
 **origin github:**
  https://github.com/anguyen8/sam
