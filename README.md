# GraphX-Convolution for Point Cloud Deformation in 2D-to-3D conversion

## 22-bit image
<p style="text-align:right;"> <b> Aman Rojjha </b> - 2019111018 <br> 
<b> Tejas Chaudhari </b>  - 2019111013 </p>

---

# Introduction
3D shape reasoning is important in the field of computer vision as it plays a vital role  in robotics, modeling, graphics, and so on.  With the current methods, we are able to estimate reliable shapes of the object using multiple images from different viewpoints. But, we humans can reasonably estimate the shape of an object from a single image. Our task is to train machines to do so.   
This seems unlikely to be solved since some information is lost when we go from 3D to 2D, however if a machine is able to learn a shape prior like us humans, then it would be able to infer 3D shapes from 2D images reliably.   


The key properties of the system which models this behavior  are:
- Model should make predictions based on not only local features but also high-level semantics
- Model should consider spatial correlation between points
- Method should be scalable/output point cloud can be of arbitrary size

This approach is better than the original paper we had taken "A Point Set Generation Network for  
3D Object Reconstruction from a Single Image, Fan et al" since it is far more scalable. Fan et al had proposed an encoder-decoder architecture with various shortcuts to directly map an input image to its point cloud representation. The disadvantage was that the number of trainable parameters are proportional to the number of points in the output cloud since it  directly generates a point cloud. In contrast, in this approach we deform a point cloud instead of generating one, which makes the system far more scalable.

---

# Approach
![[arch-1.png]]
The overall approach is as follows:
1) Encode the input object image using a CNN to extract multi-scale feature map
2) Distill global and point specific shape information from the features and blend it into a randomly generated point cloud
3) Use the deformation network on the mixture to obtain the 3D point cloud

## Image encoding
- VGG(VGG stands for Visual Geometry Group; it is a standard deep Convolutional Neural Network (CNN) architecture with multiple layers) like architecture.
- Feed-forward network without any shortcuts from lower layers
- Consists of several spatial down-samplings and channel up-samplings at the same time
- Gives multi-scale representation of the original image

## Feature blending
### Point specific shape information
- Extracts a feature vector for each individual point by projecting the points onto the feature maps

### Global shape information
- Processes the initial point cloud with a simple multi-layer perceptron (MLP) encoder composed of several blocks of fully connected (FC) layers to obtain features at multiple scales.  

![[1.png]]

### Point cloud feature extraction
- To get a single feature vector for each point, concatenate the two features with the point coordinates together.

## Point cloud deformation
- Produces a point cloud representation of the input object via an NN.
- Generating a precise and representative point cloud requires some communication between points in the set
	- The  $X$ convolution fits this purpose as it is carried out in a neighborhood of each point, but it is computationally expensive as it runs the k-nearest neighbor every iteration
	- Graph convolution considers the local interactions of the vertices, but is designed for mesh representation which requires adjacency matrix
- The paper proposes an amalgamation of  the two operators called graphX-convolution (GraphX) which has similar functionality as the graph convolution, but works on unordered point sets like X-conv.  

![[2.png]]

---

# Results
```ad-info
Since the dataset was huge ($234$ GBs uncompressed), we've trained and tested our model for a subset ($24$ GBs) and got decent results with the default 80-20 split.
```

> Refer to [this link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/aman_rojjha_research_iiit_ac_in/Eocm4Q_ViTpIq_hbC9-F8C8BZuXzjAkZproDCQ7qYxT_Zw?e=K4VcGU) to obtain the respective trained models and test results from the model.

#### Training Loss 
For training purposes, we have opt to consider **Chamfer Loss** as our loss metric.

![[results/train-chamfer.jpg]]

## Testing Loss
> Refer to [this file](../results/find_test_chamfer_loss_avg.sh) for finding respective chamfer loss for the final model.
> 
- Test loss on final model: $0.283819$.

## Model Output Comparision
| Example | **Ground truth** | **Model Prediction** |
| ---: | --- | ---|
|  1. |  ![[graphx-conv/src/results/test-and-demo/plots/04256520/0/ground/40.jpg]]|![[graphx-conv/src/results/test-and-demo/plots/04256520/0/pred/40.jpg]] |
| 2. | ![[graphx-conv/src/results/test-and-demo/plots/04256520/0/ground/87.jpg]] | ![[graphx-conv/src/results/test-and-demo/plots/04256520/0/pred/87.jpg]] |
| 3. |  ![[graphx-conv/src/results/test-and-demo/plots/04256520/1/ground/90.jpg]] | ![[graphx-conv/src/results/test-and-demo/plots/04256520/1/pred/90.jpg]] |
| 4. | ![[graphx-conv/src/results/test-and-demo/plots/04256520/14/ground/89.jpg]] | ![[graphx-conv/src/results/test-and-demo/plots/04256520/14/pred/89.jpg]] |
| 5. | ![[graphx-conv/src/results/test-and-demo/plots/04256520/9/ground/35.jpg]] | ![[graphx-conv/src/results/test-and-demo/plots/04256520/9/pred/35.jpg]] |
| 6. | ![[graphx-conv/src/results/test-and-demo/plots/04256520/9/ground/80.jpg]] | ![[graphx-conv/src/results/test-and-demo/plots/04256520/9/pred/80.jpg]]|

### Interactive Point Clouds
To visualize the respective models as *3D-point clouds*, kindly download the [results.zip](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/aman_rojjha_research_iiit_ac_in/Eocm4Q_ViTpIq_hbC9-F8C8BZuXzjAkZproDCQ7qYxT_Zw?e=K4VcGU) to download pre-processeed test-set point clouds and load them using [visualizer.py](../graphx-conv/visualizer.py).

