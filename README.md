# Convolutional Supervised Autoencoder - CSAE
This GitHub repository contains the Keras implementation on the MNIST Dataset and the Supplementary Material of the [https://arxiv.org/abs/2208.12152v1][Supervised Dimensionality Reduction and Classification with Convolutional Autoencoders].

## Experiment on MNIST Dataset
### Usage
1. Clone the code to local machine.
```
git clone https://github.com/JohnNellas/CSAE
```
2. Create the environment from the ```environment.yml``` file.
```
conda env create -f environment.yml
```
3. Activate the environment.
```
conda activate csae_env
```
4. Run the experiment on MNIST dataset and produce a *dimensions_latent_space* dimensional latent space.
```
python3 main.py dimensions_latent_space
```

### Example

Run the experiment for a CSAE with a 10D Latent Space.
```
python3 main.py 10
```
### Folder Creation
The script execution produces a checkpoint folder along with a folder that contains the classification and visualization results, which are entitled as ```checkpoints_classification_ae``` and ```classification_and_visualization_results_CSAE```, correspondingly.

### Classification Results and Visualizations
The script execution involves the evaluation of the utilized classification methodologies and the creation of visualizations that emerge from the Convolutional Supervised Autoencoder.

#### Classification Results
The Classification methodologies are evaluated during the script execution and the results are saved in the ```classification_results``` sub folder of the directory ```classification_and_visualization_results_CSAE```.

#### Visualizations
Additionally, the script execution creates a set of visualizations. Initially, a Latent Space visualization is conducted. In more detail, if the derived Latent Space has a dimensionality equal to two (2) or three (3), then a 2D or 3D scatter plot of the raw embeddings is constructed respectively, while for a dimensionality greater than 3, the Latent Space is visualized by creating a 2D scatter plot of the embeddings produced by applying the PCA or t-SNE algorithms onto the resulting Latent Representations. Moreover, concerning the case where the Latent Space of CSAE is two dimensional, a 2D scatterplot of the raw embeddings along with the Classifier's Decision boundary is constructed, where some points are replaced by the corresponding original images, while for the same case, a figure of the images derived from decoding a grid of latent representations is also generated. Finally, a plot of the accuracy, reconstruction and classification error per epoch during the training and validation procedure is constructed. The resulting visualizations are saved in the ```visualization_results``` sub folder of the directory ```classification_and_visualization_results_CSAE```.  Some examples of the afforemetioned figures, for the two dimensional latent space are presented below.
![](classification_and_visualization_results_CSAE/visualization_results/visualization_results_MNIST_num_class_10_lat_dims_2/mnist_network_embedded_space_decision_boundary_2D_latent_dim_10_classes.png)
![](classification_and_visualization_results_CSAE/visualization_results/visualization_results_MNIST_num_class_10_lat_dims_2/MNNIST_decoded_grid_of_points.png)

## Supplementary Material
Finally, the supplementary material is contained in the pdf file entitled as ```supplementary_material.pdf```.
