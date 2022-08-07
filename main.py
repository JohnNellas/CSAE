import tensorflow as tf
import numpy as np
import os
import csae_model as csae
import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.manifold
import sklearn.manifold
import sklearn.decomposition
import sklearn.metrics
import sklearn.neighbors
import sklearn.svm
import sklearn.naive_bayes

# check the number of parameters
# if a wrong number of parameters is given, print a usage message and exit
if len(sys.argv) != 2:
    print(f"<Usage> python3 {sys.argv[0]} number_of_latent_dimensions")
    sys.exit()

# try to convert the argument from a string number to an integer 
# if conversion fails print an error message and exit
try:
    latent_dims = int(sys.argv[1])
except Exception as e:
    print("Please provide valid input")
    sys.exit()

# if a non positive number for the Latent Space's number of dimensions is provided
# print an error message and exit
if latent_dims <= 0:
    print("Please provide a positive number as the number of dimensions in the Latent Space")
    sys.exit()

# load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
num_classes = 10

# Set the value of some training parameters
epochs, batch_size = 200, 128
learning_rate = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# create a validation set that is equal to the 10% of the train set
val_size = 0.1
x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(x_train,
                                                                        y_train,
                                                                        test_size=val_size,
                                                                        shuffle=True,
                                                                        stratify=y_train,
                                                                        random_state=42)

# Normalize the images
x_train = x_train/255.0
x_val   = x_val/255.0
x_test  = x_test/255.0

# Create the Channels Last format
x_train = np.expand_dims(x_train, axis=-1)
x_val   = np.expand_dims(x_val, axis=-1)
x_test  = np.expand_dims(x_test, axis=-1)

print("shape of training set:", x_train.shape)
print("shape of validation set", x_val.shape)
print("shape of test set", x_test.shape)

# Build the Convolutional Autoencoder and the Classifier Network
autoencoder, class_model = csae.build_test_model((x_train.shape[1], x_train.shape[2], x_train.shape[3]),
                                                 latent_dims=latent_dims,
                                                 number_of_classes=num_classes)


# Define the Loss Functions for the Convolutional Autoencoder and the Classifier Network
loss_function_autoencoder = tf.keras.losses.MeanSquaredError()
loss_function_classifier = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# Define the Metrics for the Convolutional Autoencoder and the Classifier Network
ae_metric = tf.keras.metrics.MeanSquaredError(name="ae_loss")
class_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=False,
                                                            name="class_loss")

# Build the Convolutional Supervised Autoencoder Model
model = csae.CSAE(autoencoder, class_model)
model.build((None, x_train.shape[1], x_train.shape[2], x_train.shape[3]))

# see a summary of the model
model.summary()

# Compile the Model
model.compile(optimizer,
            optimizer,
            loss_function_classifier,
            loss_function_autoencoder,
            class_metric,
            ae_metric)


# Create directories to save results and checkpoints 
checkpoint_parent_path = os.path.join(".", "checkpoints_classification_ae")
if not os.path.isdir(checkpoint_parent_path):
    os.mkdir(checkpoint_parent_path)

checkpoint_path = os.path.join(checkpoint_parent_path, f"checkpoints_ae_classification_{num_classes}_{latent_dims}_classes")
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

classification_results_parent_path = os.path.join(".", "classification_and_visualization_results_CSAE")
if not os.path.isdir(classification_results_parent_path):
    os.mkdir(classification_results_parent_path)

classification_results_middle_path = os.path.join(classification_results_parent_path, "classification_results")
if not os.path.isdir(classification_results_middle_path):
    os.mkdir(classification_results_middle_path)

visualization_results_middle_path = os.path.join(classification_results_parent_path, "visualization_results")
if not os.path.isdir(visualization_results_middle_path):
    os.mkdir(visualization_results_middle_path)

classification_results_path = os.path.join(classification_results_middle_path, f"classification_results_MNIST_num_class_{num_classes}_lat_dims_{latent_dims}")
if not os.path.isdir(classification_results_path):
    os.mkdir(classification_results_path)

visualization_results_path = os.path.join(visualization_results_middle_path, f"visualization_results_MNIST_num_class_{num_classes}_lat_dims_{latent_dims}")
if not os.path.isdir(visualization_results_path):
    os.mkdir(visualization_results_path)

# Define a Learning Rate Scheduler
def lr_schedule(epoch, lr):
    if ((epoch+1)%50) == 0:
        return lr/3
    else:
        return lr

# define the callbacks
callbacks=[tf.keras.callbacks.ModelCheckpoint(os.path.join(checkpoint_path, "class_best_model_weights.tf"),
                                            monitor="val_accuracy",
                                            save_best_only=True,
                                            save_weights_only=True,
                                            mode="max",
                                            verbose=1
                                            ),
            tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)
            ]

# Train the CSAE Model
history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_val, y_val),
                    shuffle=True,
                    callbacks=callbacks,
                    verbose=1)


# =============================================================================
#                    VISUALIZATION  OF THE TRAINING PROCEDURE
# =============================================================================

# Create a Plot of the validation loss, classification error and accuracy per epoch
plt.plot(history.epoch, history.history["val_ae_loss"])
plt.plot(history.epoch, history.history["val_class_loss"])
plt.plot(history.epoch, history.history["val_accuracy"])
plt.legend(["Recon. Loss", "Class. Loss", "Accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Training Procedure: Validation")
plt.savefig(os.path.join(visualization_results_path,f"mnist_train_procedure_validation_{num_classes}_classes.png"),
            bbox_inches='tight',
            dpi=300
            )
plt.close()

# Create a Plot of the training loss, classification error and accuracy per epoch
plt.plot(history.epoch, history.history["ae_loss"])
plt.plot(history.epoch, history.history["class_loss"])
plt.plot(history.epoch, history.history["accuracy"])
plt.legend(["Recon. Loss", "Class. Loss", "Accuracy"])
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.title("Training Procedure: Training")
plt.savefig(os.path.join(visualization_results_path,f"mnist_train_procedure_train_{num_classes}_classes.png"),
            bbox_inches='tight',
            dpi=300
            )
plt.close()


# merge back the trainining and validation set and remove the validation set
x_train = np.concatenate([x_train, x_val], axis=0)
y_train = np.concatenate([y_train, y_val], axis=0)

del x_val, y_val

# Loading best weights - criterion: highest validation accuracy
print("Loading the best weights...")
model.load_weights(os.path.join(checkpoint_path, "class_best_model_weights.tf"))

# Predict the probability of each class for each image in the test set 
# and assign it to the class of highest probability
predictions = model.class_model.predict(x_test)
classes = np.argmax(predictions,axis=1)

# Acquire the Evaluation Results of the Classifier and save them to a file
print("Evaluating...")
print(sklearn.metrics.classification_report(y_test, classes))
res_dict = sklearn.metrics.classification_report(y_test, classes, output_dict=True)
print(f"Analytically, Accuracy: {res_dict['accuracy']}, weighted_avg_f1: {res_dict['weighted avg']['f1-score']}")

with open(os.path.join(classification_results_path, f"MNIST_network_output_class_report_{num_classes}_classes.txt"), "w") as f:
    print(sklearn.metrics.classification_report(y_test, classes),
          f"\n\nAnalytically, Accuracy: {res_dict['accuracy']}, weighted_avg_f1: {res_dict['weighted avg']['f1-score']}",
          file=f)

# Acquire the Encoder Network of CSAE and compute the Latent Representations of the images in the test set
encoder = tf.keras.Model(inputs=model.autoencoder.input,
                        outputs=model.autoencoder.get_layer("LatentLayer").output)
embeddings = encoder.predict(x_test)

if latent_dims  == 2:
    # =============================================================================
    #                                DECODER GRID 
    # =============================================================================
    print("Constructing the Decoded Grid of Latent Representations...")
    save_path = os.path.join(visualization_results_path, "MNNIST_decoded_grid_of_points.png")
    number_of_points = 20
    decoder_first_layer_name = "Decoder3"
    csae.decode_grid_of_latent_representations(model, embeddings, number_of_points, save_path, decoder_first_layer_name)
    
    #=============================================================================
    #                             Decision Boundary 
    # =============================================================================
    print("Constructing a scatter plot of the Latent Representations along with the derived Classifier's Decision Boundary")
    print("where some points are replaced by the corresponding original image...")
    classifier_first_layer_name = "Class1"
    zoom_figX_figY = (0.5, 12, 10)
    save_path = os.path.join(visualization_results_path,f"mnist_network_embedded_space_decision_boundary_2D_latent_dim_{num_classes}_classes.png")
    csae.decision_boundary_on_latent_space(model, embeddings, x_test, y_test, zoom_figX_figY, classifier_first_layer_name, save_path)

# =============================================================================
#                         Latent Space Visualizations  
# =============================================================================
print("Creating the Latent Space Visualization...")
# if the number of dimensions of the Latent Space is greater than 3
# do a 2D scatter plot of the aforementioned space using t-SNE and PCA
if latent_dims > 3:
    
    # Make a 2D scatter plot of the embeddings produced by the PCA algorithm
    # coloured with the ground truth classes
    print("Performing PCA to visualize Embedding Space...")
    embeddings_pca = sklearn.decomposition.PCA(n_components=2).fit_transform(embeddings)

    df = pd.DataFrame(data=embeddings_pca, columns=["PC1", "PC2"])
    df["Y"] = y_test

    sns.scatterplot(x="PC1", y="PC2", hue="Y", data=df, palette="bright", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.axis("Off")
    plt.tight_layout()

    # save it to a file
    plt.savefig(os.path.join(visualization_results_path, f"MNIST_network_embedded_space_2D_PCA_latent_dim_num_classes_{num_classes}_classes.png"),
                bbox_inches='tight',
                dpi=300)
    plt.close()

    # Make a 2D scatter plot of the embeddings produced by the t-SNE algorithm
    # coloured with the ground truth classes
    print("Performing t-SNE to visualize Embedded Space...")
    embeddings_tsne = sklearn.manifold.TSNE(n_components=2,
                                            random_state=42).fit_transform(embeddings)
    df = pd.DataFrame(data=embeddings_tsne, columns=["E1", "E2"])

    df["Y"] = y_test

    sns.scatterplot(x="E1", y="E2", hue="Y", data=df, palette="bright", alpha=0.4)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.axis("off")
    plt.tight_layout()

    # save it to a file
    plt.savefig(os.path.join(visualization_results_path, f"MNIST_network_embedded_space_2D_TSNE_latent_dim_num_classes_{num_classes}_classes.png"),
                bbox_inches='tight',
                dpi=300)
    plt.close()

# if the number of dimensions of the Latent Space is equal to 2
# do a 2D scatter plot of the raw embeddings
if latent_dims == 2:
    # Make a 2D Scatter plot of the raw embeddings
    # coloured with the ground truth class
    df = pd.DataFrame(data=embeddings, columns=["E1", "E2"])
    df["Y"] = y_test

    sns.scatterplot(x="E1", y="E2", hue="Y", data=df, palette="bright", alpha=0.4)
    plt.axis("off")

    # save it to a file
    plt.savefig(os.path.join(visualization_results_path,f"MNIST_network_embedded_space_2D_latent_dim_num_classes_{num_classes}_classes.png"),
                bbox_inches='tight',
                dpi=300)
    plt.close()


# if the number of dimensions of the Latent Space is equal to 3
# do a 3D scatter plot of the raw embeddings
if latent_dims == 3:
    
    # Make a 3D Scatter plot of the raw embeddings
    # coloured with the ground truth class
    df = pd.DataFrame(data=embeddings, columns=["E1", "E2", "E3"])
    df["Y"] = y_test

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    scatters = list()
    for cur_class in np.unique(y_test):
        ind = (y_test==cur_class)
        scat = ax.scatter(df['E1'][ind], df['E2'][ind], df['E3'][ind], alpha=0.6)
        scatters.append(scat)

    ax.legend(scatters, np.unique(y_test))
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)

    plt.tight_layout()

    # save it to a file
    plt.savefig(os.path.join(visualization_results_path, f"MNIST_network_embedded_space_3D_latent_dim_num_classes_{num_classes}_classes.png"),
                bbox_inches='tight',
                dpi=300)
    plt.close()

# =================================================================================================================
#                     Improving Traditional Classification Methods with CSAE's Latent Representations
# =================================================================================================================

print("------------------------------ Improving Traditional Classification Methods with CSAE's Latent Representations -----------------------------------------")
# Get the embeddings of the training set
embeddings_train = encoder.predict(x_train)

print("---------------------------- k-Nearest Neighbors ---------------------")

# Train a k-Nearest Neighbors Classifier with number of neighbors equal to 3
# on the test set's embeddings derived from CSAE
n_neighbors = 3
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors).fit(embeddings, y_test)

# Predict the class of the test set's Latent Representations constructed by CSAE using the trained classifier 
predictions = knn.predict(embeddings)

# Acquire the Evaluation Results of the Classification
print(sklearn.metrics.classification_report(y_test, predictions))
res_dict = sklearn.metrics.classification_report(y_test, predictions, output_dict=True)
print(f"Analytically, Accuracy: {res_dict['accuracy']}, weighted_avg_f1: {res_dict['weighted avg']['f1-score']}")

# Save it to a file
with open(os.path.join(classification_results_path, f"MNIST_KNN_{n_neighbors}_{num_classes}_classes.txt"), "w") as f:
    print(sklearn.metrics.classification_report(y_test, predictions),
          f"\n\nAnalytically, Accuracy: {res_dict['accuracy']}, weighted_avg_f1: {res_dict['weighted avg']['f1-score']}",
          file=f)

del predictions, knn

print("---------------------------- SVM with RBF ---------------------")

# Train a Support Vector Machine Classifier with the rbf kernel
# on the train set's Latent Representations derived from CSAE
svms = sklearn.svm.SVC(random_state=42).fit(embeddings_train, y_train)

# Predict the class of the test set's Latent Representations constructed by CSAE using the trained classifier
predictions = svms.predict(embeddings)

# Acquire the Evaluation Results of the Classification
print(sklearn.metrics.classification_report(y_test, predictions))
res_dict = sklearn.metrics.classification_report(y_test, predictions, output_dict=True)
print(f"Analytically, Accuracy: {res_dict['accuracy']}, weighted_avg_f1: {res_dict['weighted avg']['f1-score']}")

# Save it to a file
with open(os.path.join(classification_results_path, f"MNIST_SVM_{num_classes}_classes.txt"), "w") as f:
    print(sklearn.metrics.classification_report(y_test, predictions),
          f"\n\nAnalytically, Accuracy: {res_dict['accuracy']}, weighted_avg_f1: {res_dict['weighted avg']['f1-score']}",
          file=f)

del predictions, svms

print("---------------------------- Naive Bayes ---------------------")

# Train a Gaussian Naive Bayes Classifier
# on the train set's Latent Representations derived from CSAE
gnb = sklearn.naive_bayes.GaussianNB().fit(embeddings_train, y_train)

# Predict the class of the test set's Latent Representations constructed by CSAE using the trained classifier
predictions = gnb.predict(embeddings)

# Acquire the Evaluation Results of the Classification
print(sklearn.metrics.classification_report(y_test, predictions))
res_dict = sklearn.metrics.classification_report(y_test, predictions, output_dict=True)
print(f"Analytically, Accuracy: {res_dict['accuracy']}, weighted_avg_f1: {res_dict['weighted avg']['f1-score']}")

# Save it to file
with open(os.path.join(classification_results_path, f"MNIST_GNB_{num_classes}_classes.txt"), "w") as f:
    print(sklearn.metrics.classification_report(y_test, predictions),
          f"\n\nAnalytically, Accuracy: {res_dict['accuracy']}, weighted_avg_f1: {res_dict['weighted avg']['f1-score']}",
          file=f)

del predictions, gnb
