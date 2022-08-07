from numpy import save
import tensorflow as tf
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns

class CSAE(tf.keras.Model):
    def __init__(self, autoencoder, class_model, **kwargs):
        """
        autoencoder: The autoencoder model
        class_model: The classifier model
        """
        super(CSAE,self).__init__(**kwargs)
        self.autoencoder = autoencoder
        self.class_model = class_model


    def compile(self, class_optimizer, ae_optimizer, class_loss_fn, ae_fn, class_metric, ae_metric):
        super(CSAE, self).compile()
        """
            class_optimizer: The optimizer for the classifier Network
            ae_optimizer: The optimizer for the Convolutional Autoencoder network
            class_loss_fn: The Loss Functions of the Classifier Network
            ae_fn: The Loss Functions of the Convolutional Autoencoder Network
            class_metric: The Metric for the classifier Network
            ae_metric: The Metric for the Convolutional Autoencoder network
        """
        self.class_optimizer = class_optimizer
        self.ae_optimizer = ae_optimizer
        self.class_loss_fn = class_loss_fn
        self.ae_loss_fn = ae_fn
        self.ae_metric = ae_metric
        self.class_metric = class_metric
        self.acc_metric = tf.keras.metrics.Accuracy()

    @property
    def metrics(self):
        return [self.ae_metric, self.class_metric, self.acc_metric]

    def train_step(self, data):
        x_data, y_data = data

        # AUTOENCODER PART
        with tf.GradientTape() as tape:
            reconstructions = self.autoencoder(x_data, training=True)
            loss_value = self.ae_loss_fn(x_data, reconstructions)

        grads = tape.gradient(loss_value, self.autoencoder.trainable_weights)
        self.ae_optimizer.apply_gradients(zip(grads, self.autoencoder.trainable_weights))

        # CLASSIFIER PART
        with tf.GradientTape() as tape:
            predictions = self.class_model(x_data, training=True)
            loss_value_class = self.class_loss_fn(y_data, predictions)

        grads_class = tape.gradient(loss_value_class, self.class_model.trainable_weights)
        self.class_optimizer.apply_gradients(zip(grads_class, self.class_model.trainable_weights))

        # update states
        self.ae_metric.update_state(x_data, reconstructions)
        self.class_metric.update_state(y_data, predictions)
        self.acc_metric.update_state(y_data, tf.math.argmax(predictions,axis=1))

        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        x_data, y_data = data
        
        # compute the reconstructions and the predictions
        reconstructions = self.autoencoder(x_data, training=False)
        predictions = self.class_model(x_data, training=False)

        # update states
        self.ae_metric.update_state(x_data, reconstructions)
        self.class_metric.update_state(y_data, predictions)
        self.acc_metric.update_state(y_data, tf.math.argmax(predictions,axis=1))


        return {m.name: m.result() for m in self.metrics}

    def call(self, data):
        return self.autoencoder(data), self.class_model(data)


def build_test_model(input_shape: tuple, latent_dims: int, number_of_classes: int):
    
    first_filter, second_filter = 32, 64
    input_layer = tf.keras.Input(shape=input_shape, name="InputLayer")
    
    # ENCODER PART
    x = tf.keras.layers.Conv2D(filters=first_filter,
                               kernel_size=(3,3),
                               strides=2,
                               activation="relu", 
                               padding="same",
                               name="CONV1")(input_layer)
    
    x = tf.keras.layers.Conv2D(filters=second_filter,
                               kernel_size=(3,3),
                               strides=2,
                               activation="relu", 
                               padding="same",
                               name="CONV2")(x)
    
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(units=128,
                              activation="relu",
                              name="Encoder1")(x)

    x = tf.keras.layers.Dense(units=128,
                              activation="relu",
                              name="Encoder2")(x)


    enc = tf.keras.layers.Dense(units=latent_dims,
                                activation="linear",
                                name="LatentLayer")(x)
    
    # CLASSIFIER PART
    enc1 = tf.keras.layers.Dense(units=128,
                                activation="relu",
                                name="Class1")(enc)

    enc1 = tf.keras.layers.Dense(units=128,
                                activation="relu",
                                name="Class2")(enc1)

    classification_layer = tf.keras.layers.Dense(units=number_of_classes,
                                                 activation="softmax",
                                                 name="ClassLayer")(enc1)

    # DECODER PART
    x = tf.keras.layers.Dense(units=128,
                              activation="relu",
                              name="Decoder3")(enc)

    x = tf.keras.layers.Dense(units=128,
                              activation="relu",
                              name="Decoder2")(x)
    
    x = tf.keras.layers.Dense(units=second_filter*(input_shape[0]//4)*(input_shape[1]//4),
                              activation="relu",
                              name="FlatToSpatial")(x)
    
    
    x = tf.keras.layers.Reshape((input_shape[0]//4, input_shape[0]//4, second_filter))(x)
    

    x = tf.keras.layers.Conv2DTranspose(filters=first_filter,
                                       kernel_size=(3,3),
                                       strides=2,
                                       activation="relu",
                                       padding="same",
                                       name="CONV2DEC")(x)
    
    output = tf.keras.layers.Conv2DTranspose(filters=input_shape[-1],
                                       kernel_size=(3,3),
                                       strides=2,
                                       activation="sigmoid",
                                       padding="same",
                                       name="CONV1DEC")(x)

    # create the convolutional autoencoder and the classifier network respectively
    autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=output, name="Autoencoder")
    class_model = tf.keras.models.Model(inputs=input_layer, outputs=classification_layer, name="ClassificationModel")

    # see a summary of the architectures
    autoencoder.summary()
    class_model.summary()
    
    return autoencoder, class_model


def decode_grid_of_latent_representations(csae, embeddings:np.array, number_of_points:int, save_path:str, decoder_first_layer_name:str):
    
    # check if the embeddings are 2D
    if embeddings.shape[-1] != 2:
        print("Please provide 2D embeddings")
        return 

    # Get the Decoder model
    decoder_input = tf.keras.Input((2,))
    first = False
    is_first_entry = True
    for layer in csae.autoencoder.layers:
        if first or (decoder_first_layer_name in layer.name):
            if is_first_entry:
                is_first_entry=False
                first=True
                x = layer(decoder_input)
            else:
                x = layer(x)

    decoder = tf.keras.Model(inputs=decoder_input, outputs=x)

    # Create a grid of points
    min_x = np.round(np.min(embeddings[:,0]),2)
    max_x = np.round(np.max(embeddings[:,0]),2)

    min_y = np.round(np.min(embeddings[:,1]),2)
    max_y = np.round(np.max(embeddings[:,1]),2)

    x = np.linspace(min_x, max_x, number_of_points)
    y = np.linspace(min_y, max_y, number_of_points)

    X,Y = np.meshgrid(x, y[::-1])

    # Decode the grid of points and construct a figure of the resulting images
    fig, axis = plt.subplots(nrows=number_of_points,
                            ncols=number_of_points,
                            figsize=(number_of_points+1, number_of_points+1),
                            gridspec_kw={'wspace':0.0, 'hspace':0.0}
                            # squeeze=True
                        )

    for enum, (k,l) in enumerate(zip(X,Y)):
        for enum_cols, (val1, val2) in enumerate(zip(k,l)):
            decode = decoder.predict([[val1,val2]])

            axis[enum, enum_cols].axis("off")
            axis[enum, enum_cols].imshow(decode.squeeze(), cmap="gray")

    # save the figure to a file
    plt.savefig(save_path,
                bbox_inches='tight',
                dpi=300)
    plt.close()

def decision_boundary_on_latent_space(csae, embeddings, images_sample_pool, y_test, zoom_figX_figY:tuple, classifier_first_layer_name:str, save_path:str, cmap="gray", number_of_grid_points=75, number_of_samples_per_class=5, seed_val=42):
    
    # check if the embeddings are 2D
    if embeddings.shape[-1] != 2:
        print("Please provide 2D embeddings")
        return 

    # Isolate the classifier network
    classifier_input = tf.keras.Input((2,))
    first = False
    is_first_entry = True
    for layer in csae.class_model.layers:
        if first or (classifier_first_layer_name in layer.name):
            if is_first_entry:
                is_first_entry=False
                first=True
                x = layer(classifier_input)
            else:
                x = layer(x)

    classifier_net = tf.keras.Model(inputs=classifier_input, outputs=x)

    # create a grid around the embeddings
    min_x = np.round(np.min(embeddings[:,0]),2)
    max_x = np.round(np.max(embeddings[:,0]),2)

    min_y = np.round(np.min(embeddings[:,1]),2)
    max_y = np.round(np.max(embeddings[:,1]),2)

    x = np.linspace(min_x, max_x, number_of_grid_points)
    y = np.linspace(min_y, max_y, number_of_grid_points)

    X,Y = np.meshgrid(x, y)

    # specify the figure settings
    zoom, fig_x, fig_y = zoom_figX_figY

    fig = plt.figure(figsize=(fig_x, fig_y))
    ax  = fig.add_subplot(111)
    
    # create a list of all the 2d grid points
    grid_of_points = list()
    for k,l in zip(X,Y):

        for val1, val2 in zip(k,l):
            grid_of_points.append([val1, val2])

    grid_of_points = np.array(grid_of_points)

    # predict their class with the isolated classifier network
    predictions_probabilities = classifier_net.predict(grid_of_points)
    predictions_grid = np.argmax(predictions_probabilities, axis=1)

    # Do a scatter plot of the classified grid points coloured by the predicted class
    df = pd.DataFrame(data=grid_of_points, columns=["E1", "E2"])
    df["Y"] = predictions_grid
    sns.scatterplot(x="E1", y="E2", hue="Y", data=df, palette="bright", alpha=0.4, legend=False)

    # Do a scatter plot of the raw embeddings coloured by the ground truth class
    df_sc = pd.DataFrame(data=embeddings, columns=["E1", "E2"])
    df_sc["Y"] = y_test
    sns.scatterplot(x="E1", y="E2", hue="Y", data=df_sc, palette="bright")

    # get a random number of samples from each class
    np.random.seed(seed_val)
    samples_each_class = list()
    for uq_class in np.unique(y_test):
        inds_class = np.where(y_test == uq_class)[0]
        samples_of_class = np.random.choice(inds_class, size=(number_of_samples_per_class,), replace=False)
        samples_each_class.append(samples_of_class)
    samples_to_display = np.concatenate(samples_each_class, axis=0)
    
    # replace the sampled points by the corresponding original images in the scatter plot    
    for sample in samples_to_display:
        im = matplotlib.offsetbox.OffsetImage(images_sample_pool[sample].squeeze(), zoom=zoom, cmap=cmap)
        x0, y0 = embeddings[sample]
        ab = matplotlib.offsetbox.AnnotationBbox(im, (x0, y0), frameon=False)
        ax.add_artist(ab)
    plt.axis("off")

    # save it to a file
    plt.savefig(save_path,
                dpi=300,
                bbox_inches='tight')
    plt.close()

    # # Take a Random Sample of a number of images from the images_sample_pool and present it on the latent space
    # np.random.seed(seed_val)
    # samples_to_display = np.random.randint(0, y_test.shape[0], size=number_of_samples)
    # cmap = "gray" if len(images_sample_pool.shape) == 3 or (len(images_sample_pool.shape) == 4 and images_sample_pool.shape[-1]==1) else None
    # for sample in samples_to_display:
    #     im = matplotlib.offsetbox.OffsetImage(images_sample_pool[sample].squeeze(), zoom=zoom, cmap=cmap)
    #     x0, y0 = embeddings[sample]
    #     ab = matplotlib.offsetbox.AnnotationBbox(im, (x0, y0), frameon=False)
    #     ax.add_artist(ab)
    # plt.axis("off")

    # # save the figure to a file
    # plt.savefig(save_path,
    #             bbox_inches='tight',
    #             dpi=300)
    # plt.close()