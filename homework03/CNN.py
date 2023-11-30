import tensorflow as tf
import keras
from keras import layers, losses, optimizers


class CNN(tf.keras.Model): 
    def __init__(self, input_size=(64, 64, 3), num_filters=[1], kernel_size=[(3,3)], strides=[(1,1)], conv_activation=['relu'], cnn_pool_type=["max_pool"], 
                 padding=["valid"], use_bias=[False], dense_activation=['relu'], dense_sizes=[256], num_classes=10, flatten_type="global_max", name="CNN model"):

        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv_activation = conv_activation
        self.cnn_pool_type = cnn_pool_type
        self.padding = padding
        self.use_bias = use_bias
        self.dense_activation = dense_activation
        self.dense_sizes = dense_sizes
        self.num_classes = num_classes
        self.flatten_type = flatten_type
        self.name = name

        self.model = None
        self.create_cnn()
        self.compile_cnn()



    def create_cnn(self):
        inputs = layers.Input(shape=(self.input_shape), dtype=tf.float32)
        x = inputs

        # Create CNN part with pooling layers
        for i, num_filter in enumerate(self.num_filters):
            x = layers.Conv2D(num_filter, self.kernel_size[i], self.strides[i], self.padding[i], activation=self.conv_activation[i], use_bias=self.use_bias[i])(x)
            if self.cnn_pool_type is "max_pool":
                x = layers.MaxPool2D()(x) #TODO: make the pool size adjustable too?
            else: 
                x = layers.AveragePooling2D()(x) #TODO: make the pool size adjustable too?
        
        # Flatten the output of the CNN part for it to fit into the MLP part
        if self.flatten_type is "global_max":
            x = layers.GlobalMaxPool2D()(x)
        else: 
            x = layers.GlobalAveragePooling2D()(x)

        # Create the MLP part
        for i, dense_size in enumerate(self.dense_sizes):
            x = layers.Dense(dense_size, activation=self.dense_activation[i])(x)

        # Create the output part
        y = layers.Dense(units=self.num_classes, activation='softmax')(x)

        self.model = keras.Model(inputs=inputs, outputs=y, name=self.name)
        self.model.summary()



    def __call__(self, x):
        return self.model(x)
    

    def compile_cnn(self):
        self.model.compile(loss=losses.CategoricalCrossentropy(), optimizer= optimizers.RMSprop(), metrics=["accuracy"])
    

    def train(self, x_train, y_train, x_test, y_test, batchsize=64, epochs=2, validation_split=0.2):
        history = self.model.fit(x_train, y_train, batch_size=batchsize, epochs=epochs, validation_split=validation_split)
        test_scores = self.model.evaluate(x_test, y_test)
        return test_scores
    

    def get_model(self):
        return self.model
    
