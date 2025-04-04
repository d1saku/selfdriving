import tensorflow as tf
from tensorflow.keras import layers

class PNetCNN(tf.keras.Model):
    def __init__(self, steering_head=True, speed_head=True):
        """the model is adapted from nvidia's paper on end to end training a car
        which can be found by the link following:
        https://arxiv.org/pdf/1604.07316

        Args:
            steering_head (bool, optional): controls if the model architecture predicts steering values. Defaults to True.
            speed_head (bool, optional): controls if the model predicts speed values. Defaults to True.
        """
        super(PNetCNN, self).__init__()
        
        assert steering_head or speed_head, "model should have at least one prediction head"

        self.steering = steering_head
        self.speed = speed_head

        # Define the convolutional layers with VALID padding to match PyTorch
        self.conv1 = layers.Conv2D(24, kernel_size=5, strides=2, padding='valid', activation='relu')
        self.conv2 = layers.Conv2D(36, kernel_size=5, strides=2, padding='valid', activation='relu')
        self.conv3 = layers.Conv2D(48, kernel_size=5, strides=2, padding='valid', activation='relu')
        self.conv4 = layers.Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu')
        self.conv5 = layers.Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu')
        
        # Define the flatten operation
        self.flatten = layers.Flatten()
        
        # Define the fully connected layers
        self.fc1 = layers.Dense(100, activation='relu')
        self.fc2 = layers.Dense(50, activation='relu')
        self.fc3 = layers.Dense(10, activation='relu')
        
        # Define the dropout layer
        self.dropout = layers.Dropout(0.5)
        
        # Define the output heads
        if self.steering:
            self.steering_head = layers.Dense(1, activation='sigmoid', name='angle')
        
        if self.speed:
            self.speed_head = layers.Dense(1, activation='sigmoid', name='speed')
    
    def call(self, inputs, training=False):
        # Ensure input has the right shape (66, 200, 3)
        x = self.conv1(inputs)  # Output: (31, 98, 24)
        x = self.conv2(x)       # Output: (14, 47, 36)
        x = self.conv3(x)       # Output: (5, 22, 48)
        x = self.conv4(x)       # Output: (3, 20, 64)
        x = self.conv5(x)       # Output: (1, 18, 64)
        
        # Flatten to match PyTorch's expected 1152 dimensions
        x = self.flatten(x)     # Output: 1152
        
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        x = self.fc3(x)
        
        outputs = {}
        
        if self.steering:
            outputs['angle'] = self.steering_head(x)
        
        if self.speed:
            outputs['speed'] = self.speed_head(x)
        
        return outputs
    
    @staticmethod
    def count_parameters(model):
        """Count the number of trainable parameters in a model"""
        return sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

# Example of how to build and compile the model
def build_model(input_shape=(66, 200, 3), steering_head=True, speed_head=True):
    # Create input tensor with the EXACT shape expected (66, 200, 3)
    # The model shape calculation is very sensitive to this
    inputs = tf.keras.Input(shape=input_shape)
    
    model = PNetCNN(steering_head=steering_head, speed_head=speed_head)
    outputs = model(inputs)
    
    # Create model with the functional API
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Define loss functions and metrics for each output
    losses = {}
    metrics = {}
    
    if steering_head:
        losses['angle'] = 'mse'
        metrics['angle'] = ['mae']
    
    if speed_head:
        losses['speed'] = 'mse'
        metrics['speed'] = ['mae']
    
    # Compile the model
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=losses,
        metrics=metrics
    )
    
    return keras_model