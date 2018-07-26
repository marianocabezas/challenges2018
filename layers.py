from keras.engine.topology import Layer


class ScalingLayer(Layer):

    def __init__(self, **kwargs):
        super(ScalingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.w = self.add_weight(name='kernel',
                                      shape=input_shape[1:],
                                      initializer='uniform',
                                      trainable=True)
        self.b = self.add_weight(name='kernel',
                                 shape=input_shape[1:],
                                 initializer='normal',
                                 trainable=True)
        super(ScalingLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x * self.w + self.b