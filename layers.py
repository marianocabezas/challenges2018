from keras import backend as K
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


class ThresholdingLayer(Layer):
    def __init__(self, thresholds, **kwargs):
        self.thresholds = thresholds
        super(ThresholdingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ThresholdingLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        min_cat = K.less(x, self.thresholds[0])
        max_cat = K.greater_equal(x, self.thresholds[-1])
        other_cat = map(
            lambda (th1, th2): K.all(K.stack([K.greater_equal(x, th1), K.less(x, th2)], axis=0), axis=0),
            zip(self.thresholds[:-1], self.thresholds[1:])
        )
        return K.cast(K.concatenate([min_cat] + other_cat + [max_cat]), K.floatx())

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1 + len(self.thresholds))
