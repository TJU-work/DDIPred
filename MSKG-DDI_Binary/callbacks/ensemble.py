import os
from keras.callbacks import Callback


class SWA(Callback):

    def __init__(self, swa_model, checkpoint_dir, model_name, swa_start=1):

        super(SWA, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.swa_start = swa_start
        self.swa_model = swa_model

    def on_train_begin(self, logs=None):
        self.epoch = 0
        self.swa_n = 0

        self.swa_model.set_weights(self.model.get_weights())

    def on_epoch_end(self, epoch, logs=None):
        if (self.epoch + 1) >= self.swa_start:
            self.update_average_model()
            self.swa_n += 1

        self.epoch += 1

    def update_average_model(self):
        alpha = 1. / (self.swa_n + 1)
        for layer, swa_layer in zip(self.model.layers, self.swa_model.layers):
            weights = []
            for w1, w2 in zip(swa_layer.get_weights(), layer.get_weights()):
                weights.append((1 - alpha) * w1 + alpha * w2)
            swa_layer.set_weights(weights)

    def on_train_end(self, logs=None):
        print(f'Logging Info - Saving SWA model checkpoint: {self.model_name}_swa.hdf5')
        self.swa_model.save_weights(os.path.join(self.checkpoint_dir,
                                                 f'{self.model_name}_swa.hdf5'))
        print('Logging Info - SWA model Saved')
