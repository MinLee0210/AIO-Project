from data import *
from model import Transformer

'''
    1. n_layers: numbers of transformer encoder-decoder block.
    2. d_model: embedding diments of a token represented in transformer.
    3. d_ff: numbers of node at the hidden layer of the Feed-Forward layer.
    4. n_heads: numbers of head in multi-head attention layers.
    5. dropout_rate: 
    6. vocab_size: 
'''

N_LAYERS = 4
D_MODEL = 128
D_FF = 512
N_HEADS = 8
DROPOUT_RATE = 0.2

transformer = Transformer (
    num_layers = N_LAYERS, d_model = D_MODEL,
    num_heads = N_HEADS, dff =D_FF,
    vocab_size = VOCAB_SIZE, dropout_rate = DROPOUT_RATE
)

batches = train_ds.take(1)
for batch in batches:
    X_try, y_try = batch[0], batch[1]
    break

output = transformer(X_try)


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        result = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        return result
    
learning_rate = CustomSchedule(D_MODEL)

def masked_loss(label, pred):
    mask = label != 0
    loss_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    result = tf.reduce_sum(loss)/tf.reduce_sum(mask)

    return result

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)

optimizer = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

EPOCHS = 70

transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
history = transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)