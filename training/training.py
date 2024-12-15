import tensorflow

def trainModel(model, inputs, targets):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            targets, logits, from_logits = True
        ))
    gradients = tape.gradient(loss, model.Trainables)
    optimizer = tf.keras.optimizers.Adam()
    optimizer.apply_gradients(zip(gradients, model.Trainables))
    return
