import tensorflow as tf

def create_look_ahead_mask(size):
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def masked_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)

    # Scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add the mask to the scaled tensor.
    if mask is not None:
        mask = tf.expand_dims(mask, axis=1)  # (batch_size, 1, seq_len, seq_len)
        mask = tf.expand_dims(mask, axis=1)  # (batch_size, num_heads, seq_len, seq_len)
        scaled_attention_logits += (mask * -1e9)  # Apply mask: set future positions to -inf

    # Softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
    print(attention_weights.shape)
    print(v.shape)
    output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth_v)

    return output