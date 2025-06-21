import tensorflow as tf

def create_look_ahead_mask(size):
    mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def masked_attention(q, k, v, dtype=None, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (batch_size, num_heads, seq_len_q, seq_len_k)

    # Scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], dtype)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += ((1.0-tf.cast(mask, dtype)) * -1e4)  # Apply mask: set future positions to -inf

    # Softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (batch_size, num_heads, seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len_q, depth_v)

    return output

# simple blanket mask
def simple_mask(logits, idx, dtype=None):
    return logits + tf.one_hot(idx, depth = logits.shape[2], dtype=dtype) * -1e4
