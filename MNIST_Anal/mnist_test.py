import tensorflow as tf

m=tf.keras.models.load_model('./model')

print(m.summary())
