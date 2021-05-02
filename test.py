import tensorflow as tf
BCE = tf.keras.losses.BinaryCrossentropy()
print(BCE([[0.9]]*2, [[1.0], [1.0]]))
print(BCE([[0.9]]*2, [[0.9], [0.9]]))
print(BCE([[0.9]]*2, [[0.1], [0.1]]))
print(BCE([[0.9]]*2, [[0.0], [0.0]]))
print(BCE([[0.9]]*2, [[1.0], [0.0]]))
print(BCE([[0.9]], [[0.0]]))
