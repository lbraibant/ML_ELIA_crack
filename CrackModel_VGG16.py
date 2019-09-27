import tensorflow as tf

## MODEL = VGG16 encoder + average over filters
#                + dense layer with one output neuron
base_model = tf.keras.applications.VGG16(input_shape=(224,224,3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


for layer in base_model.layers:
    print(layer.name, layer.input_shape, layer.output_shape)