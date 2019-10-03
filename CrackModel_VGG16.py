import tensorflow as tf
from ML_utils import imageClassifierDataSet
import os
import matplotlib.pyplot as plt



## REPOSITORY WITH DATA
main_dir = "C:\\Users\\lorra\\Projets\\ML_detection_fissures\\ML_ELIA_crack\\"
data_dir = main_dir+"data\\Pylone_Classification\\"

## CREATE DATASETS
test_size = 0.2
batch_size = 32
random_seed = 100
# negative class = without crack
temp_dir = os.path.join(data_dir,"uncracked_concrete")
data_paths = []
for filename in os.listdir(temp_dir):
    assert os.path.exists(os.path.join(temp_dir.split(main_dir)[-1], filename))
    data_paths.append(os.path.join(temp_dir.split(main_dir)[-1], filename))
nnegative = len(data_paths)
data_labels = ["negative"] * nnegative
# positive class = with crack
temp_dir = os.path.join(data_dir,"cracked_concrete")
for filename in os.listdir(temp_dir):
    assert os.path.exists(os.path.join(temp_dir.split(main_dir)[-1], filename))
    data_paths.append(os.path.join(temp_dir.split(main_dir)[-1], filename))
npositive = len(data_paths) - nnegative
data_labels += ["positive"] * npositive
ndata = len(data_paths)
print("Concrete Pylon dataset : %i images, %i in  positive class and %i in negative class" %
      (ndata,npositive,nnegative))

dataset = imageClassifierDataSet(data_paths, data_labels, dataset_path=None,
                                 data_directory=main_dir,target_size=(224,224),color_mode="rgb")
tr_gen, vl_gen, ts_gen = dataset.get_train_validation_test(test_size=test_size, validation_size=0., train_size=None,
                                                           stratified=True, random_state=random_seed,
                                                           batch_size=batch_size, verbose=True)

class_mapping = tr_gen.class_indices
print(class_mapping)

## MODEL = VGG16 encoder + average over filters
base_model = tf.keras.applications.VGG16(input_shape=(224,224,3),include_top=False,weights='imagenet')
base_model.trainable = False
model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(len(class_mapping), activation='softmax')
])
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss='binary_crossentropy',
              metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),
                       tf.keras.metrics.AUC()])
# Set checkpointing
filepath = "log/weihgts-{epoch:02d}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, save_freq=1,
                                                verbose=0)
callbacks_list = [checkpoint]
# Train
epochs = 10
steps_per_epoch = tr_gen.n // batch_size
validation_steps = ts_gen.n // batch_size
history = model.fit_generator(tr_gen, steps_per_epoch=steps_per_epoch, epochs=epochs,
                              validation_data=ts_gen, validation_freq=1, validation_steps=validation_steps,
                              callbacks=callbacks_list, class_weight={0:1.,1:10.}, verbose=1)


acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')