#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# In[2]:


def dataset_processing(address):

    data = []
    for folder in os.listdir(address):
        folder = address + '/' + str(folder)
        if os.path.isdir(folder):
            for temp_part in os.listdir(folder):
                part = temp_part
                part_address = folder + '/' + str(temp_part)
                for temp_id in os.listdir(part_address):
                    id = temp_id
                    id_address = part_address + '/' + str(temp_id)
                    for temp_label in os.listdir(id_address):
                     if temp_label.split('_')[-1] == 'positive':
                            label = 'fractured'
                     elif temp_label.split('_')[-1] == 'negative':
                            label = 'normal'
                            label_address = id_address + '/' + str(temp_label)
                            for temp_image in os.listdir(label_address):
                                img_address = label_address + '/' + str(temp_image)
                                data.append(
                                    {
                                        'Label': part,
                                        'Image_address': img_address
                                    }
                                )
    return data


# In[3]:


Labels = ["ELBOW","FINGER","FOREARM","HAND","HUMERUS","SHOULDER","WRIST"]

directory = r'C:\Users\ankit\OneDrive\Desktop\capstone-final\Bone Fracture Identification\Bone Fracture Identification'+'/dataset'
dataset = dataset_processing(directory)
list_labels = []
list_paths = []


for row in dataset:
    list_labels.append(row['Label'])
    list_paths.append(row['Image_address'])

list_paths = pd.Series(list_paths, name='Path').astype(str)
list_labels = pd.Series(list_labels, name='Label')

image_data = pd.concat([list_paths, list_labels], axis=1)


data_train, data_test = train_test_split(image_data, train_size=0.9, shuffle=True, random_state=1)


data_train_generator = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                                  preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
                                                                  validation_split=0.2)


  

train_data = data_train_generator.flow_from_dataframe(
    dataframe=data_train,
    x_col='Path',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='training'
)

val_data = data_train_generator.flow_from_dataframe(
    dataframe=data_train,
    x_col='Path',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=True,
    seed=42,
    subset='validation'
)


data_test_generator = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.resnet50.preprocess_input)
test_data = data_test_generator.flow_from_dataframe(
    dataframe=data_test,
    x_col='Path',
    y_col='Label',
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

pretrained_model = tf.keras.applications.resnet50.ResNet50(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg')


pretrained_model.trainable = False

inputs = pretrained_model.input
x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
x = tf.keras.layers.Dense(50, activation='relu')(x)

   
outputs = tf.keras.layers.Dense(len(Labels),activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
   
   
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(train_data, validation_data=val_data, epochs=25, callbacks=[callbacks])


model.save(r"C:\Users\ankit\OneDrive\Desktop\capstone-final\Bone Fracture Identification\Bone Fracture Identification\models" + "/bodypart_frac.h5")
train_results=model.evaluate(train_data, verbose=0)
test_results = model.evaluate(test_data, verbose=0)
print("\nTRAINING")
print(f"Accuracy:{np.round(train_results[1]*100,2)}%")
print(f"Loss: {np.round(train_results[0]*100,2)}%")

   
print("\nTESTING")
print(f"Accuracy:{np.round(test_results[1]*100,2)}%")
print(f"Loss:{np.round(test_results[0]*100,2)}%")


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy of the model')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss of the model')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



# In[ ]:





# In[ ]:




