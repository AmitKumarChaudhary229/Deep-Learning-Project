import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Fetch MNIST dataset from tensorflow
(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = tf.keras.datasets.mnist.load_data()

# Print full information about dataset shape and size
print(f"Training data shape: {mnist_x_train.shape}, Training labels shape: {mnist_y_train.shape}")
print(f"Test data shape: {mnist_x_test.shape}, Test labels shape: {mnist_y_test.shape}")
print(f"Data type: {mnist_x_train.dtype}, Label type: {mnist_y_train.dtype}")
mnist_x_train = mnist_x_train.astype('float32') / 255.0
mnist_x_test = mnist_x_test.astype('float32') / 255.0

# Add channel dimension for CNN (28,28) -> (28,28,1)
mnist_x_train = np.expand_dims(mnist_x_train, -1)
mnist_x_test = np.expand_dims(mnist_x_test, -1)

# Split train into 70:30 train and validation
x_train, x_val, y_train, y_val = train_test_split(mnist_x_train, mnist_y_train, test_size=0.3, random_state=42)

print(f"Train set shape: {x_train.shape}, Labels: {y_train.shape}")
print(f"Validation set shape: {x_val.shape}, Labels: {y_val.shape}")
print(f"Test set shape: {mnist_x_test.shape}, Labels: {mnist_y_test.shape}")
def create_cnn_l2_model():
    weight_decay = 1e-4
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay), input_shape=(28, 28, 1)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model
# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=10,      # rotate by ±10 degrees
    zoom_range=0.1,         # zoom in/out by 10%
    width_shift_range=0.1,  # shift horizontally by 10%
    height_shift_range=0.1  # shift vertically by 10%
)

def create_cnn_aug_model():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                            input_shape=(28, 28, 1)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model
def create_cnn_dropout_model():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                            input_shape=(28, 28, 1)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout with 50% rate
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model
# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=10,      # rotate by ±10 degrees
    zoom_range=0.1,         # zoom in/out by 10%
    width_shift_range=0.1,  # shift horizontally by 10%
    height_shift_range=0.1  # shift vertically by 10%
)

def create_cnn_combined_model():
    weight_decay = 1e-4
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay), input_shape=(28, 28, 1)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(layers.Dropout(0.5))  # Dropout with 50% rate
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model
def collect_metrics_dict_only(model, x_train, y_train, x_val, y_val, x_test, y_test):
    results_dict = {}
    
    def get_metrics(model, x, y):
        preds = model.predict(x, verbose=0)
        pred_labels = preds.argmax(axis=1)
        misclassified = (pred_labels != y).sum()
        accuracy = (pred_labels == y).mean()
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = loss_fn(y, preds).numpy()
        total = y.shape[0]
        return accuracy, loss, misclassified, total

    for dataset_name, x, y in [('Train', x_train, y_train),
                               ('Validation', x_val, y_val),
                               ('Test', x_test, y_test)]:
        accuracy, loss, misclassified, total = get_metrics(model, x, y)
        results_dict[dataset_name] = {
            'accuracy': accuracy,
            'loss': loss,
            'misclassified': misclassified,
            'total': total
        }
    return results_dict
print("--- Training CNN model with L2 regularization ---")
cnn_l2_model = create_cnn_l2_model()
cnn_l2_model.summary()


history_l2 = cnn_l2_model.fit(
    x_train, y_train, epochs=5, batch_size=64, validation_data=(x_val, y_val), verbose=2
)

print("--- Training CNN model with Data Augmentation ---")
cnn_aug_model = create_cnn_aug_model()
cnn_aug_model.summary()


# Train with data augmentation
train_generator = datagen.flow(x_train, y_train, batch_size=64)
steps_per_epoch = len(x_train) // 64


history_aug = cnn_aug_model.fit(
    train_generator, steps_per_epoch=steps_per_epoch, epochs=5, 
    validation_data=(x_val, y_val), verbose=2
)

print("--- Training CNN model with Dropout ---")
cnn_dropout_model = create_cnn_dropout_model()
cnn_dropout_model.summary()


history_dropout = cnn_dropout_model.fit(
    x_train, y_train, epochs=5, batch_size=64, validation_data=(x_val, y_val), verbose=2
)

print("--- Training CNN model with Combined Regularization (L2 + Data Augmentation + Dropout) ---")
cnn_combined_model = create_cnn_combined_model()
cnn_combined_model.summary()


# Train with data augmentation
train_generator = datagen.flow(x_train, y_train, batch_size=64)
steps_per_epoch = len(x_train) // 64


history_combined = cnn_combined_model.fit(
    train_generator, steps_per_epoch=steps_per_epoch, epochs=5, 
    validation_data=(x_val, y_val), verbose=2
)
metrics_data = {
    'L2 Regularization': collect_metrics_dict_only(cnn_l2_model, x_train, y_train, x_val, y_val, mnist_x_test, mnist_y_test),
    'Data Augmentation': collect_metrics_dict_only(cnn_aug_model, x_train, y_train, x_val, y_val, mnist_x_test, mnist_y_test),
    'Dropout': collect_metrics_dict_only(cnn_dropout_model, x_train, y_train, x_val, y_val, mnist_x_test, mnist_y_test),
    'Combined (L2+Aug+Dropout)': collect_metrics_dict_only(cnn_combined_model, x_train, y_train, x_val, y_val, mnist_x_test, mnist_y_test)
}


# import pandas as pd
# %pip install openpyxl

# rows = []
# for technique, dataset_metrics in metrics_data.items():
#     for dataset, metrics in dataset_metrics.items():
#         rows.append({
#             'Technique': technique,
#             'Dataset': dataset,
#             'Accuracy': metrics['accuracy'],
#             'Loss': metrics['loss'],
#             'Misclassified': metrics['misclassified'],
#             'Total Samples': metrics['total']
#         })

# df = pd.DataFrame(rows)

# # Save to CSV and Excel files
# df.to_excel('mnist_regularization_metrics.xlsx', index=False)

# print("Metrics saved as 'mnist_regularization_metrics.csv' and 'mnist_regularization_metrics.xlsx'.")
