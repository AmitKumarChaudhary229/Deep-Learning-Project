import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load and preprocess MNIST dataset
(digits_train, labels_train), (digits_test, labels_test) = tf.keras.datasets.mnist.load_data()

# Display dataset information
print(f"Training images shape: {digits_train.shape}, Labels shape: {labels_train.shape}")
print(f"Test images shape: {digits_test.shape}, Labels shape: {labels_test.shape}")
print(f"Image data type: {digits_train.dtype}, Label data type: {labels_train.dtype}")

# Normalize and reshape data
digits_train = digits_train.astype('float32') / 255.0
digits_test = digits_test.astype('float32') / 255.0
digits_train = np.expand_dims(digits_train, axis=-1)
digits_test = np.expand_dims(digits_test, axis=-1)

# Split training data into 75:25 train/validation
x_train_data, x_val_data, y_train_data, y_val_data = train_test_split(
    digits_train, labels_train, test_size=0.25, random_state=123
)

print(f"Training set: {x_train_data.shape}, Labels: {y_train_data.shape}")
print(f"Validation set: {x_val_data.shape}, Labels: {y_val_data.shape}")
print(f"Test set: {digits_test.shape}, Labels: {labels_test.shape}")

# Define CNN with L2 regularization
def build_l2_cnn():
    l2_factor = 0.0001
    model = models.Sequential([
        layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_factor), input_shape=(28, 28, 1)),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_factor)),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_factor)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

# Define data augmentation
aug_generator = ImageDataGenerator(
    rotation_range=15,
    shear_range=0.1,
    width_shift_range=0.15,
    height_shift_range=0.15
)

# Define CNN for augmentation
def build_aug_cnn():
    model = models.Sequential([
        layers.Conv2D(16, (5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

# Define CNN with dropout
def build_dropout_cnn():
    model = models.Sequential([
        layers.Conv2D(16, (5, 5), strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu'),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

# Define CNN with combined regularization
def build_combined_cnn():
    l2_factor = 0.0001
    model = models.Sequential([
        layers.Conv2D(16, (5, 5), strides=(1, 1), padding='valid', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_factor), input_shape=(28, 28, 1)),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Conv2D(32, (3, 3), strides=(1, 1), padding='valid', activation='relu',
                      kernel_regularizer=regularizers.l2(l2_factor)),
        layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_factor)),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

# Function to compute and print metrics
def display_metrics(model, x_tr, y_tr, x_vl, y_vl, x_ts, y_ts, model_name):
    def calc_metrics(x, y):
        preds = model.predict(x, verbose=0)
        pred_labels = np.argmax(preds, axis=1)
        acc = np.mean(pred_labels == y)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        loss = loss_fn(y, preds).numpy()
        misclass = np.sum(pred_labels != y)
        total = y.shape[0]
        return acc, loss, misclass, total

    print(f"\nResults for {model_name}:")
    print(f"{'Dataset':<12} {'Accuracy':<10} {'Loss':<10} {'Misclassified':<15} {'Total':<10}")
    print("-" * 50)
    for name, x, y in [('Train', x_tr, y_tr), ('Validation', x_vl, y_vl), ('Test', x_ts, y_ts)]:
        acc, loss, misclass, total = calc_metrics(x, y)
        print(f"{name:<12} {acc:.4f}     {loss:.6f}  {misclass:<15} {total:<10}")

# Train and evaluate L2 model
print("\n--- Training CNN with L2 Regularization ---")
l2_model = build_l2_cnn()
l2_model.summary()
history_l2 = l2_model.fit(x_train_data, y_train_data, epochs=3, batch_size=32,
                          validation_data=(x_val_data, y_val_data), verbose=1)
display_metrics(l2_model, x_train_data, y_train_data, x_val_data, y_val_data, digits_test, labels_test, "L2 Model")

# Train and evaluate augmentation model
print("\n--- Training CNN with Data Augmentation ---")
aug_model = build_aug_cnn()
aug_model.summary()
train_gen = aug_generator.flow(x_train_data, y_train_data, batch_size=32)
steps = len(x_train_data) // 32
history_aug = aug_model.fit(train_gen, steps_per_epoch=steps, epochs=3,
                            validation_data=(x_val_data, y_val_data), verbose=1)
display_metrics(aug_model, x_train_data, y_train_data, x_val_data, y_val_data, digits_test, labels_test, "Augmentation Model")

# Train and evaluate dropout model
print("\n--- Training CNN with Dropout ---")
dropout_model = build_dropout_cnn()
dropout_model.summary()
history_dropout = dropout_model.fit(x_train_data, y_train_data, epochs=3, batch_size=32,
                                   validation_data=(x_val_data, y_val_data), verbose=1)
display_metrics(dropout_model, x_train_data, y_train_data, x_val_data, y_val_data, digits_test, labels_test, "Dropout Model")

# Train and evaluate combined model
print("\n--- Training CNN with Combined Regularization ---")
combined_model = build_combined_cnn()
combined_model.summary()
train_gen = aug_generator.flow(x_train_data, y_train_data, batch_size=32)
steps = len(x_train_data) // 32
history_combined = combined_model.fit(train_gen, steps_per_epoch=steps, epochs=3,
                                     validation_data=(x_val_data, y_val_data), verbose=1)
display_metrics(combined_model, x_train_data, y_train_data, x_val_data, y_val_data, digits_test, labels_test, "Combined Model")

# Commented-out code for saving metrics
# import pandas as pd
# %pip install openpyxl
# results = []
# for model_name, model in [('L2', l2_model), ('Augmentation', aug_model),
#                          ('Dropout', dropout_model), ('Combined', combined_model)]:
#     for ds_name, x, y in [('Train', x_train_data, y_train_data),
#                           ('Validation', x_val_data, y_val_data),
#                           ('Test', digits_test, labels_test)]:
#         preds = model.predict(x, verbose=0)
#         pred_labels = np.argmax(preds, axis=1)
#         acc = np.mean(pred_labels == y)
#         loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
#         loss = loss_fn(y, preds).numpy()
#         misclass = np.sum(pred_labels != y)
#         results.append({
#             'Model': model_name,
#             'Dataset': ds_name,
#             'Accuracy': acc,
#             'Loss': loss,
#             'Misclassified': misclass,
#             'Total': y.shape[0]
#         })
# df = pd.DataFrame(results)
# df.to_excel('mnist_model_comparison.xlsx', index=False)
# print("Metrics saved to 'mnist_model_comparison.xlsx'.")