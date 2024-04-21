import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
data = tfds.load('mnist')

train_data, test_data = data["train"], data["test"]

train_data = train_data.map(lambda x: (x['image'], x['label']))
test_data = test_data.map(lambda x: (x['image'], x['label']))

train_dataset = tfds.as_numpy(train_data)
test_dataset = tfds.as_numpy(test_data)

x_train, y_train = [], []
x_test, y_test = [], []

for example in train_dataset:
    x_train.append(example[0] / 255.0)  # Access the first element of the tuple directly
    y_train.append(example[1])           # Access the second element of the tuple directly

for example in test_dataset:
    x_test.append(example[0] / 255.0)   # Access the first element of the tuple directly
    y_test.append(example[1])            # Access the second element of the tuple directly

# Convert lists to NumPy arrays
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# predictions = model(x_train[:1]).numpy()
# predictions

# tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.save("mnist_model")