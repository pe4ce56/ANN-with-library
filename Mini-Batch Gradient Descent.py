import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate synthetic regression data
np.random.seed(0)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2.5 * X + np.random.randn(50, 1) * 2


# Convert data to TensorFlow tensors
X_train = tf.convert_to_tensor(X, dtype=tf.float32)
y_train = tf.convert_to_tensor(y, dtype=tf.float32)

# make neural network model with sigmoid activation
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, activation='sigmoid', input_shape=(1,)),  # hiden layer with 3 nueron and activation function sigmoid
    tf.keras.layers.Dense(1, activation=None)  # output layer with  1 neuron and Linear Activation
])

# compile the model with MAE and Mini-Batch Gradient Descent (SGD) 
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='mean_absolute_error')

# tran model, 100 epoch, 30 batch size
history = model.fit(X_train, y_train, epochs=100, batch_size=30, verbose=1)

# predictions of y
y_pred = model.predict(X)

# Plot actual vs predicted values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, label="Actual Data", color="blue")
plt.plot(X, y_pred, label="Predicted Data", color="red")
plt.xlabel("Input X")
plt.ylabel("Output y")
plt.title("Regression Prediction using TensorFlow Neural Network")
plt.legend()

# Plot loss history
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], color="green")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.grid()

plt.show()
 