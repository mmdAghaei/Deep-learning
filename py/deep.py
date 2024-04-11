# Import Package
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Variable value
x = np.arange(-80,81,.5)
y = x ** 2 + x * 5

# Split Test and Train
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2)

# 1) Model
model_1 = keras.Sequential()
model_1.add(keras.layers.Dense(units=1024,activation="relu",input_shape=[1]))
model_1.add(keras.layers.Dense(units=1))

# Compile
model_1.compile(optimizer="adam",loss="mean_squared_error")

# Train Model
hist_1 = model_1.fit(x_train,y_train,batch_size=64,epochs=1000,validation_data=(x_test,y_test))

# 2) Model
model_2 = keras.Sequential()
model_2.add(keras.layers.Dense(units=256,activation="relu",input_shape=[1]))
model_2.add(keras.layers.Dense(units=256,activation="relu"))
model_2.add(keras.layers.Dense(units=256,activation="relu"))
model_2.add(keras.layers.Dense(units=256,activation="relu"))
model_2.add(keras.layers.Dense(units=1))

# Compile
model_2.compile(optimizer="adam",loss="mean_squared_error")

# Train
hist_2 = model_2.fit(x_train,y_train,batch_size=64,epochs=1000,validation_data=(x_test,y_test))

# Predict Model
y_pred_model_1 = model_1.predict(x_test)
y_pred_model_2 = model_2.predict(x_test)

# Plot
plt.scatter(x_test,y_test,color="red")
plt.scatter(x_test,y_pred_model_1,color="green")
plt.scatter(x_test,y_pred_model_2,color="blue")
plt.show()

# History
plt.plot(hist_2.history['val_loss'][800:])
plt.show()
