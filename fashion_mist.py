import tensorflow as tf

"""download dataset"""

(x_train,y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_train.ndim)

x_train = x_train.reshape(x_train.shape[0],28,28,1)
print(x_train.ndim)

x_test = x_test.reshape(x_test.shape[0],28,28,1)
print(x_test.ndim)

X_train = x_train.astype('float32')
print(X_train)
X_test = x_test.astype('float32')
print(X_test)

# normalize 
x_train =(x_train/255)
x_test =(x_test/255)
print(x_train)
print(x_test)

# packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# packages installation

model1 = Sequential()
model1.add(Conv2D(28,kernel_size = (3,3), input_shape = (28,28,1)))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Flatten())
model1.add(Dense(168,activation = tf.nn.relu)) 
model1.add(Dropout(0.2))
model1.add(Dense(10, activation = tf.nn.softmax))

#compiling & fitting model
model1.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy',metrics= ['accuracy'])
model1.fit(x = x_train, y = y_train, epochs=20)
model1.evaluate(x_test, y_test)