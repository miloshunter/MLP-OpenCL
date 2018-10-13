import tensorflow as tf
from PIL import Image
import numpy
from skimage import color
from skimage import io

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

bla = model.get_weights()

im = io.imread('sedmica.png', as_gray=True)

slika = 255 - (im * 255)

ulaz = numpy.array(slika)
y = ulaz.reshape(1, 28, 28)

predikcija = model.predict_on_batch(y)

print("Rezultat: ")
print(predikcija)

print("Tezine:")
print(model.get_weights())

pass