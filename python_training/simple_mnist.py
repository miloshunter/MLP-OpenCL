import tensorflow as tf
from PIL import Image
import numpy
from skimage import color
from skimage import io


def save_2d_array_c(array, name):
    file = open(name+".h", "w")
    file.writelines("//Testiranje ispisa\n")
    file.writelines("//Niz je: " + str(array.shape) + "\n")
    file.writelines("double "+name+"["+
                    str(array.shape[1])+"]["+str(array.shape[0])+
                    "];\n")
    file.close()

    file = open(name+".c", "w")
    file.writelines("double "+name+"[" +
                    str(array.shape[1]) + "][" + str(array.shape[0]) +
                    "] = {\n")

    for i in range(array.shape[1]):
        file.writelines("{ ")
        tmp = array[:, i]
        for j in range(array.shape[0]):
            file.writelines(str(tmp[j]))
            if j < array.shape[0] - 1:
                file.writelines(",")
        file.writelines("}")
        if i < array.shape[1]-1:
            file.writelines(",")
        file.writelines("\n")
    file.writelines("};\n")
    file.close()


def save_1d_array_c(array, name):
    file = open(name+".h", "w")
    file.writelines("//Testiranje ispisa\n")
    file.writelines("//Niz je: " + str(array.shape) + "\n")
    file.writelines("extern double* "+name+"["+
                    str(array.shape[0])+
                    "];\n")
    file.close()
    file = open(name+".c", "w")
    file.writelines("const double* " + name + "[" +
                    str(array.shape[0]) +
                    "] = {\n")
    for j in range(array.shape[0]):
        file.writelines(str(array[j]))
        if j < array.shape[0] - 1:
            file.writelines(",")
    file.writelines("};\n")
    file.close()


im = io.imread('../test_pics/dvojka.png', as_gray=True)
save_2d_array_c(im, "dvojka")


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

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)

weights = model.get_weights()


save_2d_array_c(weights[0], "L1_w")
save_2d_array_c(weights[2], "L2_w")
save_1d_array_c(weights[1], "L1_b")
save_1d_array_c(weights[3], "L2_b")




slika = 255 - (im * 255)

ulaz = numpy.array(slika)
y = ulaz.reshape(1, 28, 28)

predikcija = model.predict_on_batch(y)

print("Rezultat: ")
print(predikcija)

#print("Tezine:")
#print(model.get_weights())

pass

