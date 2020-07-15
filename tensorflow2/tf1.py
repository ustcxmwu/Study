import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout

mnist = tf.keras.datasets.mnist


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.flatten = Flatten(input_shape=(28, 28))
        self.d1 = Dense(128, activation='relu')
        self.drop1 = Dropout(0.2)
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)

        x = self.drop1(x)
        return self.d2(x)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(28, 28)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10, activation='softmax')
    # ])
    model = MyModel()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    print("evaluate")
    model.evaluate(x_test, y_test)
