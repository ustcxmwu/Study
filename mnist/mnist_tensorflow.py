import tensorflow as tf
from tensorflow.keras import layers


if __name__ == '__main__':
    # Load mnist data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize data，将图像的像素值都处理到[0,1]范围
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    # Add model
    model = tf.keras.Sequential()
    # Add three convoluational layers and pool layers，使用3层卷积层和对应的最大池化层，卷积核为3，池化大小为2
    # convoluational layer1
    model.add(layers.Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(3, 3), kernel_initializer='he_normal',
                            strides=1, padding='same', activation='relu', name='conv1'))
    # maxpooling layer1
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same', name='pool1'))
    # convoluational layer2
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='he_normal', strides=1,
                            padding='same', activation='relu', name='conv2'))
    # maxpooling layer2
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same', name='pool2'))
    # convoluational layer3
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer='he_normal', strides=1,
                            padding='same', activation='relu', name='conv3'))

    # maxpooling layer3
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same', name='pool3'))
    # Add flatten ,dropout ,and FC layer，卷积池化后加一个Flatten层，然后加一个dropout，最后加一个softmax进行分类；
    model.add(layers.Flatten(name='flatten'))
    model.add(layers.Dropout(rate=0.5, name='dropout'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(), metrics=['accuracy'])

    print(model.summary())

    model.fit(x_train, y_train, epochs=5, batch_size=64)

    model.evaluate(x_test, y_test)
