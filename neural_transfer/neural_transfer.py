import numpy as np

from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
from keras import backend as K


def preprocess_image(image_path, height=None, width=None):
    height = 400 if not height else height
    width = width if width else int(width * height / height)
    img = load_img(image_path, target_size=(height, width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def style_loss(style, combination, height, width):

    def build_gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram_matrix = K.dot(features, K.transpose(features))
    return gram_matrix

    S = build_gram_matrix(style)
    C = build_gram_matrix(combination)
    channels = 3
    size = height * width
    return K.sum(K.square(S - C))/(4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x):
    a = K.square(
    x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width
    - 1, :])
    b = K.square(
    x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height -
    1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# This is the path to the image you want to transform.
TARGET_IMG = 'lotr.jpg'
# This is the path to the style image.
REFERENCE_STYLE_IMG = 'pattern1.jpg'

width, height = load_img(TARGET_IMG).size
img_height = 480
img_width = int(width * img_height / height)

target_image = K.constant(preprocess_image(TARGET_IMG, height=img_height, width=img_width))
style_image = K.constant(preprocess_image(REFERENCE_STYLE_IMG, height=img_height, width=img_width))

# Placeholder for our generated image
generated_image = K.placeholder((1, img_height, img_width, 3))

# Combine the 3 images into a single batch
input_tensor = K.concatenate([target_image, style_image, generated_image], axis=0)

model = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)


# weights for the weighted average loss function
content_weight = 0.05
total_variation_weight = 1e-4

content_layer = 'block4_conv2'
style_layers = ['block1_conv2', 'block2_conv2',
'block3_conv3','block4_conv3', 'block5_conv3']
style_weights = [0.1, 0.15, 0.2, 0.25, 0.3]

# initialize total loss
loss = K.variable(0.)

# add content loss
layer_features = layers[content_layer]
target_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(target_image_features,
combination_features)

# add style loss
for layer_name, sw in zip(style_layers, style_weights):
layer_features = layers[layer_name]
style_reference_features = layer_features[1, :, :, :]
combination_features = layer_features[2, :, :, :]
sl = style_loss(style_reference_features, combination_features,
height=img_height, width=img_width)
loss += (sl*sw)

# add total variation loss
loss += total_variation_weight * total_variation_loss(generated_image)



class Evaluator(object):
    def __init__(self, height=None, width=None):
    self.loss_value = None
    self.grads_values = None
    self.height = height
    self.width = width

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, self.height, self.width, 3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
    
evaluator = Evaluator(height=img_height, width=img_width)

if __name__ == '__main__':
    print(1)