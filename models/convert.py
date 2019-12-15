from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, ReLU, Conv2D, Conv2DTranspose, Concatenate, \
                                    MaxPooling2D, UpSampling2D, BatchNormalization \

def LikeUnet():
    # input
    im = Input(shape=(None, None, 1))
    pre = Conv2D(8, 3, padding='same', activation='relu')(im)
    # conv1
    x = Conv2D(16, 3, padding='same', name='conv1')(pre)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    conv1 = MaxPooling2D(2)(x)
    # conv2
    x = Conv2D(32, 3, padding='same', name='conv2')(conv1)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    conv2 = MaxPooling2D(2)(x)
    # conv3
    x = Conv2D(32, 3, padding='same', name='conv3')(conv2)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    conv3 = MaxPooling2D(2)(x)
    # conv4
    x = Conv2D(32, 3, padding='same', name='conv4')(conv3)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    conv4 = MaxPooling2D(2)(x)
    # conv5
    x = Conv2D(64, 3, padding='same', name='conv5')(conv4)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    conv5 = MaxPooling2D(2)(x)
    # transconv1
    x = Conv2D(64, 3, padding='same', name='conv6')(conv5)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(2, interpolation='bilinear')(x)
    # transconv2
    x = Concatenate()([conv4, x])
    x = Conv2D(32, 3, padding='same', name='conv7')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(32, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(2, interpolation='bilinear')(x)
    # transconv3
    x = Concatenate()([conv3, x])
    x = Conv2D(32, 3, padding='same', name='conv8')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(32, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(2, interpolation='bilinear')(x)
    # transconv4
    x = Concatenate()([conv2, x])
    x = Conv2D(32, 3, padding='same', name='conv9')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(32, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(2, interpolation='bilinear')(x)
    # transconv5
    x = Concatenate()([conv1, x])
    x = Conv2D(16, 3, padding='same', name='conv10')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2DTranspose(16, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = UpSampling2D(2, interpolation='bilinear')(x)
    # output
    x = Conv2D(8, 3, padding='same', activation='relu')(x)
    out = Conv2D(2, 1, padding='same', activation='softmax')(x)
    return Model(inputs=im, outputs=out)

with open('dark-light-pre.json', 'r') as f:
    json_string = f.read()
pre_model = model_from_json(json_string)
pre_model.load_weights('dark-light-pre-50-0.9735.h5')

# ACNet:Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric ConvolutionBlocks
# https:arxiv.org/abs/1908.03930

# convert weights
model = LikeUnet()
model.load_weights('dark-light-pre-50-0.9735.h5', by_name=True)
for i in range(1, 11):
    conv = pre_model.get_layer(f'conv{i}-1').get_weights()
    conv_ = pre_model.get_layer(f'conv{i}-2').get_weights()
    conv[0][1,:,:,:] += conv_[0][0]
    conv[1] += conv_[1]
    conv_ = pre_model.get_layer(f'conv{i}-3').get_weights()
    conv[0][:,1,:,:] += conv_[0][:,0]
    conv[1] += conv_[1]
    model.get_layer(f'conv{i}').set_weights(conv)
    print(f'load layer conv{i}')

with open('dark-light.json', 'w') as f:
    f.write(model.to_json())
model.save_weights('dark-light-50-0.9735.h5')