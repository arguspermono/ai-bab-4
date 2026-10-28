# EXERCISE 4.9
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications.vgg19 import decode_predictions
import numpy as np

model = VGG19(weights='imagenet')
print(model.summary())
