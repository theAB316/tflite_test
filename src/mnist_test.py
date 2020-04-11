from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("mnist.h5")

test_image = cv2.imread("sample_image.png", cv2.IMREAD_GRAYSCALE)
test_image = cv2.resize(test_image, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
test_image = test_image.astype('float32')
test_image /= 255

test_image = np.asarray([test_image])
test_image = test_image.reshape(test_image.shape[0], 28, 28, 1)

print(test_image.shape)

pred = model.predict(test_image)
print(pred)
print(np.argmax(pred))