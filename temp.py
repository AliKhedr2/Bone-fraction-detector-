from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript()
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename

  #--------------- ------------------- --------------------#


import tensorflow
import numpy as np
import cv2


import cv2 
from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image(filename))
  print(filename)
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))

np.set_printoptions(suppress=True)


model = tensorflow.keras.models.load_model('AA.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)


cam = cv2.VideoCapture(1)
text = ""

img = cv2.imread(filename)
img = cv2.resize(img,(224, 224))
image_array = np.asarray(img)
print(image_array)

normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

data[0] = normalized_image_array

prediction = model.predict(data)

for i in prediction:
    if i[0] > 0.7:
        text ="Broken"
    if i[1] > 0.7:
        text ="Normal"
    if i[2] > 0.7:
        text = "Not X-ray"
    
    print(text)
