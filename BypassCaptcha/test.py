from main import *
from tensorflow.keras.models import load_model
dict_temp = {3: '2', 0: '?', 1: '0', 6: '5', 8: '7', 10: '9', 2: '1', 4: '3', 5: '4', 9: '8', 7: '6'}

model = load_model('w/save_md.h5')

image = cv2.imread('data_test/66.jpg')
predict(image, dict_temp, model)
cv2.imshow('License Plate', image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)


