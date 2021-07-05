import keras, cv2, os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="dataset", target_size=(224, 224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test", target_size=(224, 224))
numOfRetrain = 5
if numOfRetrain != 0:
    reconstructed_model = keras.models.load_model("my_model")
    for i in range(0, numOfRetrain):
        try:
            print(i)
            reconstructed_model.fit(traindata, validation_data=testdata, epochs=1, verbose=2)
            reconstructed_model.save("my_model")
        except:
            print("Got error.")

