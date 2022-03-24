import os
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from sklearn.model_selection import train_test_split

def createPickle(imageFolderPath, dirs, cls, size):
  imageData = []
  for dir in dirs:
      files = os.listdir(imageFolderPath + dir)
      for file in files:
          print(file)
          image = cv2.imread(imageFolderPath + dir + "/" + file)
                              try:
                                      image = cv2.resize(image, size)
                                              image = image.flatten()
                                                      image = np.append(image, cls[dir])
                                                              imageData.append(image)
                                                                    except Exception as e:
                                                                            print(str(e))
  imageData = np.asarray(imageData)
    pickleFile = str(size[0]) +'_' + str(size[1]) + '_color.pickle'
      fp = open(imageFolderPath + pickleFile, "wb")
        pickle.dump(imageData, fp)
          fp.close()
def create_encoder(imageFolderPath, pickleFile):
  fp = open(imageFolderPath + pickleFile, "rb")
    imageData = pickle.load(fp)
      fp.close()
        print(imageData.shape)
  X = imageData[:, 0:-1]
    Y = imageData[:, -1]
  X = np.reshape(X, (X.shape[0], size[0], size[1], 3))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
      print(X_train.shape)
  X_train = X_train / 255.0
    X_test = X_test / 255.0
  encoder = Sequential()
    encoder.add(Flatten(input_shape=[size[0], size[1], 3]))
      encoder.add(Dense(6000, activation="relu"))
        encoder.add(Dense(3000, activation="relu"))
          encoder.add(Dense(1000, activation="relu"))
  ### Decoder
    decoder = Sequential()
      decoder.add(Dense(1000, input_shape=[1000], activation='relu'))
        decoder.add(Dense(3000, activation='relu'))
          decoder.add(Dense(6000, activation='relu'))
            decoder.add(Dense(size[0] * size[1] * 3, activation="relu"))
              decoder.add(Reshape([size[0], size[1], 3]))
  ### Autoencoder
    autoencoder = Sequential([encoder, decoder])
      autoencoder.compile(loss="mse")
        autoencoder.fit(X_train, X_train, batch_size=100, epochs=3, verbose=1)
  #X_train_trans = encoder.predict(X_train)
    #X_test_trans = encoder.predict(X_test)
if __name__ == "__main__":
  imageFolderPath = "D:/Research/Tejas Sir/CornDataset/"
    dirs = ['Blight', 'Healthy', 'Gray_Leaf_Spot', 'Common_Rust']
      cls = {'Blight': 0, 'Healthy': 1, 'Gray_Leaf_Spot': 2, 'Common_Rust': 3}
        size = (64, 64)
          #createPickle(imageFolderPath, dirs, cls, size)
  pickleFile = str(size[0]) + '_' + str(size[1]) + '_color.pickle'
    create_encoder(imageFolderPath, pickleFile)
