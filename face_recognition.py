import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pickle
import csv

from random import randint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras import backend as K
from keras import regularizers
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def train_model():
    path = './att_faces/orl_faces/'

    # 1-hot encoding
    a = np.array([i for i in range(43)])
    classes = np.zeros((a.size, a.max() + 1))
    classes[np.arange(a.size), a] = 1

    train_array = []
    test_array = []
    dir_array = []

    # let random index and 9th(enumerate begins at 0 so index 8 represents image 9) pgm for test
    for dir in os.listdir(path):
        i1, i2 = 8, randint(0, 7)
        for idx, img in enumerate(sorted(os.listdir(path + dir))):
            image = cv2.imread(path + dir + '/' + img, 0)
            image = cv2.resize(image, (32, 32))
            image = image[:, :, np.newaxis]
            if idx == i1 or idx == i2:
                test_array.append((image, classes[os.listdir(path).index(dir)]))
                continue

            train_array.append((image, classes[os.listdir(path).index(dir)]))
        dir_array.append(dir)

    input_shape = (32, 32, 1)

    model = Sequential()

    # convolutional layer 16 windows/filters of 3x3
    model.add(Conv2D(16, kernel_size=(3, 3),
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.),
                     activity_regularizer=regularizers.l2(0.),
                     input_shape=input_shape))

    # max of each 2x2 block
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # normalization
    model.add(BatchNormalization())

    # avoid overfitting
    model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.),
                     activity_regularizer=regularizers.l2(0.)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    # flatten for final layers
    model.add(Flatten())

    # fully-connected layer
    model.add(Dense(3000, activation='relu',
                    kernel_regularizer=regularizers.l2(0.),
                    activity_regularizer=regularizers.l2(0.)))

    model.add(Dropout(0.25))

    model.add(Dense(43, activation='softmax'))

    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    train_images, train_labels = np.array([t[0] for t in train_array]), np.array([t[1] for t in train_array])
    test_images, test_labels = np.array([t[0] for t in test_array]), np.array([t[1] for t in test_array])

    history = model.fit(train_images, train_labels,
                        batch_size=20,
                        epochs=10,
                        verbose=2,
                        validation_data=(test_images, test_labels))

    with open('face_recognition_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    return test_images, test_labels, classes, dir_array

def switch_demo(argument):
    switcher = {
        1: "sorin",
        2: "andreea",
        3: "alex",
        4: "javier",
        5: "jose",
        6: "jonal",
        7: "maria",
        8: "orly"
    }
    return switcher.get(argument, "Invalid month")

def evaluate_model(test_images, test_labels, classes, dir_array):
    with open('face_recognition_model.pkl', 'rb') as f:
        model = pickle.load(f)

    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    try:
        with open('csvfile.csv','w') as file:
            writer = csv.DictWriter(file, fieldnames = ["Expect", "Predict"])
            writer.writeheader()
            for i in range(20):
                #print(switch_demo(randint(1, 5)))
                for i in range(1,5):
                    imageindex = i
                    name = switch_demo(randint(1, 8))

                    pathperson = './att_faces/orl_faces/'+name+'/' + str(imageindex) + '.pgm'
                    person = cv2.imread(pathperson, cv2.IMREAD_UNCHANGED)
                    #cv2.imshow("imageof"+switch_demo(randint(1, 5))+" ", sorin1)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    person = cv2.resize(person, (32, 32))
                    person = person[:, :, np.newaxis]

                    people_test_image = np.array([person])
                    [prediction1] = model.predict(people_test_image)

                    a = max([(c, cosine_similarity([prediction1], [c])) for c in classes], key=lambda t:t[1])
                    print(name+" "+dir_array[list(a[0]).index(1)]+"\n")
                    writer.writerow({'Expect': name, 'Predict': dir_array[list(a[0]).index(1)]})
                    #file.write('\n')
    except Exception as e:
        print("Exception:",e)
'''
    index=1
    # evaluate test images added in the orl database
    imageindex = index

    pathsorin1 = './att_faces/orl_faces/sorin/' + str(imageindex) + '.pgm'
    sorin1 = cv2.imread(pathsorin1, cv2.IMREAD_UNCHANGED)
    cv2.imshow("imageofsorin", sorin1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sorin1 = cv2.resize(sorin1, (32, 32))
    sorin1 = sorin1[:, :, np.newaxis]

    imageindex = index

    pathandreea1 = './att_faces/orl_faces/andreea/' + str(imageindex) + '.pgm'
    andreea1 = cv2.imread(pathandreea1, cv2.IMREAD_UNCHANGED)
    cv2.imshow("imageofandreea", andreea1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    andreea1 = cv2.resize(andreea1, (32, 32))
    andreea1 = andreea1[:, :, np.newaxis]

    imageindex = index

    pathalex1 = './att_faces/orl_faces/alex/' + str(imageindex) + '.pgm'
    alex1 = cv2.imread(pathalex1, cv2.IMREAD_UNCHANGED)
    cv2.imshow("imageofalex", alex1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    alex1 = cv2.resize(alex1, (32, 32))
    alex1 = alex1[:, :, np.newaxis]

    imageindex = index

    pathjavier1 = './att_faces/orl_faces/javier/' + str(imageindex) + '.pgm'
    javier1 = cv2.imread(pathjavier1, cv2.IMREAD_UNCHANGED)
    cv2.imshow("imageofjavier", javier1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    javier1 = cv2.resize(javier1, (32, 32))
    javier1 = javier1[:, :, np.newaxis]

    imageindex = index

    pathjose1 = './att_faces/orl_faces/jose/' + str(imageindex) + '.pgm'
    jose1 = cv2.imread(pathjose1, cv2.IMREAD_UNCHANGED)
    cv2.imshow("imageofjose", jose1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    jose1 = cv2.resize(jose1, (32, 32))
    jose1 = jose1[:, :, np.newaxis]

    people_test_image = np.array([sorin1, andreea1, alex1,javier1,jose1])

    [prediction1, prediction2, prediction3,prediction4,prediction5] = model.predict(people_test_image)

    a = max([(c, cosine_similarity([prediction1], [c])) for c in classes], key=lambda t:t[1])
    b = max([(c, cosine_similarity([prediction2], [c])) for c in classes], key=lambda t:t[1])
    c = max([(c, cosine_similarity([prediction3], [c])) for c in classes], key=lambda t:t[1])
    d = max([(c, cosine_similarity([prediction4], [c])) for c in classes], key=lambda t:t[1])
    e = max([(c, cosine_similarity([prediction5], [c])) for c in classes], key=lambda t:t[1])

    print(dir_array[list(a[0]).index(1)])
    print(dir_array[list(b[0]).index(1)])
    print(dir_array[list(c[0]).index(1)])
    print(dir_array[list(d[0]).index(1)])
    print(dir_array[list(e[0]).index(1)])    
'''    

ti, tl, c, d = train_model()
evaluate_model(ti, tl, c, d)
