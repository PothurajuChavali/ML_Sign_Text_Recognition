from sklearn import svm
from scipy import ndimage as ndi
import numpy as np
import cv2
import pandas as pd
import pickle


def obtain_featurevector(filename, img_row, img_col):
    gray_image = cv2.imread(filename, 0)
    gray_image = cv2.resize(gray_image, (256, 256))
    # cv2.imshow('Gray_image', gray_image)
    # cv2.waitKey(0)
    ret, thresh_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('Binary_image', thresh_image)
    # cv2.waitKey(0)
    number_of_white_pix = np.sum(thresh_image == 255)
    number_of_black_pix = np.sum(thresh_image == 0)
    if number_of_white_pix > number_of_black_pix:
        thresh_image = cv2.bitwise_not(thresh_image)
        # cv2.imshow('Inverted_image', thresh_image)
        # cv2.waitKey(0)
    kernel_open = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN, kernel_open)
    # cv2.imshow('Open_image', opening)
    # cv2.waitKey(0)
    kernel_erosion = np.ones((3, 3), np.uint8)
    img_erosion = cv2.erode(opening, kernel_erosion)
    cv2.imshow('Eroded_image', img_erosion)
    cv2.waitKey(1)
    label_objects, nb_labels = ndi.label(img_erosion)
    feature_vector = np.zeros((1, img_row * img_col), dtype='float64')
    for i in (range(1, nb_labels)):
        num_pixels = np.sum(label_objects == i)
        if num_pixels > 500 and num_pixels < 2500:
            tmp = label_objects == i
            r, = np.where(tmp.sum(axis=1) > 1)
            c, = np.where(tmp.sum(axis=0) > 1)
            # crop with border
            tmp_img = gray_image[(r.min() - 1):(r.max() + 2), (c.min() - 1):(c.max() + 2)]
            digit_img = cv2.resize(tmp_img, (img_row, img_col))
            cv2.imshow('character', digit_img)
            cv2.waitKey(1)
            digit_img = digit_img.reshape((1, (img_row * img_col)))
            feature_vector = np.vstack((feature_vector, digit_img))
    # remove first row
    feature_vector = np.delete(feature_vector, 0, 0)

    return feature_vector


# ******************************************        Main Program Start        *************************************** #
if __name__ == '__main__':
    train = False
    test = True

    if train is True:
        train_data = pd.read_csv('C:/Users/uidq6830/Downloads/archive/emnist-balanced-train.csv')
        # extract the features and labels from train data set
        features = np.array(train_data.iloc[:, 1:].values)
        labels = np.array(train_data.iloc[:, 0].values)
        print(features.shape)
        print(labels.shape)
        # cv2.imshow(x1[0].reshape(28, 28))
        # cv2.waitkey(0)

        # train with svm
        clf_svm = svm.SVC()
        clf_svm.fit(features, labels)
        print("Training set score: %f" % clf_svm.score(features, labels))
        # save the model to disk
        filename = 'signtext_recognition_model_balanced_dataset.pkl'
        pickle.dump(clf_svm, open(filename, 'wb'))

    if test is True:
        saved_model = pickle.load(open('signtext_recognition_model_balanced_dataset.pkl', 'rb'))
        mapping = np.loadtxt('C:/Users/uidq6830/Downloads/archive/emnist-balanced-mapping.txt')
        img_row = 28
        img_col = 28
        image_path = 'C:/Users/uidq6830/Downloads/Sign_detection_TestSet/TestSet/15.png'
        features = obtain_featurevector(image_path, img_row, img_col)
        print(features.shape)
        if features.shape[0] < 1:
            features = features.reshape(1, -1)

        predictions = saved_model.predict(features)
        print('Detected text from given sign image:')
        for i in range(len(predictions)):
            print('{}'.format(chr(int(mapping[predictions[i]][1]))))

