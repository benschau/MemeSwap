# -*- coding: utf-8 -*-
"""
Module to connect reddit web scraping to the google cloud api and create art form it
"""
import meme, vision_detector, faceSwap2
import cv2
from google.cloud import vision
from google.cloud.vision import types
import random
import numpy as np

class Pipeline:
    def __init__(self):
        # probably a good idea to use wholesome memes instead of dankmemes for presentation
        self.subreddit = 'wholesomememes'
        self.vision_detector = vision_detector.VisionDetector()
        
    def get_n_memes(self, n):
        """
        Method to grab images from a given subreddit.
        :param subreddit: The subreddit to grab images from
        :param n: The number of images to return
        :return:
        """
        reddit = meme.get_secrets('cert.txt')
   
        gen = meme.MemeGenerator(reddit, self.subreddit, limit=n)
        memes = gen.get_memes(num=n)

        img_paths = []
        for m in memes:
            local_image_url = meme.download_img(m.url)
            if local_image_url != None:
                img_paths.append(local_image_url)
        # print("get_n_memes output len %d" %len(img_paths))
        return img_paths
            
    def study_memes(self, img_paths):
        """
        Method to use google cloud api to analyze faces in images.
        :param img_paths: list of paths to images to study
        :return: a list of lists of dictionaries describing the faces
                  See: vision_detector.clean_face_features() for one entry in that list
        """
        clean_faces = []
        for local_path in img_paths:
            face = self.vision_detector.read_image(local_path)
            # print("TRUE FACE VALUE:\n%s\n\n" % face)
            # print("Type:\t%s" %type(face))
            try:  # hope to throw an error if object is too complicated
                isValid = type(None) != type(face) and face != None
            except TypeError :
                isValid = True

            if isValid:
                cleaned_face = self.vision_detector.clean_face_features(face)

                if not (cleaned_face is None):
                    # add cleaned face dictionary to list
                    clean_faces.append(cleaned_face)
            
        return clean_faces
    
    def create_meme(self, image1, image2, features1, features2, location):
        """
        Method to perform face swap on two individual images. The resulting image will superimpose image2's
        face over image1's face.
        :param image1: The base image whose faces will be covered as np.array
        :param image2: The image whose faces will cover another face as np.array
        :param features1: the feature dictionaries for image one
        :param features2: the feature dictionaries for image2
        :param location: The location to write the resulting work of art to
        :return: One face-swapped art-transcending work of genius
        """
        # turn image filepaths into np.arrays
        # print("Test of feature1 values:\n%s\nLen: %d" % (str(features1), len(features1)))
        # print("Test of feature2 values:\n%s\nLen: %d" % (str(features2), len(features2)))
        image1 = cv2.imread(image1, cv2.IMREAD_COLOR)
        image1 = cv2.resize(image1, (image1.shape[1] * 1,
                                 image1.shape[0] * 1))
        image2 = cv2.imread(image2, cv2.IMREAD_COLOR)
        image2 = cv2.resize(image2, (image2.shape[1] * 1,
                                     image2.shape[0] * 1))
        random.seed(69)  # for debugging and the memes
        count = 1
        for feature2 in features2:
            print("swapping face #%d" %count)
            feature1 = random.choice(features1)
            
            # make subimage1
            if feature1['outer_bound_dict']:  # handle no bound box edge case
                xT1, yL1 = feature1['outer_bound_dict']['UPPER_LEFT']
                xB1, yR1 = feature1['outer_bound_dict']['LOWER_RIGHT']
            elif feature1[1]:
                xT1, yL1 = feature1['inner_bound_dict']['UPPER_LEFT']
                xB1, yR1 = feature1['inner_bound_dict']['LOWER_RIGHT']
            else:
                print("Something went wrong, spicy boi")
            width1 = yR1 - yL1  # col values
            height1 = xB1 - xT1  # row values
            sub_image1 = np.array([np.array([image1[i + xB1][j + yL1] for j in range(width1)]) for i in range(height1)])
            
            # shift values in dictionaray
            subfeature1 = {}
            for key1 in feature1['facial_features_dict'].keys():
                orig1 = (feature1['facial_features_dict'])[key1]
                subfeature1[key1] = np.array(orig1) - np.array([xT1, yL1])
                
            # make subimage2
            # print("Feature 1:\t%s" %str(feature1))
            # print("Feature 2:\t%s" %str(feature2))
            if feature2['outer_bound_dict']:  # handle no bound box edge case
                xT2, yL2 = feature2['outer_bound_dict']['UPPER_LEFT']
                xB2, yR2 = feature2['outer_bound_dict']['LOWER_RIGHT']
            elif feature2[1]:
                xT2, yL2 = feature2['inner_bound_dict']['UPPER_LEFT']
                xB2, yR2 = feature2['inner_bound_dict']['LOWER_RIGHT']
            else:
                print("Something went wrong, spicy boi")
            width2 = yR2 - yL2  # col values
            height2 = xB2 - xT2  # row values
            sub_image2 = np.array([np.array([image2[i + xB2][j + yL2] for j in range(width2)]) for i in range(height2)])
                
            # shift values in dictionary
            subfeature2 = {}
            for key2 in feature2['facial_features_dict'].keys():
                orig2 = (feature2['facial_features_dict'])[key2]
                subfeature2[key2] = np.array([orig2]) - np.array([xT2, yL2])

            # get swapped subimage
            sub_swap_img = faceSwap2.swap_faces(sub_image1, sub_image2, feature1, feature2)
            print("swapped %d faces" % count)
            count += 1

            # insert subimage
            for i in range(height1):
                for j in range(width1):
                    image2[i + xB2][j + yL2] = sub_swap_img[i][j]

            # write image file to location specified
            cv2.imwrite(location, image1)


if __name__ == "__main__":
    # setup user data
    pipeline = Pipeline()
    user_image = "photos/aaron.jpg"
    # user_image = "photos/multiple.jpg"
    user_faces = pipeline.study_memes([user_image])[0]
    # print("USER FACE:\n%s\n\n" % str(user_faces))

    # scrape data
    image_urls = pipeline.get_n_memes(10)
    # process data
    cleaned_faces = pipeline.study_memes(image_urls)
    # print("CLEANED FACE:\n%s\n\n" % str(cleaned_faces))
    # swap individual images
    count = 1
    for face, img in zip(cleaned_faces, image_urls):
        print("creating art # %d" % count)
        pipeline.create_meme(user_image, img, user_faces, face, "art#%d.jpg" % count)
        count += 1

        print("Cleaned %d dirty faces" %count)
