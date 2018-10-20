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

        return img_paths
            
    def study_memes(self, img_paths):
        """
        Method to use google cloud api to analyze faces in images.
        :param img_paths: list of paths to images to study
        :return: a list of dictionaries containing dictionaries describing the faces
                  See: vision_detector.clean_face_features() for one entry in that list
        """
        clean_faces = []
        for local_path in img_paths:
            face = self.vision_detector.read_image(local_path)
            if face != None:
                cleaned_face = self.vision_detector.clean_face_features(face)

                if cleaned_face != None:
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
        random.seed(69)  # for debugging and the memes
        count = 1
        for feature2 in features2:
            feature1 = random.choice(features1)
            
            # make subimage1
            xT1, yL1 = feature1[0].UPPER_LEFT
            xB1, yR1 = feature1[0].BOTTOM_RIGHT
            width1 = yR1 - yL1  # col values
            height1 = xB1 - xT1  # row values
            sub_image1 = np.array([np.array([image1[i + xB1][j + yL1] for j in range(width1)]) for i in range(height1)])
            
            # shift values in dictionaray
            subfeature1 = {}
            for key1 in feature1[2].keys():
                subfeature1[key1] = np.array((feature1[2])[key1]) - np.array(xT1, yL1)
                
            # make subimage1
            xT2, yL2 = feature2[0].UPPER_LEFT
            xB2, yR2 = feature2[0].BOTTOM_RIGHT
            width2 = yR2 - yL2  # col values
            height2 = xB2 - xT2  # row values
            sub_image2 = np.array([np.array([image2[i + xB2][j + yL2] for j in range(width2)]) for i in range(height2)])
                
            # shift values in dictionary
            subfeature2 = {}
            for key2 in feature2[2].keys():
                subfeature2[key2] = np.array((feature2[2])[key2]) - np.array(xT2, yL2)

            sub_swap_img = faceSwap2.swap_faces(image1, image2,feature1,feature2, "art#%d.jpg" % count)
            print("swapped %d faces" % count)
            count += 1

            # insert subimage
            for i in range(height1):
                for j in range(width1):
                    image2[i + xB2][j + yL2] = sub_swap_img[i][j]

            # write image file to location specified
            cv2.imwrite(location, image1)

pipeline = Pipeline()
image_urls = pipeline.get_n_memes(50)
cleaned_faces = pipeline.study_memes(image_urls)
print(cleaned_faces)
