# run this before running file
# export GOOGLE_APPLICATION_CREDENTIALS="meme_swap_owner_account_key.json"
import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types


class VisionDetector:
    def __init__(self):
        # Instantiates a client
        self.client = vision.ImageAnnotatorClient()
    
    def read_image(self, image):
        '''
        Send an image to Vision API and find facial features

        Input:
            image: string of directory/file_name
        Output:
            returns FaceAnnotation object
        '''
        # The name of the image file to annotate
        file_name = os.path.join(
            os.path.dirname(__file__), image)

        # Loads the image into memory
        with io.open(file_name, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

        # Performs landmark detection on the image file (eyes, etc.)
        response = self.client.face_detection(image=image)
        face = response.face_annotations

        return face[0]

    def clean_face_features(self, face):
        '''
        Given a set of facial features, return relevant data points

        Input:
            face: JSON of facial features
        Output:
            A dictionary of:
                outer_bound_dict: dict(corner, (x,y)),
                inner_bound_dict: dict(corner, (x,y)),
                facial_features:           dict(feature_name, (x,y))

            
        NOTE: roll_angle is angle theta relative to vertical y-axis clockwise
        '''

        corners = ['LOWER_LEFT', 'LOWER_RIGHT', 'UPPER_RIGHT', 'UPPER_LEFT']
        
        # outer square 
        outer_bound = face.bounding_poly     # entire face
        outer_bound_dict = {}
        for corner, vertex in zip(corners, outer_bound.vertices):
            outer_bound_dict[corner] = (vertex.x, vertex.y)
            
        # inner square
        inner_bound = face.fd_bounding_poly  # only skin part
        inner_bound_dict = {}
        for corner, vertex in zip(corners, inner_bound.vertices):
            inner_bound_dict[corner] = (vertex.x, vertex.y)

        # map int (constant type) to readable string
        type_int_to_string_dict = {
            0: 'UNKNOWN_LANDMARK',
            1: 'LEFT_EYE',
            2: 'RIGHT_EYE',
            3: 'LEFT_OF_LEFT_EYEBROW',
            4: 'RIGHT_OF_LEFT_EYEBROW',
            5: 'LEFT_OF_RIGHT_EYEBROW',
            6: 'RIGHT_OF_RIGHT_EYEBROW',
            7: 'MIDPOINT_BETWEEN_EYES',
            8: 'NOSE_TIP',
            9: 'UPPER_LIP',
            10: 'LOWER_LIP',
            11: 'MOUTH_LEFT',
            12: 'MOUTH_RIGHT',
            13: 'MOUTH_CENTER',
            14: 'NOSE_BOTTOM_RIGHT',
            15: 'NOSE_BOTTOM_LEFT',
            16: 'NOSE_BOTTOM_CENTER',
            17: 'LEFT_EYE_TOP_BOUNDARY',
            18: 'LEFT_EYE_RIGHT_CORNER',
            19: 'LEFT_EYE_BOTTOM_BOUNDARY',
            20: 'LEFT_EYE_LEFT_CORNER',
            21: 'RIGHT_EYE_TOP_BOUNDARY',
            22: 'RIGHT_EYE_RIGHT_CORNER',
            23: 'RIGHT_EYE_BOTTOM_BOUNDARY',
            24: 'RIGHT_EYE_LEFT_CORNER',
            25: 'LEFT_EYEBROW_UPPER_MIDPOINT',
            26: 'RIGHT_EYEBROW_UPPER_MIDPOINT',
            27: 'LEFT_EAR_TRAGION',
            28: 'RIGHT_EAR_TRAGION',
            29: 'LEFT_EYE_PUPIL',
            30: 'RIGHT_EYE_PUPIL',
            31: 'FOREHEAD_GLABELLA',
            32: 'CHIN_GNATHION',
            33: 'CHIN_LEFT_GONION',
            34: 'CHIN_RIGHT_GONION'
        }

        # map (x,y) to feature type in dictionary
        landmarks_dict = {}
        for landmark in face.landmarks:
            # convert 'type' (int) to a string name
            feature_type_int = landmark.type
            landmark_key = type_int_to_string_dict[feature_type_int]
            
            # match (x,y) tuple and insert
            position = (landmark.position.x, landmark.position.y)
            landmarks_dict[landmark_key] = position

        return {'outer_bound_dict': outer_bound_dict,
                'inner_bound_dict': inner_bound_dict,
                'facial_features_dict': landmarks_dict}
        
vision = VisionDetector() 
face = vision.read_image('images/meme5.jpg')
clean_face = vision.clean_face_features(face)
print(clean_face)
