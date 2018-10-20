#!/usr/bin/python

# Copyright (c) 2015 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.
import cv2
import numpy

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR_FRAC = 0.6


def draw_convex_hull(im, points, color):
    """
    Method from original faceSwap. Fills in the convex hull generated from points
    in the image with a specific color.

    :param im: Image to be edited
    :param points: List of points generating the convex hull
    :param color: Color to fill in the image with (0.0 to 1.0)
    :return: void
    """
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    """
    Method from original faceSwap. Generates mask refering to regions covered by the
    convex hull formed by the sets of points in overlay_points
    :param im: The image that the mask refers to
    :param landmarks: the points in the image being used to find regions of interest in the mask
    :param overlay_points: A collection of list of points corresponding to regions of interest for swapping
    :return: A mask of points in image to be replaced/moved to another image
    """
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)

    # whites out mouth and eyes
    for group in [landmarks]:  # just a reminder for how we could segment face swap by features
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = numpy.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_from_points(points1, points2):
    """
    Method from original faceSwap.
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    :param points1: points of interest in image one
    :param points2: points of image 2 to map to points1
    :return: Transformation matrix approximating transformation from image2 onto image1 by solving procrustes problem.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    # singular value decomposition
    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


def warp_im(im, M, dshape):
    """
    Method from original faceSwap. Applies the transformation matrix M to im
    :param im: Image to be warped
    :param M: Transformation matrix
    :param dshape: ???
    :return: The image after being transformed by matrix
    """
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, left_eye_points, right_eye_points):
    """

    :param im1: Base image in color correction
    :param im2: Additional image in color correction
    :param landmarks1: set of points of interest containing both eyes
    :param left_eye_points: set of xy points describing the left eye
    in im1
    :param right_eye_points: set of xy points describing the right eye in im2
    :return: Im2 after being color corrected
    """
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(left_eye_points, axis=0) -
                              numpy.mean(right_eye_points, axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
            im2_blur.astype(numpy.float64))


def subset(dict1, dict2):
    """
    Helper method to find the intersecting subset of key-value pairs contained by two dictionaries.
    :param dict1: a dictionary
    :param dict2: another dictionary
    :return: Tuple of dicts containing the values with identical keys in both dicts
    """
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    intersection1 = dict1.fromkeys(keys1 & keys2)
    intersection2 = dict2.fromkeys(keys1 & keys2)
    for key in intersection1.keys():
        intersection1[key] = dict1[key]
        intersection2[key] = dict2[key]

    return intersection1, intersection2


def find_eyes(landmarks):
    """
    Helper method to find all key-value pairs in a dict whose keys reference 'left_eye' and 'right_eye'
    The dictionary should have string keys or it will return a None tuple
    :param landmarks: Dictionary with String keys referencing left and right eyes
    :return: a tuple of lists containing pixel locations of landmarks representing left_eye and right_eye
    """
    left_eye = {}
    right_eye = {}
    for key in landmarks.keys():
        if str(key).startswith("left_eye"):
            left_eye[key] = landmarks[key]
        elif key.startswith("right_eye"):
            right_eye[key] = landmarks[key]
    return left_eye, right_eye


def swap_faces(im1, im2, features1, features2, location):
    """
    Method to write out an image putting the face in im2 over the face in im1.
    Writes out to file at location (must be jpg probably)
    :param im1: Base image whose face will be replaced
    :param im2: Image whose face will be in final image
    :param features1: Information on im1 including a large and small bounding box
                      for the face as well as a dictionary of landmark points
    :param features2: Information on im2 including a large and small bounding box
                      for the face as well as a dictionary of landmark points
    :param location: The file to write the final image to
    :return: void
    """
    landmarks1 = features1[2]
    landmarks2 = features2[2]
    # convert dicts into lists for mask style accessability (important)
    landmarks1, landmarks2 = subset(landmarks1, landmarks2)
    left_eye1, right_eye1 = find_eyes(landmarks1)
    left_eye1 = left_eye1.values()
    right_eye1 = right_eye1.values()
    landmarks1 = landmarks1.values()
    landmarks2 = landmarks2.values()

    # calculate points used for aligning image
    # calculate transformation matrix
    m = transformation_from_points(landmarks1, landmarks2)
    # calculate mask for im2
    mask = get_face_mask(im2, landmarks2)
    # transform the mask of im2
    warped_mask = warp_im(mask, m, im1.shape)
    combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
                              axis=0)
    # warp and corredt im2 to mask onto im1
    warped_im2 = warp_im(im2, m, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, left_eye1, right_eye1)
    # mask im2 onto im1
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    print("Writing to: %s" % location)
    cv2.imwrite(location, output_im)
    return location
