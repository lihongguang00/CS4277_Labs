""" CS4277/CS5477 Lab 1: Metric Rectification and Robust Homography Estimation.
See accompanying file (lab1.pdf) for instructions.

Name: Li Hongguang
Email: e0725309@u.nus.edu
Student ID: A0233309L
"""

import numpy as np
import cv2
from helper import *
from math import floor, ceil, sqrt



def compute_homography(src, dst):
    """Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    """

    h_matrix = np.eye(3, dtype=np.float64)

    """ YOUR CODE STARTS HERE """
    # Compute normalization matrix
    centroid_src = np.mean(src, axis=0)
    d_src = np.linalg.norm(src - centroid_src[None, :], axis=1)
    s_src = sqrt(2) / np.mean(d_src)
    T_norm_src = np.array([[s_src, 0.0, -s_src * centroid_src[0]],
                           [0.0, s_src, -s_src * centroid_src[1]],
                           [0.0, 0.0, 1.0]])

    centroid_dst = np.mean(dst, axis=0)
    d_dst = np.linalg.norm(dst - centroid_dst[None, :], axis=1)
    s_dst = sqrt(2) / np.mean(d_dst)
    T_norm_dst = np.array([[s_dst, 0.0, -s_dst * centroid_dst[0]],
                           [0.0, s_dst, -s_dst * centroid_dst[1]],
                           [0.0, 0.0, 1.0]])

    srcn = transform_homography(src, T_norm_src)
    dstn = transform_homography(dst, T_norm_dst)

    # Compute homography
    n_corr = srcn.shape[0]
    A = np.zeros((n_corr*2, 9), dtype=np.float64)
    for i in range(n_corr):
        A[2 * i, 0] = srcn[i, 0]
        A[2 * i, 1] = srcn[i, 1]
        A[2 * i, 2] = 1.0
        A[2 * i, 6] = -dstn[i, 0] * srcn[i, 0]
        A[2 * i, 7] = -dstn[i, 0] * srcn[i, 1]
        A[2 * i, 8] = -dstn[i, 0] * 1.0

        A[2 * i + 1, 3] = srcn[i, 0]
        A[2 * i + 1, 4] = srcn[i, 1]
        A[2 * i + 1, 5] = 1.0
        A[2 * i + 1, 6] = -dstn[i, 1] * srcn[i, 0]
        A[2 * i + 1, 7] = -dstn[i, 1] * srcn[i, 1]
        A[2 * i + 1, 8] = -dstn[i, 1] * 1.0

    u, s, vt = np.linalg.svd(A)
    h_matrix_n = np.reshape(vt[-1, :], (3, 3))

    # Unnormalize homography
    h_matrix = np.linalg.inv(T_norm_dst) @ h_matrix_n @ T_norm_src
    h_matrix /= h_matrix[2, 2]

    # src = src.astype(np.float32)
    # dst = dst.astype(np.float32)
    # h_matrix = cv2.findHomography(src, dst)[0].astype(np.float64)
    """ YOUR CODE ENDS HERE """

    return h_matrix


def transform_homography(src, h_matrix):
    """Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    Prohibited functions:
        cv2.perspectiveTransform()

    """
    transformed = None

    """ YOUR CODE STARTS HERE """
    transformed = np.copy(src)
    for i in range(len(transformed)):
        x, y = src[i]
        new_coordinates = np.matmul(h_matrix, np.array([x, y, 1]).reshape(3, 1))
        transformed[i][0] = new_coordinates[0] / new_coordinates[2]
        transformed[i][1] = new_coordinates[1] / new_coordinates[2]
    """ YOUR CODE ENDS HERE """
    
    return transformed


def warp_image(src, dst, h_matrix):
    """Applies perspective transformation to source image to warp it onto the
    destination (background) image

    Args:
        src (np.ndarray): Source image to be warped
        dst (np.ndarray): Background image to warp template onto
        h_matrix (np.ndarray): Warps coordinates from src to the dst, i.e.
                                 x_{dst} = h_matrix * x_{src},
                               where x_{src}, x_{dst} are the homogeneous
                               coordinates in I_{src} and I_{dst} respectively

    Returns:
        dst (np.ndarray): Source image warped onto destination image

    Prohibited functions:
        cv2.warpPerspective()
    You may use the following functions: np.meshgrid(), cv2.remap(), transform_homography()
    """
    dst = dst.copy()  # deep copy to avoid overwriting the original image

    """ YOUR CODE STARTS HERE """
    height = np.shape(dst)[0]
    width = np.shape(dst)[1]
    y = np.linspace(0, height - 1, height)
    x = np.linspace(0, width - 1, width)
    xgrid, ygrid = np.meshgrid(x, y)
    M = np.vstack((xgrid.flatten(), ygrid.flatten()))
    M = np.transpose(M)
    destination = transform_homography(M, np.linalg.inv(h_matrix))
    destination = destination.astype(np.float32)
    map_x = destination[:,0].reshape(height, width).astype(np.float32)
    map_y = destination[:,1].reshape(height, width).astype(np.float32)
    cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT, dst=dst)
    """ YOUR CODE ENDS HERE """
##    cv2.warpPerspective(src, h_matrix, dsize=dst.shape[1::-1],
##                         dst=dst, borderMode=cv2.BORDER_TRANSPARENT)
    return dst

def compute_affine_rectification(src_img:np.ndarray,lines_vec: list):
    '''
       The first step of the stratification method for metric rectification. Compute
       the projective transformation matrix Hp with line at infinity. At least two
       parallel line pairs are required to obtain the vanishing line. Then warping
       the image with the predicted projective transformation Hp to recover the affine
       properties. X_dst=Hp*X_src

       Args:
           src_img: Original image X_src
           lines_vec: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           Xa: Affinely rectified image by removing projective distortion

    '''
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    Hp= np.zeros((3,3))
    """ YOUR CODE STARTS HERE """
    line_inf_list = []
    for i in range(len(lines_vec) // 2):
        l1 = lines_vec[i*2]
        l2 = lines_vec[i*2+1]
        line_inf_list.append(l1.intersetion_point(l2))
    line_inf = Line(line_inf_list[0], line_inf_list[1])
    A, B, C = line_inf.vec_para
    h_mat = np.eye(3)
    h_mat[0][2] = -A / C
    h_mat[1][2] = -B / C
    h_mat[2][2] = 1 / C
    h_mat_inv = np.linalg.inv(h_mat)
    h_mat_inv_t = np.transpose(h_mat_inv)
    scale_factor = 0.24
    scale_matrix = np.array([[scale_factor, 0, 0],
                             [0, scale_factor, 0],
                             [0, 0, 1]])
    Hp = np.matmul(scale_matrix, h_mat_inv_t)
    dst = warp_image(src_img, dst, Hp)
    """ YOUR CODE ENDS HERE """
   
    return dst
     


def compute_metric_rectification_step2(src_img:np.ndarray,line_vecs: list):
    '''
       The second step of the stratification method for metric rectification. Compute
       the affine transformation Ha with the degenerate conic from at least two
       orthogonal line pairs. Then warping the image with the predicted affine
       transformation Ha to recover the metric properties. X_dst=Ha*X_src

       Args:
           src_img: Affinely rectified image X_src
           line_vecs: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           X_dst: Image after metric rectification

    '''
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    Ha = np.zeros((3, 3))
    """ YOUR CODE STARTS HERE """
    l1 = line_vecs[0]
    m1 = line_vecs[1]
    l2 = line_vecs[2]
    m2 = line_vecs[3]
    l1_x, l1_y, _ = l1.vec_para
    m1_x, m1_y, _ = m1.vec_para
    l2_x, l2_y, _ = l2.vec_para
    m2_x, m2_y, _ = m2.vec_para
    constraints = np.array([[l1_x*m1_x, l1_x*m1_y + l1_y*m1_x, l1_y*m1_y],
                            [l2_x*m2_x, l2_x*m2_y + l2_y*m2_x, l2_y*m2_y]])
    
    U, Sigma, V_T = np.linalg.svd(constraints)
    s = V_T[-1]        
    S = np.array([[s[0], s[1]],
                  [s[1], s[2]]])
    K = np.linalg.cholesky(S)
    Ha[0][0], Ha[0][1], Ha[1][0], Ha[1][1] = K[0][0], K[0][1], K[1][0], K[1][1]
    Ha[2][2] = 1
    Ha = np.linalg.inv(Ha)
    scale_factor = 0.7
    scale_matrix = np.array([[scale_factor, 0, 0],
                             [0, scale_factor, 0],
                             [0, 0, 1]])
    Ha = np.matmul(scale_matrix, Ha)
    dst = warp_image(src_img, dst, Ha)
    """ YOUR CODE ENDS HERE """

  
    return dst

def compute_metric_rectification_one_step(src_img:np.ndarray,line_vecs: list):
    '''
       One-step metric rectification. Compute the transformation matrix H (i.e. H=HaHp) directly
       from five orthogonal line pairs. Then warping the image with the predicted affine
       transformation Ha to recover the metric properties. X_dst=H*X_src
       Args:
           src_img: Original image Xc
           line_infinity: list of lines constraint with homogeneous form (A,B,C) (i.e Ax+By+C=0)
       Returns:
           Xa: Image after metric rectification

    '''
    dst = np.zeros_like(src_img)  # deep copy to avoid overwriting the original image
    H = np.zeros((3, 3))

    """ YOUR CODE STARTS HERE """
    l1, m1 = line_vecs[0], line_vecs[1]
    l2, m2 = line_vecs[2], line_vecs[3]
    l3, m3 = line_vecs[4], line_vecs[5]
    l4, m4 = line_vecs[6], line_vecs[7]
    l5, m5 = line_vecs[8], line_vecs[9]

    l1_x, l1_y, l1_z = l1.vec_para
    m1_x, m1_y, m1_z = m1.vec_para
    constraint_1 = np.array([l1_x*m1_x, (l1_x*m1_y + l1_y*m1_x)/2, l1_y*m1_y,\
                             (l1_x*m1_z + l1_z*m1_x)/2,\
                             (l1_y*m1_z + l1_z*m1_y)/2, l1_z * m1_z])

    l2_x, l2_y, l2_z = l2.vec_para
    m2_x, m2_y, m2_z = m2.vec_para
    constraint_2 = np.array([l2_x*m2_x, (l2_x*m2_y + l2_y*m2_x)/2, l2_y*m2_y,\
                             (l2_x*m2_z + l2_z*m2_x)/2,\
                             (l2_y*m2_z + l2_z*m2_y)/2, l2_z * m2_z])

    l3_x, l3_y, l3_z = l3.vec_para
    m3_x, m3_y, m3_z = m3.vec_para
    constraint_3 = np.array([l3_x*m3_x, (l3_x*m3_y + l3_y*m3_x)/2, l3_y*m3_y,\
                             (l3_x*m3_z + l3_z*m3_x)/2,\
                             (l3_y*m3_z + l3_z*m3_y)/2, l3_z * m3_z])
    
    l4_x, l4_y, l4_z = l4.vec_para
    m4_x, m4_y, m4_z = m4.vec_para
    constraint_4 = np.array([l4_x*m4_x, (l4_x*m4_y + l4_y*m4_x)/2, l4_y*m4_y,\
                             (l4_x*m4_z + l4_z*m4_x)/2,\
                             (l4_y*m4_z + l4_z*m4_y)/2, l4_z * m4_z])

    l5_x, l5_y, l5_z = l5.vec_para
    m5_x, m5_y, m5_z = m5.vec_para
    constraint_5 = np.array([l5_x*m5_x, (l5_x*m5_y + l5_y*m5_x)/2, l5_y*m5_y,\
                             (l5_x*m5_z + l5_z*m5_x)/2,\
                             (l5_y*m5_z + l5_z*m5_y)/2, l5_z * m5_z])
    
    constraint_mat = np.vstack([constraint_1, constraint_2, constraint_3, constraint_4, constraint_5])

    U, Sigma, V_T = np.linalg.svd(constraint_mat)
    null_vec = V_T[-1]
    C_inf_img = np.array([[null_vec[0], null_vec[1]/2, null_vec[3]/2],
                          [null_vec[1]/2, null_vec[2], null_vec[4]/2],
                          [null_vec[3]/2, null_vec[4]/2, null_vec[5]]])
    U_C, Sigma_C, _ = np.linalg.svd(C_inf_img)
    scale_mat = np.sqrt(np.array([[Sigma_C[0], 0, 0],
                                  [0, Sigma_C[1], 0],
                                  [0, 0, 1]]))
    H = np.matmul(U_C, scale_mat)
    H = np.linalg.inv(H)

    dx, dy = 100, 80
    scale_factor = -0.15
    offset_matrix = np.matrix([[scale_factor, 0, dx],
                               [0, scale_factor, dy],
                               [0, 0, 1]])
    H = np.matmul(offset_matrix, H)
    dst = warp_image(src_img,dst, H)
    """ YOUR CODE ENDS HERE """

    return dst


def compute_homography_error(src, dst, homography):
    """Compute the squared bidirectional pixel reprojection error for
    provided correspondences

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        homography (np.ndarray): Homography matrix that transforms src to dst.

    Returns:
        err (np.ndarray): Array of size (N, ) containing the error d for each
        correspondence, computed as:
          d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
        where ||a|| denotes the l2 norm (euclidean distance) of vector a.
    """
    d = np.zeros(src.shape[0], np.float64)

    """ YOUR CODE STARTS HERE """
    H_inv = np.linalg.inv(homography)
    dst_est = transform_homography(src, homography)
    src_est = transform_homography(dst, H_inv)
    dst_err = (dst - dst_est)
    src_err = (src - src_est)
    d = np.diagonal(np.matmul(dst_err, np.transpose(dst_err))) + \
        np.diagonal(np.matmul(src_err, np.transpose(src_err)))
    """ YOUR CODE ENDS HERE """
    return d


def compute_homography_ransac(src, dst, thresh=16.0, num_tries=200):
    """Calculates the perspective transform from at least 4 points of
    corresponding points in a robust manner using RANSAC. After RANSAC, all the
    inlier correspondences will be used to re-estimate the homography matrix.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)
        thresh (float): Maximum allowed squared bidirectional pixel reprojection
          error to treat a point pair as an inlier (default: 16.0). Pixel
          reprojection error is computed as:
            d(x,x') = ||x - inv(H)x'||^2 +  ||x' - Hx||^2,
          where ||a|| denotes the l2 norm (euclidean distance) of vector a.
        num_tries (int): Number of trials for RANSAC

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.
        mask (np.ndarraay): Output mask with dtype np.bool where 1 indicates
          inliers

    Prohibited functions:
        cv2.findHomography()
    """

    h_matrix = np.eye(3, dtype=np.float64)
    mask = np.ones(src.shape[0], dtype=bool)

    """ YOUR CODE STARTS HERE """
    best = None
    most_inlier = 0
    for i in range(num_tries):
        rand_indices = np.random.choice(src.shape[0], 4, False)
        src_selected = src[rand_indices]
        dst_selected = dst[rand_indices]
        H = compute_homography(src_selected, dst_selected)

        error_vect = compute_homography_error(src, dst, H)
        num_inlier = len(error_vect[error_vect < thresh])

        if num_inlier > most_inlier:
            best = H
            most_inlier = num_inlier
            mask = error_vect < thresh
    h_matrix = best
    """ YOUR CODE ENDS HERE """

    return h_matrix, mask


