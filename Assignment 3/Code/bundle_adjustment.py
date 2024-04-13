import numpy as np
import cv2


def compute_ba_residuals(parameters: np.ndarray, intrinsics: np.ndarray, num_cameras: int, points2d: np.ndarray,
                         camera_idxs: np.ndarray, points3d_idxs: np.ndarray) -> np.ndarray:
    """
    For each point2d in <points2d>, find its 3d point, reproject it back into the image and return the residual
    i.e. euclidean distance between the point2d and reprojected point.

    Args:
        parameters: list of camera parameters [r1, r2, r3, t1, t2, t3, ...] where r1, r2, r3 corresponds to the
                    Rodriguez vector. There are 6C + 3M parameters where C is the number of cameras
        intrinsics: camera intrinsics 3 x 3 array
        num_cameras: number of cameras, C
        points2d: N x 2 array of 2d points
        camera_idxs: camera_idxs[i] returns the index of the camera for points2d[i]
        points3d_idxs: points3d[points3d_idxs[i]] returns the 3d point corresponding to points2d[i]

    Returns:
        N residuals

    """
    num_camera_parameters = 6 * num_cameras
    camera_parameters = parameters[:num_camera_parameters]
    points3d = parameters[num_camera_parameters:]
    num_points3d = points3d.shape[0] // 3
    points3d = points3d.reshape(num_points3d, 3)

    camera_parameters = camera_parameters.reshape(num_cameras, 6)
    camera_rvecs = camera_parameters[:, :3]
    camera_tvecs = camera_parameters[:, 3:]

    extrinsics = []
    for rvec in camera_rvecs:
        rot_mtx, _ = cv2.Rodrigues(rvec)
        extrinsics.append(rot_mtx)
    extrinsics = np.array(extrinsics)  # C x 3 x 3
    extrinsics = np.concatenate([extrinsics, camera_tvecs.reshape(-1, 3, 1)], axis=2)  # C x 3 x 4

    residuals = np.zeros(shape=points2d.shape[0], dtype=float)
    """ 
    YOUR CODE HERE: 
    NOTE: DO NOT USE LOOPS 
    HINT: I used np.matmul; np.sum; np.sqrt; np.square, np.concatenate etc.
    """
    proj_mtx = extrinsics[camera_idxs]
    proj_mtx = np.einsum('ij,kjl->kil', intrinsics, proj_mtx)

    homo_2d = np.concatenate((points2d, np.ones((points2d.shape[0],1))), axis=1)
    homo_3d = np.concatenate((points3d[points3d_idxs], np.ones((points3d[points3d_idxs].shape[0], 1))), axis=1)

    proj_2d = np.einsum('kj,klj->kl', homo_3d, proj_mtx)
    proj_2d = proj_2d / np.repeat(proj_2d[:,2].reshape(proj_2d.shape[0], 1), 3, axis=1) # Normalizing

    residuals = np.linalg.norm(homo_2d - proj_2d, axis=1)
    """ END YOUR CODE HERE """
    return residuals
