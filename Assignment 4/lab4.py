""" CS4277/CS5477 Lab 4: Plane Sweep Stereo
See accompanying Jupyter notebook (lab4.ipynb) for instructions.

Name: Li Honggunag
Email: e0725309@u.nus.edu
NUSNET ID: e0725309

"""
import json
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import scipy.ndimage

"""Helper functions: You should not have to touch the following functions.
"""
class Image(object):
    """
    Image class. You might find the following member variables useful:
    - image: RGB image (HxWx3) of dtype np.float64
    - pose_mat: 3x4 Camera extrinsics that transforms points from world to
        camera frame
    """
    def __init__(self, qvec, tvec, name, root_folder=''):
        self.qvec = qvec
        self.tvec = tvec
        self.name = name  # image filename
        self._image = self.load_image(os.path.join(root_folder, name))

        # Extrinsic matrix: Transforms from world to camera frame
        self.pose_mat = self.make_extrinsic(qvec, tvec)

    def __repr__(self):
        return '{}: qvec={}\n tvec={}'.format(
            self.name, self.qvec, self.tvec
        )

    @property
    def image(self):
        return self._image.copy()

    @staticmethod
    def load_image(path):
        """Loads image and converts it to float64"""
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im.astype(np.float64) / 255.0

    @staticmethod
    def make_extrinsic(qvec, tvec):
        """ Make 3x4 camera extrinsic matrix from colmap pose

        Args:
            qvec: Quaternion as per colmap format (q_cv) in the order
                  q_w, q_x, q_y, q_z
            tvec: translation as per colmap format (t_cv)

        Returns:

        """
        rotation = Rotation.from_quat(np.roll(qvec, -1))
        return np.concatenate([rotation.as_matrix(), tvec[:, None]], axis=1)

def write_json(outfile, images, intrinsic_matrix, img_hw):
    """Write metadata to json file.

    Args:
        outfile (str): File to write to
        images (list): List of Images
        intrinsic_matrix (np.ndarray): 3x3 intrinsic matrix
        img_hw (tuple): (image height, image width)
    """

    img_height, img_width = img_hw

    images_meta = []
    for im in images:
        images_meta.append({
            'name': im.name,
            'qvec': im.qvec.tolist(),
            'tvec': im.tvec.tolist(),
        })

    data = {
        'img_height': img_height,
        'img_width': img_width,
        'K': intrinsic_matrix.tolist(),
        'images': images_meta
    }
    with open(outfile, 'w') as fid:
        json.dump(data, fid, indent=2)


def load_data(root_folder):
    """Loads dataset.

    Args:
        root_folder (str): Path to data folder. Should contain metadata.json

    Returns:
        images, K, img_hw
    """
    print('Loading data from {}...'.format(root_folder))
    with open(os.path.join(root_folder, 'metadata.json')) as fid:
        metadata = json.load(fid)

    images = []
    for im in metadata['images']:
        images.append(Image(np.array(im['qvec']), np.array(im['tvec']),
                            im['name'], root_folder=root_folder))
    img_hw = (metadata['img_height'], metadata['img_width'])
    K = np.array(metadata['K'])

    print('Loaded data containing {} images.'.format(len(images)))
    return images, K, img_hw


def invert_extrinsic(cam_matrix):
    """Invert extrinsic matrix"""
    irot_mat = cam_matrix[:3, :3].transpose()
    trans_vec = cam_matrix[:3, 3, None]

    inverted = np.concatenate([irot_mat,  -irot_mat @ trans_vec], axis=1)
    return inverted


def concat_extrinsic_matrix(mat1, mat2):
    """Concatenate two 3x4 extrinsic matrices, i.e. result = mat1 @ mat2
      (ignoring matrix dimensions)
    """
    r1, t1 = mat1[:3, :3], mat1[:3, 3:]
    r2, t2 = mat2[:3, :3], mat2[:3, 3:]
    rot = r1 @ r2
    trans = r1@t2 + t1
    concatenated = np.concatenate([rot, trans], axis=1)
    return concatenated


def rgb2hex(rgb):
    """Converts color representation into hexadecimal representation for K3D

    Args:
        rgb (np.ndarray): (N, 3) array holding colors

    Returns:
        hex (np.ndarray): array (N, ) of size N, each element indicates the
          color, e.g. 0x0000FF = blue
    """
    rgb_uint = (rgb * 255).astype(np.uint8)
    hex = np.sum(rgb_uint * np.array([[256 ** 2, 256, 1]]),
                 axis=1).astype(np.uint32)
    return hex

"""Functions to be implemented
"""
# Part 1
def get_plane_sweep_homographies(K, relative_pose, inv_depths):
    """Compute plane sweep homographies, assuming fronto parallel planes w.r.t.
    reference camera

    Args:
        K (np.ndarray): Camera intrinsic matrix (3,3)
        relative_pose (np.ndarray): Relative pose between the two cameras
          of shape (3, 4)
        inv_depths (np.ndarray): Inverse depths to warp of size (D, )

    Returns:
        homographies (D, 3, 3)
    """

    homographies = []

    """ YOUR CODE STARTS HERE """
    R = relative_pose[:,:3].T
    C = R @ relative_pose[:,-1].reshape((3, 1))
    n = np.array([0, 0, 1]).reshape((3,1))

    for d in inv_depths:
        homographies.append(K @ (R.T + (R.T @ C @ n.T) * d) @ np.linalg.inv(K))
    """ YOUR CODE ENDS HERE """

    return np.array(homographies)

# Part 2
def compute_plane_sweep_volume(images, ref_pose, K, inv_depths, img_hw):
    """Compute plane sweep volume, by warping all images to the reference camera
    fronto-parallel planes, before computing the variance for each pixel and
    depth.

    Args:
        images (list[Image]): List of images which contains information about
          the camera extrinsics for each image
        ref_pose (np.ndarray): Reference camera pose
        K (np.ndarray): 3x3 intrinsic matrix (assumed same for all cameras)
        inv_depths (list): List of inverse depths to consider for plane sweep
        img_hw (tuple): tuple containing (H, W), which are the output height
          and width for the plane sweep volume.

    Returns:
        ps_volume (np.ndarray):
          Plane sweep volume of size (D, H, W), with dtype=np.float64, where
          D is len(inv_depths), and (H, W) are the image heights and width
          respectively. Each element should contain the variance of all pixel
          intensities that warp onto it.
        accum_count (np.ndarray):
          Accumulator count of same size as ps_volume, and dtype=np.int32.
          Keeps track of how many images are warped into a certain pixel,
          i.e. the number of pixels used to compute the variance.
    """

    D = len(inv_depths)
    H, W = img_hw
    ps_volume = np.zeros((D, H, W), dtype=np.float64)
    accum_count = np.zeros((D, H, W), dtype=np.int32)

    """ YOUR CODE STARTS HERE """
    ref_image = None

    # Find reference image
    for image_id in range(len(images)):
        image = images[image_id]
        image_extrinsics = image.pose_mat

        if (np.all(image_extrinsics == ref_pose)):
            ref_image = image.image

    # Compute variance
    for image_id in range(len(images)):
        image = images[image_id]
        image_extrinsics = image.pose_mat

        relative_pose = concat_extrinsic_matrix(ref_pose, invert_extrinsic(image_extrinsics)) 
        homographies = get_plane_sweep_homographies(K, relative_pose, inv_depths)
        for d in range(D):
            homography = homographies[d]
            transform_img = cv2.warpPerspective(src=image.image, M=homography, dsize=(W, H))
            
            mask = (transform_img != 0)
            accum_count[d][np.any(mask, axis=2)] += 1
            
            valid_ref_img = ref_image * mask
            diff = (valid_ref_img - transform_img)
            abs_diff = np.abs(diff)
            avg_abs_diff = np.mean(abs_diff, axis=2)

            ps_volume[d] += avg_abs_diff
    """ YOUR CODE ENDS HERE """
    print(ps_volume.shape)

    return ps_volume, accum_count

def compute_depths(ps_volume, inv_depths):
    """Computes inverse depth map from plane sweep volume as the
    argmin over plane sweep volume variances.

    Args:
        ps_volume (np.ndarray): Plane sweep volume of size (D, H, W) from
          compute_plane_sweep_volume()
        inv_depths (np.ndarray): List of depths considered in the plane
          sweeping (D,)

    Returns:
        inv_depth_image (np.ndarray): inverse-depth estimate (H, W)
    """

    inv_depth_image = np.zeros(ps_volume.shape[1:], dtype=np.float64)

    """ YOUR CODE STARTS HERE """
    least_var_idx = np.argmin(ps_volume, axis=0)
    inv_depth_image = inv_depths[least_var_idx]
    """ YOUR CODE ENDS HERE """

    return inv_depth_image


# Part 3
def post_process(ps_volume, inv_depths, accum_count):
    """Post processes the plane sweep volume and compute a mask to indicate
    which pixels have confident estimates of the depth

    Args:
        ps_volume: Plane sweep volume from compute_plane_sweep_volume()
          of size (D, H, W)
        inv_depths (List[float]): List of depths considered in the plane
          sweeping
        accum_count: Accumulator count from compute_plane_sweep_volume(), which
          can be used to indicate which pixels are not observed by many other
          images.

    Returns:
        inv_depth_image: Denoised Inverse depth image (similar to compute_depths)
        mask: np.ndarray of size (H, W) and dtype np.bool.
          Pixels with values TRUE indicate valid pixels.
    """

    mask = np.ones(ps_volume.shape[1:], dtype=bool)
    inv_depth_image = np.zeros(ps_volume.shape[1:], dtype=np.float64)
    #print(accum_count)
    """ YOUR CODE STARTS HERE """

    # Consider those with count less than 10th percentile as insufficient observations
    count_threshold = np.percentile(accum_count.flatten(), 0.1)
    valid_count_mask = (accum_count >= count_threshold)
    valid_ps_volume = np.where(valid_count_mask, ps_volume, np.inf)

    # Retrieve depth image after removing insufficient observations
    depth_image = compute_depths(valid_ps_volume, inv_depths)

    # Apply median filtering to reduce salt and pepper noise
    median_filtered = scipy.ndimage.median_filter(depth_image, 3)
    original = compute_depths(ps_volume, inv_depths)

    # As we are using median filtering, we find a threshold for the depth change to consider pixel as invalid
    change_threshold = np.percentile((median_filtered, original), 0.95)
    mask = ((median_filtered - original) <= change_threshold)
    inv_depth_image = median_filtered
    """ YOUR CODE ENDS HERE """

    return inv_depth_image, mask


# Part 4
def unproject_depth_map(image, inv_depth_image, K, mask=None):
    """Converts the depth map into points by unprojecting depth map into 3D

    Note: You will also need to implement the case where no mask is provided

    Args:
        image (np.ndarray): Image bitmap (H, W, 3)
        inv_depth_image (np.ndarray): Inverse depth image (H, W)
        K (np.ndarray): 3x3 Camera intrinsics
        mask (np.ndarray): Optional mask of size (H, W) and dtype=np.bool.

    Returns:
        xyz (np.ndarray): Nx3 coordinates of points, dtype=np.float64.
        rgb (np.ndarray): Nx3 RGB colors, where rgb[i, :] is the (Red,Green,Blue)
          colors for the points at position xyz[i, :]. Should be in the range
          [0, 1] and have dtype=np.float64.
    """

    xyz = np.zeros([0, 3], dtype=np.float64)
    rgb = np.zeros([0, 3], dtype=np.float64)  # values should be within (0, 1)
    H, W = image.shape[0:2]
    """ YOUR CODE STARTS HERE """
    points3d = np.zeros((H*W, 3))
    pointsrgb = np.zeros((H*W, 3))

    x, y = np.meshgrid(range(W), range(H))
    xy_homo = np.hstack((x.reshape(H*W, 1), 
                         y.reshape(H*W, 1), 
                         np.ones((H*W,1))))
    depths = np.power(inv_depth_image, -1).reshape((H*W))

    inv_K = np.linalg.inv(K)
    rays = np.apply_along_axis(lambda x: inv_K @ x, 1, xy_homo)
    
    for i in range(len(rays)):
        ray = rays[i]
        z = depths[i]
        points3d[i] = (z * ray).flatten()

    pointsrgb = image.reshape((H*W, 3))

    if (mask is not None):
        points3d = points3d[mask.reshape((H*W)),:]
        pointsrgb = pointsrgb[mask.reshape((H*W)),:]
    """ YOUR CODE ENDS HERE """

    xyz = np.array(points3d)
    rgb = np.array(pointsrgb)
    return xyz, rgb
