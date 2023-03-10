import numpy as np
from pyquaternion import Quaternion


def tf(translation, rotation):
    """
    Build transformation matrix
    """
    if not isinstance(rotation, Quaternion):
        assert isinstance(rotation, list) or isinstance(rotation, np.ndarray), f"{type(rotation)} is not supported"
        rotation = Quaternion(rotation)
    tf_mat = np.eye(4)
    tf_mat[:3, :3] = rotation.rotation_matrix
    tf_mat[:3, 3] = translation
    return tf_mat


def apply_tf(tf: np.ndarray, points: np.ndarray):
    assert points.shape[1] == 3
    assert tf.shape == (4, 4)
    points_homo = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    out = tf @ points_homo.T
    return out[:3, :].T
