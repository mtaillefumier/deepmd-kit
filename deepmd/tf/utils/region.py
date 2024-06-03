# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.tf.env import (
    tf,
)


def to_face_distance(cell):
    """Compute the to-face-distance of the simulation cell.

    Parameters
    ----------
    cell : tf.Tensor
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    dist: tf.Tensor
        the to face distances of shape [*, 3]
    """
    # generated by GitHub Copilot, converted from PT codes
    cshape = tf.shape(cell)
    cell_reshaped = tf.reshape(cell, [-1, 3, 3])
    dist = b_to_face_distance(cell_reshaped)
    return tf.reshape(dist, tf.concat([cshape[:-2], [3]], 0))


def b_to_face_distance(cell):
    # generated by GitHub Copilot, converted from PT codes
    volume = tf.linalg.det(cell)
    c_yz = tf.linalg.cross(cell[:, 1], cell[:, 2])
    _h2yz = tf.divide(volume, tf.norm(c_yz, axis=-1))
    c_zx = tf.linalg.cross(cell[:, 2], cell[:, 0])
    _h2zx = tf.divide(volume, tf.norm(c_zx, axis=-1))
    c_xy = tf.linalg.cross(cell[:, 0], cell[:, 1])
    _h2xy = tf.divide(volume, tf.norm(c_xy, axis=-1))
    return tf.stack([_h2yz, _h2zx, _h2xy], axis=1)