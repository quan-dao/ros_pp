import numpy as np
from typing import Tuple

from .batch_kalman_trackers import *


'''
State: (x, y, z, yaw, l, w, h, dx, dy, dz, dyaw)

track = [state || num_hit, num_miss, id]

observations = (x, y, z, yaw, l, w, h)
'''
YAW_INDEX = 3
CLASS_NAMES = ['car','truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']


def put_angle_in_range(angles: np.ndarray) -> np.ndarray:
        return np.arctan2(np.sin(angles), np.cos(angles))


def wrapper_kf_update(tracks: np.ndarray, tracks_P: np.ndarray, observations: np.ndarray, H: np.ndarray, S: np.ndarray) -> Tuple[np.ndarray]:
    yaw_idx = YAW_INDEX
    # =======================
    # orientation correction

    tracks[:, yaw_idx] = put_angle_in_range(tracks[:, yaw_idx])
    observations[:, yaw_idx] = put_angle_in_range(observations[:, yaw_idx])

    diff_yaw = np.abs(observations[:, yaw_idx] - tracks[:, yaw_idx])
    mask = np.logical_and(diff_yaw > np.pi / 2.0, diff_yaw < 3 * np.pi / 2.0)
    tracks[mask, yaw_idx] = put_angle_in_range(tracks[mask, yaw_idx] + np.pi)

    # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
    mask = np.abs(observations[:, yaw_idx] - tracks[:, yaw_idx]) >= 3 * np.pi / 2.0
    mask_positive_obs_angle = observations[:, yaw_idx] > 0
    tracks[mask & mask_positive_obs_angle, yaw_idx] += 2 * np.pi
    tracks[mask & np.logical_not(mask_positive_obs_angle), yaw_idx] -= 2 * np.pi
    # =======================

    tracks, tracks_P = batch_kf_update(tracks, tracks_P, observations, H, S)

    tracks[:, yaw_idx] = put_angle_in_range(tracks[:, yaw_idx])

    return tracks, tracks_P


def wrapper_kf_predict(tracks: np.ndarray, tracks_P: np.ndarray, F: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray]:
    yaw_idx = YAW_INDEX
    tracks, tracks_P = batch_kf_predict(tracks, tracks_P, F, Q)
    tracks[:, yaw_idx] = put_angle_in_range(tracks[:, yaw_idx])
    return tracks, tracks_P


def track_1step(tracks: np.ndarray, tracks_P: np.ndarray, track_counter: int, dets: np.ndarray, 
                chosen_class_idx: int, cost_threshold=11.0) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Args:
        tracks: (N, 11 + 3) - state (11), info (3)
        tracks_P: (N, 11, 11) - tracks' covariance matrix
        track_counter: to make track_id
        detections: (M, 9) - detections [x, y, z, dx, dy, dz, yaw, score, label]
        chosen_class_idx: class of object to track
    """
    assert dets.shape[1] == 7, f"expect 7 (x, y, z, yaw, dx, dy, dz), get {dets.shape[1]}"
    F = TRANSITION_MATRIX
    H = MEASUREMENT_MATRIX
    cls_name = CLASS_NAMES[chosen_class_idx]
    Q = np.diag(MATRICES_Q[cls_name])
    P_0 = np.diag(MATRICES_P[cls_name])
    R = np.diag(MATRICES_R[cls_name])
    
    # KF prediction
    tracks, tracks_P = wrapper_kf_predict(tracks, tracks_P, F, Q)

    # KF update
    if dets.shape[0] > 0 and tracks.shape[0] > 0:
        tracks_obs, tracks_S = batch_kf_predict_measurement(tracks, tracks_P, H, R)
        cost = build_cost_matrix(tracks_obs, tracks_S, dets)

        # assoc
        matches, unmatched_dets_idx, unmatched_tracks_idx = wrapper_matching(cost, cost_threshold)
        
        # ---
        # matched
        if matches.shape[0] > 0:
            tracks[matches[:, 1]], tracks_P[matches[:, 1]] = wrapper_kf_update(tracks[matches[:, 1]], 
                                                                            tracks_P[matches[:, 1]], 
                                                                            dets[matches[:, 0]], 
                                                                            H, R)
            # update matched tracks' info
            tracks[matches[:, 1], -3] += 1  # num_hit
            tracks[matches[:, 1], -2] = 0  # num_miss get reset

        # ---
        # unmatched tracks
        if unmatched_tracks_idx.shape[0] > 0:
            # update tracks info
            tracks[unmatched_tracks_idx, -3] = 0  # num_hit get reset
            tracks[unmatched_tracks_idx, -2] += 1

        # ---
        # unmatched dets
        if unmatched_dets_idx.shape[0] > 0:
            new_tracks, new_tracks_P, track_counter = spawn_new_tracks(dets[unmatched_dets_idx], P_0, track_counter)
            tracks = np.concatenate([tracks, new_tracks])
            tracks_P = np.concatenate([tracks_P, new_tracks_P])

    else:
        if dets.shape[0] == 0 and tracks.shape[0] > 0:
            # update tracks info
            tracks[:, -3] = 0  # num_hit get reset
            tracks[:, -2] += 1
        
        elif tracks.shape[0] == 0 and dets.shape[0] > 0:
            tracks, tracks_P, track_counter = spawn_new_tracks(dets, P_0, track_counter)
            
    # kill death track
    mask_alive = tracks[:, -2] < 3
    tracks = tracks[mask_alive]
    tracks_P = tracks_P[mask_alive]

    return tracks, tracks_P, track_counter


def spawn_new_tracks(dets: np.ndarray, P_0: np.ndarray, track_counter: int) -> Tuple[np.ndarray, np.ndarray, int]:
    tracks = np.pad(dets, pad_width=[(0, 0), (0, 7)])  # 4 unobservable state, 3 info (num_hit, num_miss, id)
    tracks[:, -3] = 1  # init num_hit
    tracks[:, -1] = track_counter + np.arange(dets.shape[0])  # track_id
    track_counter += dets.shape[0]
    tracks_P = np.tile(P_0[np.newaxis, ...], (dets.shape[0], 1, 1))  # (M, 11, 11)
    return tracks, tracks_P, track_counter


def build_cost_matrix(tracks_obs: np.ndarray, tracks_S: np.ndarray, dets: np.ndarray) -> np.ndarray:
    """
    Args:
        tracks_obs: (N, 7)
        tracks_S: (N, 7, 7) - covariance of obs
        dets: (M, 7)
    """
    diff = dets[:, np.newaxis, :] - tracks_obs[np.newaxis, :, :]  # (M, N, 7)
    cost = np.einsum('mnoc, mnck, mnkp -> mnop',
                      diff[:, :, np.newaxis, :], 
                      np.linalg.inv(tracks_S)[np.newaxis], 
                      diff[:, :, :, np.newaxis])[:, :, 0, 0]
    return cost


def greedy_match(distance_matrix: np.ndarray) -> np.ndarray:
    '''
    Find the one-to-one matching using greedy allgorithm choosing small distance

    Args:
        distance_matrix: (num_detections, num_tracks)

    Returns:
        matched_indices: (num_matches, 2) - (det_idx, track_idx)
    '''
    matched_indices = []

    num_detections, num_tracks = distance_matrix.shape
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)  # (num_det * num_tracks, 2)
    detection_id_matches_to_tracking_id = -np.ones(num_detections, dtype=int)
    tracking_id_matches_to_detection_id = -np.ones(num_tracks, dtype=int)
    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            matched_indices.append([detection_id, tracking_id])

    matched_indices = np.array(matched_indices)
    return matched_indices


def wrapper_matching(cost_matrix: np.ndarray, cost_threshold: float, square_cost_threshold: bool = True) -> Tuple[np.ndarray]:
    """
    Find matches, unmatched detections, unmatched tracks

    Args:
        cost_matrix: (num_dets, num_tracks)
        cost_threshold: to filter out bad match

    Returns:
        matches: (num_matches, 2) - (det_idx, track_idx)
        unmatched_dets_idx: (num_unmatched_dets)
        unmatched_tracks_idx: (num_unmatched_tracks)        
    """

    def find_unmatched_idx(num_obj: int, matched_obj_idx: np.ndarray) -> np.ndarray:
        assert matched_obj_idx.shape == (matched_obj_idx.shape[0], ), f"expect matched_obj_idx has 1 dim, get {matched_obj_idx.shape}"
        all_idx = np.arange(num_obj)

        if matched_obj_idx.shape[0] == 0:
            return all_idx

        mask_matched = np.any(all_idx[np.newaxis, :] == matched_obj_idx.reshape(-1, 1), axis=0)
        unmatched_idx = all_idx[np.logical_not(mask_matched)]
        return unmatched_idx

    matched_indices = greedy_match(cost_matrix)

    # find good matches
    if square_cost_threshold:
        cost_threshold = np.square(cost_threshold)

    matched_cost = cost_matrix[matched_indices[:, 0], matched_indices[:, 1]]  # (num_matches)
    mask_good_matches = matched_cost < cost_threshold
    matched_indices = matched_indices[mask_good_matches]  # (num_good_matches, 2) - (det_idx, track_idx)

    # find unmatched dets
    unmatched_dets_idx = find_unmatched_idx(cost_matrix.shape[0], matched_indices[:, 0])

    # find unmatched tracks
    unmatched_tracks_idx = find_unmatched_idx(cost_matrix.shape[1], matched_indices[:, 1])

    return matched_indices, unmatched_dets_idx, unmatched_tracks_idx
