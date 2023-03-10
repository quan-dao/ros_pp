import numpy as np
from typing import Tuple


'''
State: (x, y, z, yaw, l, w, h, dx, dy, dz, dyaw)

track = [state || num_hit, num_miss, id]
'''


TRANSITION_MATRIX = np.array([[1,0,0,0,0,0,0,1,0,0,0],      # state transition matrix
                            [0,1,0,0,0,0,0,0,1,0,0],
                            [0,0,1,0,0,0,0,0,0,1,0],
                            [0,0,0,1,0,0,0,0,0,0,1],  
                            [0,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0],
                            [0,0,0,0,0,0,0,1,0,0,0],
                            [0,0,0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,0,0,0,0,1]])     

MEASUREMENT_MATRIX = np.array([[1,0,0,0,0,0,0,0,0,0,0],      # measurement function,
                            [0,1,0,0,0,0,0,0,0,0,0],
                            [0,0,1,0,0,0,0,0,0,0,0],
                            [0,0,0,1,0,0,0,0,0,0,0],
                            [0,0,0,0,1,0,0,0,0,0,0],
                            [0,0,0,0,0,1,0,0,0,0,0],
                            [0,0,0,0,0,0,1,0,0,0,0]])

MATRICES_P = {
'bicycle':    [0.05390982, 0.05039431, 0.01863044, 1.29464435, 0.02713823, 0.01169572, 0.01295084, 0.04560422, 0.04097244, 0.01725477, 1.21635902],
'bus':        [0.17546469, 0.13818929, 0.05947248, 0.1979503 , 0.78867322, 0.05507407, 0.06684149, 0.13263319, 0.11508148, 0.05033665, 0.22529652],
'car':        [0.08900372, 0.09412005, 0.03265469, 1.00535696, 0.10912802, 0.02359175, 0.02455134, 0.08120681, 0.08224643, 0.02266425, 0.99492726],
'motorcycle': [0.04052819, 0.0398904 , 0.01511711, 1.06442726, 0.03291016, 0.00957574, 0.0111605 , 0.0437039 , 0.04327734, 0.01465631, 1.30414345],
'pedestrian': [0.03855275, 0.0377111 , 0.02482115, 2.0751833 , 0.02286483, 0.0136347 , 0.0203149 , 0.04237008, 0.04092393, 0.01482923, 2.0059979 ],
'trailer':    [0.23228021, 0.22229261, 0.07006275, 1.05163481, 1.37451601, 0.06354783, 0.10500918, 0.2138643 , 0.19625241, 0.05231335, 0.97082174],
'truck':      [0.14862173, 0.1444596 , 0.05417157, 0.73122169, 0.69387238, 0.05484365, 0.07748085, 0.10683797, 0.10248689, 0.0378078 , 0.76188901]
}

MATRICES_Q = {
    'bicycle': [1.98881347e-02, 1.36552276e-02, 5.10175742e-03, 1.33430252e-01, 0, 0, 0, 1.98881347e-02, 1.36552276e-02, 5.10175742e-03, 
                1.33430252e-01],
    'bus': [1.17729925e-01, 8.84659079e-02, 1.17616440e-02, 2.09050032e-01, 0, 0, 0, 1.17729925e-01, 8.84659079e-02, 1.17616440e-02, 2.09050032e-01],
    'car': [1.58918523e-01, 1.24935318e-01, 5.35573165e-03, 9.22800791e-02, 0, 0, 0, 1.58918523e-01, 1.24935318e-01, 5.35573165e-03, 9.22800791e-02],
    'motorcycle': [3.23647590e-02, 3.86650974e-02, 5.47421635e-03, 2.34967407e-01, 0, 0, 0, 3.23647590e-02, 3.86650974e-02, 5.47421635e-03, 
                   2.34967407e-01],
    'pedestrian': [3.34814566e-02, 2.47354921e-02, 5.94592529e-03, 4.24962535e-01, 0, 0, 0, 3.34814566e-02, 2.47354921e-02, 5.94592529e-03, 
                   4.24962535e-01],
    'trailer': [4.19985099e-02, 3.68661552e-02, 1.19415050e-02, 5.63166240e-02, 0, 0, 0, 4.19985099e-02, 3.68661552e-02, 1.19415050e-02, 
                5.63166240e-02],
    'truck': [9.45275998e-02, 9.45620374e-02, 8.38061721e-03, 1.41680460e-01, 0, 0, 0, 9.45275998e-02, 9.45620374e-02, 8.38061721e-03, 
              1.41680460e-01]
}

MATRICES_R = {
    'bicycle':    [0.05390982, 0.05039431, 0.01863044, 1.29464435, 0.02713823, 0.01169572, 0.01295084],
    'bus':        [0.17546469, 0.13818929, 0.05947248, 0.1979503 , 0.78867322, 0.05507407, 0.06684149],
    'car':        [0.08900372, 0.09412005, 0.03265469, 1.00535696, 0.10912802, 0.02359175, 0.02455134],
    'motorcycle': [0.04052819, 0.0398904 , 0.01511711, 1.06442726, 0.03291016, 0.00957574, 0.0111605 ],
    'pedestrian': [0.03855275, 0.0377111 , 0.02482115, 2.0751833 , 0.02286483, 0.0136347 , 0.0203149 ],
    'trailer':    [0.23228021, 0.22229261, 0.07006275, 1.05163481, 1.37451601, 0.06354783, 0.10500918],
    'truck':      [0.14862173, 0.1444596 , 0.05417157, 0.73122169, 0.69387238, 0.05484365, 0.07748085]
}


def batch_kf_predict(tracks: np.ndarray, tracks_P: np.ndarray, F: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray]:
    """
    Prediction step of a linear KF
        x_new = F @ x                   (1)
        P_new = F @ P @ F.T + Q         (2)


    Args:
        tracks: (N, 11 + 3) - state (11), info (3)
        tracks_P: (N, 11, 11) - tracks' covariance matrix
        F: (11, 11) - transition matrix
        Q: (11, 11) - noise of transition matrix

    Returns:
        tracks: (N, 11 + 3) with state changed according to (1)
        tracks_P: (N, 11, 11) according to (2)
    """
    tracks[:, :11] = tracks[:, :11] @ F.T
    tracks_P = np.einsum('ij, bjk, kh -> bih', F, tracks_P, F.T) + Q
    return tracks, tracks_P


def batch_kf_predict_measurement(tracks: np.ndarray, tracks_P: np.ndarray, H: np.ndarray, R: np.ndarray, num_observable_states: int = 7) -> np.ndarray:
    """
    Computed predicted measurement according to linear measurement model
        pred_o = H @ x   (x: state vector)
    """
    obs = tracks[:, :num_observable_states]
    S = np.einsum('ij, bjk, kh -> bih', H, tracks_P, H.T) + R
    return obs, S


def batch_kf_update(tracks: np.ndarray, tracks_P: np.ndarray, observations: np.ndarray, H: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray]:
    """
    Update step of a linear KF
        K = P @ H.T @ inv(S)                (1)
        x_new = x + K @ (obs - H @ x)       (2)
        P_new = (eye - K @ H) @ P           (3)

    Args:
        tracks: (N, 11 + 3) - state (11), info (4)
        tracks_P: (N, 11, 11) - tracks' covariance matrix
        observations: (N, 7) - tracks' match observation (1-1 mathcing)
        H: (11, 11 + 3) - measurement matrix
        S: (15, 15) - noise of measurement matrix

    Returns:
        tracks: (N, 11 + 3) according to (2)
        tracks_P: (N, 11, 11) according to (3)
    """
    obs, S = batch_kf_predict_measurement(tracks, tracks_P, H, R)  # S: (N, 7, 7)
    K = np.einsum('bii, ik, bkk -> bik', tracks_P, H.T, np.linalg.inv(S))  # (N, 11, 7)
    innov = observations - obs  # (N, 7)
    tracks[:, :11] = tracks[:, :11] + np.einsum('bih, bh -> bi', K, innov)
    tracks_P = tracks_P - np.einsum('bij, bjk, kh -> bih', tracks_P, K, H)
    return tracks, tracks_P
