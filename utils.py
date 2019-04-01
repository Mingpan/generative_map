# Fitler related helper functions

import math
import numpy as np


MACHINE_EPS = 1e-9
SINGULAR_FIX = True


def sigmoid(x):
    """Numerically stable sigmoid function"""
    y = np.zeros_like(x)
    y[x >= 0] = 1. / (1. + np.exp(- x[x >= 0]))
    z = np.exp(x[x < 0])
    y[x < 0] = z / (1. + z)
    return y


def predict(x, Sigma, u, A, B, Q, step_sim):
    """Prediction step of EKF"""
    x1 = step_sim(x, u)
    Sigma1 = A.dot(Sigma).dot(A.T) + Q
    return x1, Sigma1


def correct(x, Sigma, o, R, H, observe_sim):
    """Correction step of EKF"""
    S = H.dot(Sigma).dot(H.T) + R
    y = o - observe_sim(x)
    if SINGULAR_FIX:
        S += np.eye(S.shape[0]) * MACHINE_EPS
    K = np.linalg.solve(S.T, H.dot(Sigma.T)).T
    x1 = x + K.dot(y)
    Sigma1 = Sigma - K.dot(H).dot(Sigma)
    return x1, Sigma1


def finite_diff(simulate, x, e=1e-1, dim_out=None):
    dim = x.shape[0]
    if dim_out is None:
        dim_out = dim
    res = np.zeros((dim_out, 0))
    for i in range(dim):
        x1 = x.copy()
        x1[i] += e
        x1_inc = simulate(x1).copy()
        x1[i] -= (2 * e)
        x1_dec = simulate(x1)
        res = np.hstack((res, (x1_inc - x1_dec) / (2 * e)))
    return res


### Test


def quaternion_to_euler_angle(q):
    assert q.shape == (4, )
    w, x, y, z = q
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return np.array([X, Y, Z])


def rotation_matrix_to_euler_angle(R):
    X = math.atan2(R[1, 2], R[2, 2])

    s1 = math.sin(X)
    c1 = math.cos(X)
    c2 = math.sqrt(R[0, 0]*R[0, 0] + R[0, 1]*R[0, 1])
    
    Y = math.atan2(-1.0*R[0, 2], c2)
    Z = math.atan2(s1 * R[2, 0] - c1 * R[1, 0], c1 * R[1, 1] - s1 * R[2, 1])

    return np.array([X, Y, Z])
 
    
def rotation_matrix_to_quaternion(R):
    w = np.sqrt( max(0, 1 + R[0, 0] + R[1, 1] + R[2, 2]) ) / 2
    x = - np.sign(R[2,1] - R[1,2]) * np.sqrt( max(0, 1 + R[0,0] - R[1,1] - R[2,2]) ) / 2
    y = - np.sign(R[0,2] - R[2,0]) * np.sqrt( max(0, 1 - R[0,0] + R[1,1] - R[2,2]) ) / 2
    z = - np.sign(R[1,0] - R[0,1]) * np.sqrt( max(0, 1 - R[0,0] - R[1,1] + R[2,2]) ) / 2

    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q):
    R = [[q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
         [2*(q[1]*q[2] + q[0]*q[3]), q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2, 2*(q[2]*q[3] - q[0]*q[1])],
         [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2]]

    return np.array(R)


def rotation_matrix_to_log_quaternion(R):
    u = math.acos(0.5 * math.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1.0))
    
    v = np.array([R[1, 2] - R[2, 1],
                  R[2, 0] - R[0, 2],
                  R[0, 1] - R[1, 0]])

    log_quat = v * (u / np.sqrt(np.sum(v*v)))
    
    return log_quat


## Euler angles


def wrap_angles_to_state(angles):
    assert angles.shape == (3, )
    theta, phi, psi = angles
    res = np.array([np.cos(theta), np.sin(theta),
                    np.cos(phi), np.sin(phi),
                    np.cos(psi), np.sin(psi)])
    return res


def update_angles(ang_state, dangles):
    c1, s1, c2, s2, c3, s3 = ang_state
    dc1, ds1, dc2, ds2, dc3, ds3 = wrap_angles_to_state(dangles)
    nc1 = c1 * dc1 - s1 * ds1
    ns1 = s1 * dc1 + c1 * ds1
    nc2 = c2 * dc2 - s2 * ds2
    ns2 = s2 * dc2 + c2 * ds2
    nc3 = c3 * dc3 - s3 * ds3
    ns3 = s3 * dc3 + c3 * ds3
    return np.array([nc1, ns1, nc2, ns2, nc3, ns3])


def quaternion_to_state(q):
    assert q.shape == (4, )
    angles = quaternion_to_euler_angle(q)
    return wrap_angles_to_state(angles)


def pose_to_state(s):
    xyz = np.array(s[:3])
    angles = s[3:]
    angles = wrap_angles_to_state(np.array(angles))
    state = np.concatenate((xyz, angles))
    return state


## Quaternions ##

def quaternion_distance(q0, q1):
    assert q0.shape == q1.shape == (4,)
    return np.arccos(2 * (q0.dot(q1) ** 2) - 1)


def quaternion_multiply(q0, q1):
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])


def quaternion_inverse(q):
    n = np.sum(q * q)
    q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
    return q_inv / n


def log_quaternion(q):
    u, x, y, z = q
    l2norm2 = x ** 2 + y ** 2 + z ** 2
    if l2norm2 < MACHINE_EPS:
        return np.array([0, 0, 0])
    u = np.clip(u, -1., 1.)
    factor = np.arccos(u) / np.sqrt(l2norm2)
    return np.array([x, y, z]) * factor


def inv_log_quaternion(log_q):
    l2norm = np.sqrt(np.sum(np.square(log_q)))
    x, y, z = log_q * (np.sin(l2norm) / l2norm)
    return np.array([np.cos(l2norm), x, y, z])
