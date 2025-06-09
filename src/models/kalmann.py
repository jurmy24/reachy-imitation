import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanFilter3D:
    def __init__(self, threshold=0.5):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        dt = (
            1 / 15.0
        )  # Time step # 15 represents the approximate frame rate - Verify this

        # State transition matrix
        self.kf.F = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )

        # Measurement function
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R *= 5  # Measurement noise
        self.kf.P *= 100.0  # Initial covariance
        self.kf.Q *= 1e-2  # Process noise

        self.kf.x[:6] = np.zeros((6, 1))  # Initial state
        self.last_output = np.zeros(3)
        self.threshold = threshold  # meters

    def update(self, measurement):
        measurement = np.array(measurement)

        self.kf.predict()
        self.kf.update(measurement)
        self.last_output = self.kf.x[:3].reshape(-1)
        return self.last_output
