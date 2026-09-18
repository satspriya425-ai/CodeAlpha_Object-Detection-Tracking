import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    intersection = width * height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


class KalmanBoxTracker:
    """Kalman filter for tracking one object."""

    count = 0

    def __init__(self, bbox):
        self.bbox = bbox

        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        self.kf.P *= 10
        self.kf.R *= 1
        self.kf.Q *= 0.01

        self.kf.x[:4] = np.array(bbox).reshape(4, 1)

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

        self.time_since_update = 0
        self.hit_streak = 0

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1

        prediction = self.kf.x[:4].reshape(-1)
        return prediction

    def update(self, bbox):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(np.array(bbox))

    def get_state(self):
        return self.kf.x[:4].reshape(-1)


class Sort:
    """Simple SORT multi-object tracker."""

    def __init__(self, max_age=10, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers = []

    def update(self, detections):
        """
        detections:
            List of [x1, y1, x2, y2]
        """

        predictions = []

        for tracker in self.trackers:
            predictions.append(tracker.predict())

        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(self.trackers)))

        if predictions and detections:
            cost_matrix = np.zeros(
                (len(predictions), len(detections))
            )

            for i, prediction in enumerate(predictions):
                for j, detection in enumerate(detections):
                    cost_matrix[i, j] = -iou(
                        prediction,
                        detection
                    )

            rows, cols = linear_sum_assignment(cost_matrix)

            for row, col in zip(rows, cols):
                if -cost_matrix[row, col] >= self.iou_threshold:
                    matches.append((row, col))

            matched_tracker_ids = [m[0] for m in matches]
            matched_detection_ids = [m[1] for m in matches]

            unmatched_trackers = [
                i for i in range(len(self.trackers))
                if i not in matched_tracker_ids
            ]

            unmatched_detections = [
                i for i in range(len(detections))
                if i not in matched_detection_ids
            ]

        for tracker_index, detection_index in matches:
            self.trackers[tracker_index].update(
                detections[detection_index]
            )

        for detection_index in unmatched_detections:
            self.trackers.append(
                KalmanBoxTracker(
                    detections[detection_index]
                )
            )

        results = []

        for tracker in self.trackers[:]:
            if tracker.time_since_update <= self.max_age:
                box = tracker.get_state()

                if tracker.hit_streak >= self.min_hits:
                    results.append([
                        box[0],
                        box[1],
                        box[2],
                        box[3],
                        tracker.id
                    ])

        self.trackers = [
            tracker
            for tracker in self.trackers
            if tracker.time_since_update <= self.max_age
        ]

        return np.array(results)