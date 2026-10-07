import math


class CentroidTracker:
    """Track detected face boxes by centroid proximity."""

    def __init__(self, max_disappeared=10, max_distance=50):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}

    def _center(self, box):
        x, y, w, h = box
        return (x + w / 2.0, y + h / 2.0)

    def update(self, rects):
        if not rects:
            for object_id in list(self.objects.keys()):
                self.disappeared[object_id] = self.disappeared.get(object_id, 0) + 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]
            return {}

        if not self.objects:
            for rect in rects:
                self.objects[self.next_id] = rect
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            return dict(self.objects)

        matched_rect_indices = set()
        for object_id, prev_rect in list(self.objects.items()):
            best_distance = float("inf")
            best_index = None
            prev_center = self._center(prev_rect)

            for index, rect in enumerate(rects):
                if index in matched_rect_indices:
                    continue
                dist = math.hypot(prev_center[0] - self._center(rect)[0], prev_center[1] - self._center(rect)[1])
                if dist < best_distance:
                    best_distance = dist
                    best_index = index

            if best_index is not None and best_distance <= self.max_distance:
                self.objects[object_id] = rects[best_index]
                self.disappeared[object_id] = 0
                matched_rect_indices.add(best_index)
            else:
                self.disappeared[object_id] = self.disappeared.get(object_id, 0) + 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.disappeared[object_id]

        for index, rect in enumerate(rects):
            if index in matched_rect_indices:
                continue
            self.objects[self.next_id] = rect
            self.disappeared[self.next_id] = 0
            self.next_id += 1

        return dict(self.objects)
