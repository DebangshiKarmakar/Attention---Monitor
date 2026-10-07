import time


class AttentionState:
    """Simple state container for one tracked face."""

    def __init__(self, person_id):
        self.person_id = person_id
        self.state = "NO FACE"
        self.last_seen = time.time()
        self.total_frames = 0
        self.attentive_frames = 0
        self.distracted_frames = 0
        self.drowsy_frames = 0
        self.no_face_frames = 0

    def _record_state(self, state):
        self.state = state
        self.last_seen = time.time()
        self.total_frames += 1
        if state == "ATTENTIVE":
            self.attentive_frames += 1
        elif state == "DISTRACTED":
            self.distracted_frames += 1
        elif state == "DROWSY":
            self.drowsy_frames += 1
        elif state == "NO FACE":
            self.no_face_frames += 1

    def update(self, ear, mar, yaw, pitch):
        """Evaluate a face observation and return the active attention state."""
        if yaw is not None and abs(yaw) > 35:
            state = "DISTRACTED"
        elif pitch is not None and abs(pitch) > 25:
            state = "DISTRACTED"
        elif ear < 0.22:
            state = "DROWSY"
        elif mar > 0.30:
            state = "DISTRACTED"
        else:
            state = "ATTENTIVE"

        self._record_state(state)
        return state

    def mark_no_face(self):
        self._record_state("NO FACE")
        return self.state

    def attentiveness_percentage(self):
        if self.total_frames == 0:
            return 0
        return round((self.attentive_frames / self.total_frames) * 100)

    def attentieness_perentage(self):
        return self.attentiveness_percentage()

    def attentiness_perentage(self):
        return self.attentiveness_percentage()


class AttentionRegistry:
    """Track all currently monitored participants."""

    def __init__(self):
        self.people = {}

    def get_or_create(self, person_id):
        if person_id not in self.people:
            self.people[person_id] = AttentionState(person_id)
        return self.people[person_id]

    def prune_stale(self, max_age_seconds=2.0):
        now = time.time()
        for person_id in list(self.people.keys()):
            if now - self.people[person_id].last_seen > max_age_seconds:
                del self.people[person_id]

    def summary(self):
        counts = {"ATTENTIVE": 0, "DISTRACTED": 0, "DROWSY": 0, "NO FACE": 0}
        for person in self.people.values():
            counts[person.state] = counts.get(person.state, 0) + 1
        return counts
