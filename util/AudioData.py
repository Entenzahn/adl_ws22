class SongData:
    def __init__(self):
        self.sid = 0
        self.sr = 0
        self.key = ""
        self.samples = []  # The original song info
        self.samples_30sec = []  # Random 30 second window

        self.qt_spec = []  # Constant Q-transform
        self.qt_spec_base = []  # Adapted constant Q-transform
        self.qt_spec_30sec = []  # Constant Q-transform of 30 sec
        self.qt_spec_resized = []  # Resized to uniform dimensions
        self.qt_spec_base_resized = []
        self.segment_specs = []  # Randomly stitched together spectograph segments


class SegmentSpectrumData:
    def __init__(self, spec, sid):
        self.spec = spec
        self.sid = sid
        self.segment_id = 0