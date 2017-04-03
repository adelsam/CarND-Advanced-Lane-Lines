class DebugInfo(object):
    def __init__(self, nonzerox, nonzeroy, margin, left_lane_inds, right_lane_inds):
        self.nonzerox = nonzerox
        self.nonzeroy = nonzeroy
        self.margin = margin
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds
