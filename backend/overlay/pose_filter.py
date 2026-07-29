"""Temporal smoothing for the overlay pipeline.

Applies a per-channel `1€ Filter`_ to the raw spatial transform
(position, angle, scale, brightness) every frame so the overlay
feels stable at rest yet responds instantly to fast movements.

The ``SpatialSmoother`` wraps multiple filter instances — one pair
(x, y) per tracked point, plus one each for angle, scale, and
brightness.  Point filters are created lazily on first use and
wiped on ``reset()`` (important when switching anchor types, since
the old filter state is meaningless for a different anchor shape).

Angle values are unwrapped before filtering to prevent interpolation
spikes when crossing the ±180° boundary (e.g. 179° → -179° would
otherwise cause a massive jump).

.. _1€ Filter:
   Casiez, Roussel, Vogel. 2012. "1€ Filter: A Simple Speed-based
   Low-pass Filter for Noisy Input in Interactive Systems."
"""

import math
import time


class OneEuroFilter:
    """1€ Filter — eliminates jitter at rest, responds instantly to fast motion.

    Reference: Casiez, Roussel, Vogel. 2012.
    "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input
    in Interactive Systems."

    Parameters
    ----------
    min_cutoff : float
        Minimum cutoff frequency (Hz).  Lower = smoother at rest, but
        more latency.
    beta : float
        Speed coefficient.  Higher = less smoothing during fast motion
        (more responsive).
    d_cutoff : float
        Cutoff frequency for the derivative filter.
    """
    def __init__(self, min_cutoff=1.5, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t=None):
        if t is None:
            t = time.time()
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-5)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        dx_hat = self.dx_prev + self._alpha(self.d_cutoff, dt) * (dx - self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self.x_prev + self._alpha(cutoff, dt) * (x - self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class SpatialSmoother:
    """Smooth all overlay channels: points (x, y each), angle, scale, brightness.

    ``transform_dict`` carries ``"points": [{"x", "y"}, ...]`` because
    dual anchors (``both_shoulders``, ``both_wrists``) report two
    independent points rather than a midpoint.  Each point gets its
    OWN pair of x/y filters, keyed by index (0 = e.g. left shoulder,
    1 = right shoulder), so the two points are smoothed independently
    rather than one filter fighting to track two different signals.

    Point filters are created lazily the first time a given index is
    seen, and are wiped on ``reset()`` along with everything else —
    this matters because switching from a 1-point anchor to a 2-point
    anchor (or vice versa) means the old per-index filter state is
    meaningless for the new anchor's points.
    """
    def __init__(self):
        self._point_filters = {}   # index -> {"x": OneEuroFilter, "y": OneEuroFilter}
        self.filters = {
            "angle":      OneEuroFilter(min_cutoff=1.0,  beta=0.05),
            "scale":      OneEuroFilter(min_cutoff=0.5,  beta=0.01),
            "brightness": OneEuroFilter(min_cutoff=0.1,  beta=0.001),
        }

    def _filters_for_point(self, idx):
        if idx not in self._point_filters:
            self._point_filters[idx] = {
                "x": OneEuroFilter(min_cutoff=1.5, beta=0.05),
                "y": OneEuroFilter(min_cutoff=1.5, beta=0.05),
            }
        return self._point_filters[idx]

    def smooth(self, transform_dict: dict) -> dict:
        """Apply 1€ filtering to all points + angle/scale (+ optional brightness).

        Parameters
        ----------
        transform_dict : dict
            Must contain: ``points`` (list of 1-2 ``{"x","y"}`` dicts),
            ``angle``, ``scale``.  Optionally contains: ``brightness``.

        Returns
        -------
        dict
            Same shape as input, with every numeric value smoothed.
        """
        t = time.time()

        smoothed_points = []
        for idx, pt in enumerate(transform_dict["points"]):
            pf = self._filters_for_point(idx)
            smoothed_points.append({
                "x": round(pf["x"].filter(pt["x"], t), 1),
                "y": round(pf["y"].filter(pt["y"], t), 1),
            })

        # Unwrap angle to prevent interpolation spikes (e.g. 179° → -179°
        # would cause the filter to swing through ~358° of travel).
        angle_val = transform_dict["angle"]
        angle_filter = self.filters["angle"]
        if angle_filter.x_prev is not None:
            diff = angle_val - angle_filter.x_prev
            while diff > 180: angle_val -= 360; diff = angle_val - angle_filter.x_prev
            while diff < -180: angle_val += 360; diff = angle_val - angle_filter.x_prev

        sm_angle = angle_filter.filter(angle_val, t)
        # Wrap back to [-180, 180]
        while sm_angle > 180: sm_angle -= 360
        while sm_angle < -180: sm_angle += 360

        result = {
            "points": smoothed_points,
            "angle":  round(sm_angle, 1),
            "scale":  round(self.filters["scale"].filter(transform_dict["scale"], t), 2),
        }

        if "brightness" in transform_dict:
            result["brightness"] = round(
                self.filters["brightness"].filter(transform_dict["brightness"], t), 2
            )

        return result

    def reset(self):
        """Reset all filter channels, including per-point filters."""
        self._point_filters = {}
        for f in self.filters.values():
            f.reset()