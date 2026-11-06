"""Conservative pixel-only tracking of RGB minimap enemy observations.

Coordinates use the recognizer's u=x/(width-1), v=y/(height-1) convention.
Association requires an isolated track/detection pair in a bipartite spatial
candidate graph. Prediction uses elapsed time since the last actual observation;
both displacement from that observation and residual from the prediction must
fit max_speed_px_s * dt + position_slack_px. association_margin_px widens these
gates only to detect ambiguity, never to accept an out-of-gate association.
Overlapping candidates are deliberately not resolved by nearest-neighbor choice.

Known species conflicts make even an isolated pair ambiguous, keeping the old
species and observation. Unknown species never breaks spatial association and
may be replaced by a subsequent known species. This is conservative pixel
continuity, not a claim about real ship identity or firing authorization.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite real number')
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f'{name} must be a finite real number') from None
    if not math.isfinite(number):
        raise ValueError(f'{name} must be a finite real number')
    return number


@dataclass
class _Detection:
    u: float
    v: float
    x: float
    y: float
    species: str


@dataclass
class _Track:
    track_id: int
    u: float
    v: float
    x: float
    y: float
    last_seen_s: float
    species: str
    hits: int = 1
    consecutive_hits: int = 1
    state: str = 'tentative'
    observed: bool = True
    velocity: tuple | None = None


class EnemyTracker:
    """Track live pixel observations; storage contains no expired-track history.

    update accepts a sequence of mappings with normalized u/v and optional string
    species (default 'unknown'). Other perception metadata is ignored. Invalid
    input raises ValueError before any tracker state changes. Time must increase
    strictly for update, including after expire; expire also accepts equal time.
    A later expire represents no fresh observation and makes remaining tracks
    lost. Equal-time expire preserves the current observation flags.

    Output is a fresh JSON-ready list ordered by track_id. age_s is elapsed time
    since last_seen_s (observation staleness), hits counts unique observations, and
    last_seen_s/u/v always describe the last actual observation. usable requires
    confirmed AND observed. Confirmation needs confirm_hits consecutive unique
    observations, restarting after a missing or ambiguous frame. Velocity is a
    measured pixel displacement divided by actual observation dt; prediction
    bounds its magnitude by max_speed_px_s to avoid extrapolating detection noise.
    reset clears observations and clock, but never reuses an allocated ID.
    """

    def __init__(self, width=720, height=723, max_speed_px_s=8.,
                 position_slack_px=6., association_margin_px=3.,
                 max_gap_s=2.5, confirm_hits=2):
        for name, value in (('width', width), ('height', height)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f'{name} must be an integer >= 2')
            _finite_number(value, name)
        if (isinstance(confirm_hits, bool) or not isinstance(confirm_hits, int)
                or confirm_hits < 1):
            raise ValueError('confirm_hits must be an integer >= 1')
        self.width = width
        self.height = height
        for name, value in (('max_speed_px_s', max_speed_px_s),
                            ('position_slack_px', position_slack_px),
                            ('association_margin_px', association_margin_px),
                            ('max_gap_s', max_gap_s)):
            number = _finite_number(value, name)
            if number < 0:
                raise ValueError(f'{name} must be nonnegative')
            setattr(self, name, number)
        diagonal = math.hypot(width - 1, height - 1)
        travel = self.max_speed_px_s * self.max_gap_s
        if (not math.isfinite(diagonal)
                or not math.isfinite(diagonal + travel)
                or not math.isfinite(travel + self.position_slack_px
                                     + self.association_margin_px)):
            raise ValueError('configured pixel geometry must remain finite')
        self.confirm_hits = confirm_hits
        self._tracks = []
        self._clock_s = None
        self._next_id = 1

    def reset(self):
        self._tracks.clear()
        self._clock_s = None

    def _validate_clock(self, clock_s, *, strict):
        clock_s = _finite_number(clock_s, 'clock_s')
        if self._clock_s is not None:
            if clock_s < self._clock_s or (strict and clock_s == self._clock_s):
                rule = 'strictly increasing' if strict else 'nondecreasing'
                raise ValueError(f'clock_s must be {rule}; reset for a new sequence')
            if not math.isfinite(clock_s - self._clock_s):
                raise ValueError('elapsed clock_s must be finite')
        for track in self._tracks:
            if not math.isfinite(clock_s - track.last_seen_s):
                raise ValueError('observation age must be finite')
        return clock_s

    def _validate_detections(self, detections):
        if (not isinstance(detections, Sequence)
                or isinstance(detections, (str, bytes, bytearray))):
            raise ValueError('detections must be a sequence of mappings')
        result = []
        for index, detection in enumerate(detections):
            if not isinstance(detection, Mapping):
                raise ValueError(f'detections[{index}] must be a mapping')
            if 'u' not in detection or 'v' not in detection:
                raise ValueError(f'detections[{index}] requires u and v')
            u = _finite_number(detection['u'], f'detections[{index}].u')
            v = _finite_number(detection['v'], f'detections[{index}].v')
            if not 0 <= u <= 1 or not 0 <= v <= 1:
                raise ValueError(f'detections[{index}] u/v must be in [0, 1]')
            species = detection.get('species', 'unknown')
            if not isinstance(species, str) or not species.strip():
                raise ValueError(f'detections[{index}].species must be a nonempty string')
            result.append(_Detection(u, v, u * (self.width - 1),
                                     v * (self.height - 1), species))
        return result

    def _prune(self, clock_s):
        self._tracks = [track for track in self._tracks
                        if clock_s - track.last_seen_s <= self.max_gap_s]

    def _snapshot(self, clock_s):
        return [dict(track_id=track.track_id, state=track.state,
                     observed=track.observed,
                     usable=track.state == 'confirmed' and track.observed,
                     u=track.u, v=track.v,
                     velocity_px_s=(dict(x=track.velocity[0], y=track.velocity[1])
                                    if track.velocity is not None else None),
                     last_seen_s=track.last_seen_s,
                     age_s=clock_s - track.last_seen_s, hits=track.hits,
                     species=track.species)
                for track in self._tracks]

    def expire(self, clock_s):
        clock_s = self._validate_clock(clock_s, strict=False)
        self._prune(clock_s)
        if self._clock_s is not None and clock_s > self._clock_s:
            for track in self._tracks:
                track.state = 'lost'
                track.observed = False
                track.consecutive_hits = 0
        self._clock_s = clock_s
        return self._snapshot(clock_s)

    def _observe(self, track, detection, clock_s, velocity):
        track.velocity = velocity
        track.u, track.v = detection.u, detection.v
        track.x, track.y = detection.x, detection.y
        track.last_seen_s = clock_s
        track.hits += 1
        track.consecutive_hits += 1
        track.observed = True
        if track.species == 'unknown':
            track.species = detection.species
        track.state = ('confirmed' if track.consecutive_hits >= self.confirm_hits
                       else 'tentative')

    def update(self, detections, clock_s):
        clock_s = self._validate_clock(clock_s, strict=True)
        detections = self._validate_detections(detections)
        tracks = [track for track in self._tracks
                  if clock_s - track.last_seen_s <= self.max_gap_s]
        track_edges = [[] for _ in tracks]
        detection_edges = [[] for _ in detections]
        within_gate = set()
        for ti, track in enumerate(tracks):
            dt = clock_s - track.last_seen_s
            radius = self.max_speed_px_s * dt + self.position_slack_px
            vx, vy = track.velocity or (0., 0.)
            speed = math.hypot(vx, vy)
            if speed > self.max_speed_px_s:
                scale = self.max_speed_px_s / speed
                vx, vy = vx * scale, vy * scale
            predicted_x, predicted_y = track.x + vx * dt, track.y + vy * dt
            for di, detection in enumerate(detections):
                distance = max(math.hypot(detection.x - track.x, detection.y - track.y),
                               math.hypot(detection.x - predicted_x,
                                          detection.y - predicted_y))
                if distance <= radius + self.association_margin_px:
                    track_edges[ti].append(di)
                    detection_edges[di].append(ti)
                    if distance <= radius:
                        within_gate.add((ti, di))

        visited = set()
        associations = {}
        states = {}
        for start, track in enumerate(tracks):
            if start in visited:
                continue
            if not track_edges[start]:
                states[start] = 'lost'
                continue
            component_tracks = {start}
            component_detections = set()
            pending = [start]
            visited.add(start)
            while pending:
                ti = pending.pop()
                for di in track_edges[ti]:
                    if di in component_detections:
                        continue
                    component_detections.add(di)
                    for neighbor in detection_edges[di]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            component_tracks.add(neighbor)
                            pending.append(neighbor)
            if len(component_tracks) == len(component_detections) == 1:
                ti = start
                di = next(iter(component_detections))
                detection = detections[di]
                conflict = (track.species != 'unknown' and detection.species != 'unknown'
                            and track.species != detection.species)
                if (ti, di) in within_gate and not conflict:
                    dt = clock_s - track.last_seen_s
                    velocity = ((detection.x - track.x) / dt,
                                (detection.y - track.y) / dt)
                    if not all(math.isfinite(value) for value in velocity):
                        raise ValueError('observation dt is too small for finite velocity')
                    associations[ti] = (detection, velocity)
                    continue
            for ti in component_tracks:
                states[ti] = 'ambiguous'

        self._tracks = tracks
        for ti, track in enumerate(tracks):
            if ti in associations:
                detection, velocity = associations[ti]
                self._observe(track, detection, clock_s, velocity)
            else:
                track.state = states[ti]
                track.observed = False
                track.consecutive_hits = 0

        for di, detection in enumerate(detections):
            if detection_edges[di]:
                continue
            track = _Track(self._next_id, detection.u, detection.v,
                           detection.x, detection.y, clock_s, detection.species)
            if self.confirm_hits == 1:
                track.state = 'confirmed'
            self._tracks.append(track)
            self._next_id += 1
        self._clock_s = clock_s
        return self._snapshot(clock_s)
