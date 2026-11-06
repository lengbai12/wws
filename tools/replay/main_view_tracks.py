"""Consecutive image-only ship and HUD-marker tracks; no world identity or fire authority.

A track survives only an isolated spatial association after measured camera
translation. Ordinary markers can also use a two-way nearest match separated
from every competing match by at least one glyph size. Missing, ambiguous or
uncompensated observations start new tracks. IDs are never reused after resets.
Velocities describe compensated screen motion only.
"""
import math


class MainViewTracker:
    """Attach evidence-only continuity to current-frame image observations.

    Ship nameplates retain the original class/team association. Ordinary
    torpedo and aircraft sprites additionally require a known marker subtype,
    the same class/type/team, and an isolated two-way spatial match. IDs and
    velocities describe screen evidence only; they never identify a hull or
    establish physical/world motion.
    """
    def __init__(self):
        self._next_id = 1
        self.reset()

    def reset(self):
        self._previous = []
        self._captured_ns = None

    @staticmethod
    def _box(observation):
        return observation.get('nameplate_bbox', observation.get('bbox'))

    @staticmethod
    def _is_danger(observation):
        return (observation.get('marker_family') == 'central_dangers'
                or observation.get('observation_semantics') == 'screen_hud_danger_indicator_only')

    @staticmethod
    def _marker_type(observation):
        if observation.get('class') == 'aircraft':
            return observation.get('aircraft_type')
        if observation.get('class') == 'torpedo':
            return observation.get('marker_type')
        return None

    @classmethod
    def _team(cls, observation):
        team = observation.get('team')
        # torpedo_perception's public result is intrinsically an enemy warning
        # (native_enemy_torpedo_warning_sprite), even though it has no team key.
        if (team is None and observation.get('class') == 'torpedo'
                and observation.get('evidence') == 'native_enemy_torpedo_warning_sprite'):
            return 'enemy'
        return team

    @classmethod
    def _association_key(cls, observation):
        kind = observation.get('class')
        if kind in ('aircraft', 'torpedo'):
            if cls._is_danger(observation):
                return None
            marker_type = cls._marker_type(observation)
            # Unknown subtype or side is insufficient evidence for identity continuity.
            if not isinstance(marker_type, str) or not marker_type or marker_type == 'unknown':
                return None
            team = cls._team(observation)
            if kind == 'aircraft' and team not in ('enemy', 'ally'):
                return None
            return kind, marker_type, team
        return kind, cls._team(observation)

    @staticmethod
    def _camera_shift(view, dt):
        shift = view.get('camera_shift')
        if dt is None or not math.isfinite(dt) or not 0 < dt <= 1:
            return None
        if not isinstance(shift, dict):
            return None
        try:
            values = tuple(float(shift[key]) for key in ('dx', 'dy', 'score'))
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in values) or values[2] < .35:
            return None
        return values

    def update(self, view, captured_ns):
        previous = self._previous
        dt = None if self._captured_ns is None else (captured_ns-self._captured_ns)/1e9
        shift = self._camera_shift(view, dt)
        current_observations = list(view.get('detections', [])) + list(view.get('markers', []))
        eligible = [observation for observation in current_observations
                    if self._box(observation) is not None and self._association_key(observation) is not None]
        edges = [[] for _ in eligible]
        reverse = [[] for _ in previous]
        # Without a usable camera shift the association may still be attempted
        # over a wider window: identity can survive, but the displacement then
        # mixes camera motion with target motion, so no velocity is claimed.
        compensated = shift is not None
        # The fallback still needs a plausible frame interval: a long gap is not
        # continuity, it is a new observation.
        if compensated or (dt is not None and math.isfinite(dt) and 0 < dt <= 1):
            radius = 60 if compensated else 90
            shift_x = shift[0] if compensated else 0.0
            shift_y = shift[1] if compensated else 0.0
            for i, observation in enumerate(eligible):
                x, y, w, h = self._box(observation)
                key = self._association_key(observation)
                for j, old in enumerate(previous):
                    if key != old['key']:
                        continue
                    ox, oy, ow, oh = old['bbox']
                    dx = x+w/2-(ox+ow/2)-shift_x
                    dy = y+h/2-(oy+oh/2)-shift_y
                    if math.hypot(dx, dy) < radius:
                        edges[i].append((j, dx, dy))
                        reverse[j].append(i)

        current = []
        eligible_index = {id(observation): i for i, observation in enumerate(eligible)}
        for observation in current_observations:
            box = self._box(observation)
            key = self._association_key(observation)
            if self._is_danger(observation):
                # The central warning is a HUD alert, never an aircraft entity.
                observation.update(track_id=None, track_state='observed',
                                   screen_velocity_px_s=None,
                                   velocity_semantics='screen_hud_danger_indicator_only; entity_motion_unknown')
                continue
            if box is None or key is None:
                observation.update(track_id=None, track_state='observed',
                                   screen_velocity_px_s=None,
                                   velocity_semantics='observed_screen_pixels_per_second; not_world_velocity')
                continue
            i = eligible_index[id(observation)]
            unique = len(edges[i]) == 1 and len(reverse[edges[i][0][0]]) == 1
            selected = edges[i][0] if unique else None
            if not unique and key[0] in ('aircraft', 'torpedo') and edges[i]:
                # Nearby salvo icons often share the 60px search radius. Keep
                # identity only when both directions have a clearly separated
                # nearest match; ties/crossings remain ambiguous.
                nearest = min(edges[i], key=lambda edge: math.hypot(edge[1], edge[2]))
                j, dx, dy = nearest
                distance = math.hypot(dx, dy)
                margin = max(box[2:]+previous[j]['bbox'][2:])
                forward_clear = all(k == j or math.hypot(ex, ey)-distance >= margin
                                    for k, ex, ey in edges[i])
                reverse_clear = all(k == i or all(
                    old != j or math.hypot(ex, ey)-distance >= margin
                    for old, ex, ey in edges[k]) for k in reverse[j])
                if forward_clear and reverse_clear:
                    selected = nearest
                    unique = True
            if unique:
                j, dx, dy = selected
                track_id = previous[j]['id']
                hits = previous[j]['hits']+1
                state = 'tracked'
                velocity = [dx/dt, dy/dt] if compensated else None
            else:
                track_id = self._next_id
                self._next_id += 1
                hits, velocity = 1, None
                state = 'ambiguous' if edges[i] else 'tentative'
            if velocity is None and state == 'tracked':
                semantics = ('uncompensated_nearest_neighbour_association; no screen velocity claim; '
                             'not_physical_or_world_velocity')
            elif key[0] in ('aircraft', 'torpedo'):
                semantics = ('camera_translation_compensated_screen_marker_pixels_per_second; '
                             'not_physical_or_world_velocity')
            else:
                semantics = ('camera_translation_compensated_healthbar_pixels_per_second; '
                             'not_hull_or_world_velocity')
            observation.update(track_id=track_id, track_state=state,
                               track_hits=hits, screen_velocity_px_s=velocity,
                               velocity_semantics=semantics)
            current.append(dict(id=track_id, hits=hits, key=key, bbox=list(box)))
        self._previous, self._captured_ns = current, captured_ns
        return view

    @staticmethod
    def bind_lock(lock, view):
        """Attach only this frame's unique UI association, never a hull target."""
        lock['target_track_id'] = None
        lock['track_state'] = 'unknown'
        index = lock.get('detection_index')
        if (lock.get('confirmed') is True
                and lock.get('target_association') == 'unique_healthbar'
                and isinstance(index, int) and not isinstance(index, bool)
                and 0 <= index < len(view.get('detections', []))):
            detection = view['detections'][index]
            state = detection.get('track_state', 'unknown')
            lock['track_state'] = state
            # A spatially ambiguous association is not a usable target identity.
            # Downgrade the UI lock as well as withholding the track id.
            if state == 'ambiguous':
                lock.update(confirmed=False, state='ambiguous',
                            target_association='ambiguous_track',
                            reason='ambiguous_track_association')
            else:
                lock['target_track_id'] = detection.get('track_id')
        return lock

    @staticmethod
    def ui_target(lock, view):
        """Publish only the current unique locked enemy healthbar, never a hull."""
        index = lock.get('detection_index')
        if (lock.get('confirmed') is not True
                or lock.get('target_association') != 'unique_healthbar'
                or type(index) is not int
                or not 0 <= index < len(view.get('detections', []))):
            return None
        detection = view['detections'][index]
        track_id = detection.get('track_id')
        if (type(track_id) is not int
                or lock.get('target_track_id') != track_id
                or detection.get('track_state') not in ('tentative', 'tracked')
                or detection.get('class') != 'ship'
                or detection.get('team') != 'enemy'):
            return None
        x, y, w, h = detection['nameplate_bbox']
        return dict(id=track_id, team='enemy', state=detection['track_state'],
            evidence='white_ring_associated_enemy_healthbar',
            nameplate_bbox=list(detection['nameplate_bbox']),
            nameplate_center_px=[x+w/2, y+h/2],
            screen_velocity_px_s=detection.get('screen_velocity_px_s'),
            screen_velocity_semantics=detection.get('velocity_semantics'),
            body_association='unverified', anchor_px=None,
            velocity_px_s=None, velocity_semantics='hull_motion_unknown',
            world_velocity=None, identity_association='screen_track_only')
