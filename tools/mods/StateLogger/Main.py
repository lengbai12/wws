"""StateLogger — a PnFMods mod that records the battle state the bullets need.

Why it exists: to fire at a ship you need its BEARING, RANGE and relative VELOCITY,
not its pixels. The client knows all three exactly (it draws the aiming nomogram
from them) plus WHICH ship the player has locked -- and the replay stream cannot
carry the lock at all, because `setWeaponLock`/`setPriorityTarget` are
client->server. So the game itself is asked, through its own mod extension point
(`res_mods`), which is officially supported and needs no game files modified.

SANDBOX: the client imports a mod into a restricted interpreter whose allowed
modules are `xml, copy, struct, Keys, Math, datetime, re, math, time, array,
SpatialUI, xml.dom` -- `os`, `sys`, `json` and `traceback` are NOT importable (a
first version of this mod died on exactly that). So paths use string operations on
`__file__`, JSON is encoded by hand, and every failure is caught and written into
the log instead of relying on a traceback module.

The client's Python exposes the battle through `dataHub` component collections
(`entity[CC.<component>]`), `battle`, `constants.UiComponents` and `events`, and
this mod does NOT assume which component carries positions: it writes a DISCOVERY
dump (component names, every component's attribute names and a repr of the
interesting ones) plus which imports actually succeeded, then samples on every
event it can hook by NAME PATTERN (pos/coord/speed/dist/angle/yaw/bearing/target/
lock/aim/select/team/hp). A wrong guess about a name therefore costs a line of log
rather than a silent zero.
"""

API_VERSION = 'API_v1.0'

import time                      # whitelisted by the mod sandbox

events = dataHub = constants = battle = utils = None
BigWorld = None
IMPORT_STATUS = {}

for _name in ('events', 'ui', 'utils', 'dataHub', 'constants', 'battle', 'Math',
              'SpatialUI', 'BigWorld', 'Keys'):
    try:
        globals()[_name] = __import__(_name)
        IMPORT_STATUS[_name] = 'ok'
    except Exception as _exc:
        IMPORT_STATUS[_name] = '%s: %s' % (type(_exc).__name__, str(_exc)[:120])

MOD_NAME = 'StateLogger'
TICK_SECONDS = 0.2

# Where to append. Absolute Wine path first (the workspace the pipeline reads),
# then game-relative candidates, then next to this file (string ops, no os module).
OUTPUT_CANDIDATES = [
    'Z:/home/lengbai/Downloads/wws/profile/state-log.jsonl',
    'profile/state-log.jsonl',
    '../../../../profile/state-log.jsonl',
]
try:
    _here = __file__.replace('\\', '/').rsplit('/', 1)[0]
    OUTPUT_CANDIDATES.append(_here + '/state-log.jsonl')
except Exception:
    pass

INTERESTING = ('pos', 'coord', 'speed', 'vel', 'dist', 'angle', 'yaw', 'bearing', 'dir',
               'target', 'lock', 'aim', 'select', 'team', 'health', 'hp', 'id', 'name',
               'x', 'y', 'z')

_output_path = None
_discovery_done = False
_tick_armed = False
_tick_source = 'none'
_self_id = None


def _encode(value, depth=0):
    """Hand-rolled JSON: `json` is not importable inside the mod sandbox."""
    if depth > 4:
        return 'null'
    if value is None:
        return 'null'
    if value is True or value is False:
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (value != value or value in (float('inf'),
                                                                     float('-inf'))):
            return 'null'
        return repr(value)
    if isinstance(value, str):
        return '"%s"' % value.replace('\\', '\\\\').replace('"', '\\"')[:240]
    if isinstance(value, (list, tuple)):
        return '[%s]' % ','.join(_encode(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return '{%s}' % ','.join('"%s":%s' % (str(key)[:60], _encode(item, depth + 1))
                                 for key, item in value.items())
    return '"%s"' % repr(value)[:160].replace('"', "'")


def _log(row):
    global _output_path
    line = _encode(row)
    for path in ([_output_path] if _output_path else []) + OUTPUT_CANDIDATES:
        if not path:
            continue
        try:
            handle = open(path, 'a')
            handle.write(line + '\n')
            handle.close()
            _output_path = path
            return
        except Exception:
            continue


def _interesting_attrs(obj, limit=40):
    out = {}
    try:
        names = [n for n in dir(obj) if not n.startswith('_')]
    except Exception:
        return out
    for name in names[:400]:
        lowered = name.lower()
        if not any(token in lowered for token in INTERESTING):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if value is None or isinstance(value, (dict, list, tuple, set)) or callable(value):
            continue
        try:
            text = repr(value)
        except Exception:
            continue
        if len(text) > 140:
            continue
        out[name] = value if isinstance(value, (int, float, str, bool)) else text
        if len(out) >= limit:
            break
    return out


def _component_names():
    out = {}
    try:
        for name in dir(constants.UiComponents):
            if name.startswith('_'):
                continue
            value = getattr(constants.UiComponents, name)
            if isinstance(value, (int, str)):
                out[name] = value
    except Exception:
        pass
    return out


def _collections():
    names = []
    for holder in ('_collections', '_collectionByName', '_entitiesByName', 'collections'):
        try:
            found = getattr(dataHub, holder)
            if isinstance(found, dict):
                names.extend(str(key) for key in found.keys())
        except Exception:
            continue
    return sorted(set(names))


def _dump_discovery():
    global _self_id
    _log({'event': 'discovery', 'mod': MOD_NAME, 'imports': IMPORT_STATUS,
          'output': _output_path, 'tick_source': _tick_source})
    components = _component_names()
    _log({'event': 'components', 'count': len(components), 'names': sorted(components)})
    _log({'event': 'datahub', 'collections': _collections(),
          'attrs': [n for n in dir(dataHub) if not n.startswith('_')][:90]})
    try:
        info = battle.getSelfPlayerInfo()
        _self_id = getattr(info, 'id', None)
        _log({'event': 'self', 'id': _self_id, 'attrs': _interesting_attrs(info, 70)})
    except Exception as exc:
        _log({'event': 'self-error', 'error': repr(exc)})
    for name in _collections():
        if not any(token in name.lower() for token in
                   ('ship', 'marker', 'minimap', 'battle', 'vehicle', 'map')):
            continue
        try:
            entities = dataHub.getEntityCollections(name)
        except Exception as exc:
            _log({'event': 'collection-error', 'name': name, 'error': repr(exc)})
            continue
        if not entities:
            continue
        first = entities[0]
        try:
            keys = [str(key) for key in first.keys()]
        except Exception:
            keys = []
        _log({'event': 'collection', 'name': name, 'entities': len(entities),
              'entity_keys': keys})
        for key in keys[:12]:
            try:
                component = first[key]
            except Exception:
                continue
            attrs = _interesting_attrs(component, 30)
            if attrs:
                _log({'event': 'component', 'collection': name, 'key': key,
                      'attrs': attrs})


def _sample():
    row = {'event': 'tick', 'self_id': _self_id, 't': round(time.time(), 3),
           'ships': []}
    for label, getter in (('self_player_info', lambda: battle.getSelfPlayerInfo()),
                          ('hud', lambda: dataHub.getSingleEntity('battleHud'))):
        try:
            obj = getter()
        except Exception:
            continue
        if obj is None:
            continue
        found = {k: v for k, v in _interesting_attrs(obj, 70).items()
                 if any(token in k.lower() for token in ('target', 'lock', 'aim', 'select'))}
        if found:
            row.setdefault('lock_candidates', {})[label] = found
    for name in _collections():
        lowered = name.lower()
        if 'ship' not in lowered and 'marker' not in lowered:
            continue
        try:
            entities = dataHub.getEntityCollections(name)
        except Exception:
            continue
        for entity in entities[:80]:
            ship = {'collection': name}
            try:
                for key in entity.keys():
                    attrs = _interesting_attrs(entity[key], 26)
                    if attrs:
                        ship.setdefault('components', {})[str(key)] = attrs
            except Exception:
                continue
            if len(ship) > 1:
                row['ships'].append(ship)
    _log(row)


def _tick():
    global _tick_armed
    try:
        if not _discovery_done:
            _dump_discovery()
            globals()['_discovery_done'] = True
        _sample()
    except Exception as exc:
        _log({'event': 'tick-error', 'error': repr(exc)})
    if BigWorld is not None and hasattr(BigWorld, 'callback'):
        try:
            BigWorld.callback(TICK_SECONDS, _tick)
            globals()['_tick_source'] = 'BigWorld.callback'
            return
        except Exception as exc:
            _log({'event': 'callback-error', 'error': repr(exc)})
    _tick_armed = False


class StateLogger(object):
    def __init__(self):
        _log({'event': 'loaded', 'mod': MOD_NAME, 'api': API_VERSION,
              'imports': IMPORT_STATUS, 'candidates': OUTPUT_CANDIDATES})
        if events is None or dataHub is None or constants is None:
            _log({'event': 'load-error',
                  'error': 'client modules missing; see imports above',
                  'imports': IMPORT_STATUS})
            return
        for hook in ('onBattleShown', 'onBattleStart', 'onBattleQuit',
                     'onReceiveShellInfo', 'onMinimapUpdate'):
            callback = (self.__on_battle if hook in ('onBattleShown', 'onBattleStart')
                        else self.__on_event)
            try:
                getattr(events, hook)(callback)
                IMPORT_STATUS['hook:' + hook] = 'ok'
            except Exception as exc:
                IMPORT_STATUS['hook:' + hook] = repr(exc)[:80]
        _log({'event': 'hooked', 'hooks': IMPORT_STATUS})

    def __on_battle(self, *args):
        global _discovery_done, _tick_armed
        _log({'event': 'battle-shown'})
        _discovery_done = False
        globals()['_discovery_done'] = False
        if not _tick_armed:
            _tick_armed = True
            _tick()

    def __on_event(self, *args):
        # No timer: sample on whatever events the client does offer, so the log has
        # more than one moment even when BigWorld.callback is unavailable.
        try:
            _sample()
        except Exception as exc:
            _log({'event': 'event-error', 'error': repr(exc)})


gStateLogger = StateLogger()
