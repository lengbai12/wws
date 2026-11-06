"""Census every entity, property and METHOD a replay actually delivers.

The project rule it implements: a data class is declared absent only after the
whole stream has been censused, because the interesting signals are not always
entities. Planes were missed by an entity-only census and then found as method
calls on ``Avatar``; the aim lock was searched in properties and found to be
mostly never sent. This tool answers "what is in this replay" in one pass by
hooking the parser's two dispatch points, so the next question is a grep rather
than a guess.

  census_stream.py <replay> [--grep lock|aim|target|select] [--top 25]
"""

from __future__ import annotations

import argparse
import collections
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import export_events as E  # noqa: E402
from fetch_schema import DEFAULT_BUILD, DEFAULT_CACHE, DEFAULT_COMMIT, fetch  # noqa: E402


def census(replay_path: Path, cache_root: Path, top: int, grep: str | None) -> int:
    entity_module, _, ReplayPlayer = E._load_parser(cache_root)
    replay = E.reader.read(replay_path)
    version_list = (replay.meta.get("clientVersionFromExe") or "").split(",")

    methods: collections.Counter = collections.Counter()
    properties: collections.Counter = collections.Counter()
    entities: collections.Counter = collections.Counter()

    Entity = entity_module.Entity
    original_method = Entity.call_client_method
    original_property = Entity.set_client_property

    def call_client_method(self, exposed_index, payload):
        try:
            method = self._methods[exposed_index]
            methods["%s.%s" % (self._spec.get_name(), method.get_name())] += 1
        except Exception:  # noqa: BLE001 - a census must never become the failure
            methods["%s.<index %s>" % (self._spec.get_name(), exposed_index)] += 1
        return original_method(self, exposed_index, payload)

    def set_client_property(self, exposed_index, payload):
        try:
            prop = self.client_properties[exposed_index]
            properties["%s.%s" % (self._spec.get_name(), prop.get_name())] += 1
        except Exception:  # noqa: BLE001
            properties["%s.<index %s>" % (self._spec.get_name(), exposed_index)] += 1
        return original_property(self, exposed_index, payload)

    Entity.call_client_method = call_client_method
    Entity.set_client_property = set_client_property
    E.reset_subscriptions(entity_module)
    player = ReplayPlayer(version_list)
    original_create = entity_module.Entity.__init__
    player.play(replay.stream)
    Entity.call_client_method = original_method
    Entity.set_client_property = original_property

    for entity in player._battle_controller._entities.values():
        entities[entity._spec.get_name()] += 1

    print("replay %s" % replay_path.name)
    print("\n=== entities ===")
    for name, count in entities.most_common():
        print("  %-24s %5d" % (name, count))
    print("\n=== method calls (%d distinct, %d total) ==="
          % (len(methods), sum(methods.values())))
    for name, count in methods.most_common(top):
        print("  %-44s %6d" % (name, count))
    print("\n=== property deliveries (%d distinct, %d total) ==="
          % (len(properties), sum(properties.values())))
    for name, count in properties.most_common(top):
        print("  %-44s %6d" % (name, count))
    if grep:
        print("\n=== matches for %r ===" % grep)
        for label, table in (("method", methods), ("property", properties)):
            for name, count in sorted(table.items()):
                if grep.lower() in name.lower():
                    print("  %-8s %-44s %6d" % (label, name, count))
    return 0


def main() -> int:
    logging.getLogger().setLevel(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--grep")
    args = parser.parse_args()
    cache_root = fetch(DEFAULT_BUILD, DEFAULT_COMMIT, args.cache, False)
    return census(args.replay, cache_root, args.top, args.grep)


if __name__ == "__main__":
    sys.exit(main())
