#!/usr/bin/env python3
"""Export vehicle -> species reference from packet roster shipParamsId, for evaluation.

Use a separate process per replay: the schema parser's imports bind to one build.
This reference is never supplied to the image recognizer.
"""
import argparse
import json
import logging
from pathlib import Path

import export_events as E


def export_species(path, ships_path):
    replay = E.reader.read(path)
    version = [part.strip() for part in replay.meta['clientVersionFromExe'].split(',')]
    cache = E.cache_for_build(E.DEFAULT_CACHE, version)
    entity, _, player_type = E._load_parser(cache)
    E.reset_subscriptions(entity)
    player = player_type(version)
    player.play(replay.stream)
    ships = json.loads(Path(ships_path).read_text())
    result = {}
    for row in player._battle_controller._players.get_info().values():
        vehicle = row.get('shipId')
        params = row.get('shipParamsId')
        species = ships.get(str(params), {}).get('species')
        if vehicle is not None and species:
            result[str(vehicle)] = dict(species=species, ship_params_id=params)
    return dict(replay=str(path), build='.'.join(version[:3]),
                source='packet roster shipId -> shipParamsId -> ships.json', vehicles=result)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('replay', type=Path)
    ap.add_argument('--ships', default='state/toalba/gamedata/data/ships.json')
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)
    print(json.dumps(export_species(args.replay, args.ships)))


if __name__ == '__main__':
    main()
