#!/home/lengbai/Downloads/wws/.venv/bin/python
"""Read only WoWS's XComposite backing pixmap; never focus or capture desktop.

Acquisition timestamps bound the read operation, not game render age. They must
not be substituted for source render timestamps in firing freshness guards.
--wait-update adds a bounded XDamage wait and a host-monotonic lower bound for
later drawable damage, then reads that same pixmap. Upstream render latency is
unknown even when damage is observed; source render age stays unverified.
Build: cc -O2 tools/capture/window_pixels.c -o state/window_pixels
       $(pkg-config --cflags --libs x11 xcomposite xdamage xfixes)
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time

from PIL import Image
from probe_aim_input import game_identity, process_identity, ProbeError

ROOT = Path(__file__).resolve().parents[2]
DRAWABLE_UPDATE_SEMANTICS = (
    'xdamage_after_cleared_baseline_before_xgetimage;upstream_render_latency_unknown')


def drawable_update_evidence(stderr, started, completed):
    """Accept only native evidence bounded by this host's capture operation."""
    try:
        evidence = json.loads(stderr)
    except (ValueError, UnicodeDecodeError) as error:
        raise ProbeError('missing or malformed XDamage drawable update evidence') from error
    if not isinstance(evidence, dict):
        raise ProbeError('missing or malformed XDamage drawable update evidence')
    lower_bound = evidence.get('drawable_update_lower_bound_monotonic_ns')
    if (type(lower_bound) is not int or lower_bound <= 0 or
            not started <= lower_bound <= completed or
            evidence.get('drawable_update_semantics') != DRAWABLE_UPDATE_SEMANTICS):
        raise ProbeError('invalid XDamage drawable update evidence or clock bounds')
    return dict(drawable_update_lower_bound_monotonic_ns=lower_bound,
                drawable_update_semantics=DRAWABLE_UPDATE_SEMANTICS)


class WindowCapture:
    def __init__(self):
        self.identity = game_identity()
        self.env = {**os.environ, 'DISPLAY': self.identity['display']}
        self.binary = ROOT / 'state/window_pixels'
        if not self.binary.is_file():
            raise ProbeError('build state/window_pixels from tools/capture/window_pixels.c first')
        windows = self.command('xdotool', 'search', '--onlyvisible', '--pid', str(self.identity['pid'])).split()
        verified = []
        for window in windows:
            if self.command('xdotool', 'getwindowpid', window).strip() != str(self.identity['pid']):
                continue
            geometry = dict(line.split('=', 1) for line in self.command(
                'xdotool', 'getwindowgeometry', '--shell', window).splitlines())
            if int(geometry['WIDTH']) >= 640 and int(geometry['HEIGHT']) >= 480:
                verified.append(window)
        if len(verified) != 1:
            raise ProbeError(f'expected one visible game surface, found {len(verified)}')
        self.window = verified[0]

    def command(self, *args):
        return subprocess.run(args, env=self.env, capture_output=True, text=True,
                              timeout=5, check=True).stdout

    def check(self):
        if process_identity(self.identity['pid']) != self.identity:
            raise ProbeError('game generation changed')
        if self.command('xdotool', 'getwindowpid', self.window).strip() != str(self.identity['pid']):
            raise ProbeError('window owner changed')

    def acquire(self, wait_update=False):
        """Return an RGB PIL image and evidence without PNG encoding or writes.

        raw_pixel_sha256 hashes the row-major RGB bytes. Consumers may inspect
        this image immediately and defer save() until after their decision.
        """
        self.check()
        started = time.monotonic_ns()
        command = [str(self.binary), self.window]
        if wait_update:
            command.append('--wait-update')
        try:
            result = subprocess.run(command, env=self.env, capture_output=True,
                                    timeout=5, check=True)
        except subprocess.TimeoutExpired as error:
            if not wait_update:
                raise
            raise ProbeError('window capture exceeded its 5 second timeout') from error
        except subprocess.CalledProcessError as error:
            if not wait_update:
                raise
            detail = (error.stderr or b'').decode('utf-8', errors='replace').strip()
            raise ProbeError(f'XDamage window capture failed: {detail or error}') from error
        completed = time.monotonic_ns()
        self.check()
        update_evidence = (drawable_update_evidence(result.stderr, started, completed)
                           if wait_update else {})
        image = Image.open(io.BytesIO(result.stdout))
        image.load()
        meta = dict(source='xcomposite_window_pixmap', identity=self.identity,
                    window=self.window, width=image.width, height=image.height,
                    acquisition_started_monotonic_ns=started,
                    acquisition_completed_monotonic_ns=completed,
                    source_render_monotonic_ns=None, render_age_verified=False,
                    raw_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                    focus_changed=False, desktop_captured=False)
        meta.update(update_evidence)
        return image, meta

    def save(self, path, image, meta):
        """Persist a previously acquired image without replacing its timestamps."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as output:
            image.save(output, format='PNG')
        saved_meta = dict(meta, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        Path(str(path)+'.json').write_text(json.dumps(saved_meta, indent=2)+'\n')
        return saved_meta

    def grab(self, path, wait_update=False):
        """Acquire and persist one window capture; default behavior is unchanged."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        image, meta = self.acquire(wait_update=wait_update)
        return self.save(path, image, meta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=1)
    parser.add_argument('--interval', type=float, default=.5)
    parser.add_argument('--wait-update', action='store_true',
                        help='wait up to 500 ms for later drawable damage; render age stays unverified')
    args = parser.parse_args()
    if not 1 <= args.frames <= 100 or not .05 <= args.interval <= 10:
        parser.error('frames must be 1..100 and interval .05..10 seconds')
    args.out.mkdir(parents=True, exist_ok=False)
    capture = WindowCapture()
    frames = []
    for i in range(args.frames):
        if i:
            time.sleep(args.interval)
        frames.append(capture.grab(args.out / f'frame-{i:03d}.png', wait_update=args.wait_update))
    report = dict(identity=capture.identity, window=capture.window, frames=frames,
                  desktop_captured=False, focus_changed=False)
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
