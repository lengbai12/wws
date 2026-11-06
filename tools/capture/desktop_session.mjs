#!/usr/bin/env node
/*
 * Desktop capture/control through a pi_natives build compiled WITH
 * `--features wayland-pipewire`. The module installed inside omp on this host
 * was built without that feature, so `computer.screenshot()` reports
 * "Wayland capture requires the wayland-pipewire feature".
 *
 * This wrapper neither replaces nor modifies omp's installed module. It loads
 * an explicitly named build and reports what it actually returned. A file on
 * disk proves capture bytes, not upstream render age or input delivery.
 *
 *   node tools/capture/desktop_session.mjs capabilities
 *   node tools/capture/desktop_session.mjs displays
 *   node tools/capture/desktop_session.mjs windows
 *   node tools/capture/desktop_session.mjs screenshot <out.png> [target]
 *   node tools/capture/desktop_session.mjs control-state
 *   node tools/capture/desktop_session.mjs control-acquire
 *   node tools/capture/desktop_session.mjs control-release
 *
 * Screenshot targets follow the desktop facade: "active", "all", or a display
 * or window id from `displays`/`windows`.
 */
import { createHash } from 'node:crypto';
import { writeFileSync } from 'node:fs';

const DEFAULT_MODULE = new URL(
  '../../state/third-party/oh-my-pi-desktop/target/debug/pi_natives.wayland-probe.node',
  import.meta.url,
);

const modulePath = process.env.DESKTOP_NATIVE_MODULE || DEFAULT_MODULE.pathname;
const argv = process.argv.slice(2);
const command = argv.shift();
const positional = argv.filter((value) => !value.startsWith('--'));

const loaded = await import(modulePath);
const natives = loaded.default ?? loaded;
const { DesktopSession } = natives;

if (!DesktopSession) {
  console.error(JSON.stringify({ status: 'unavailable', module: modulePath, reason: 'DesktopSession missing' }));
  process.exit(2);
}

const session = new DesktopSession();

async function finish(payload, code = 0) {
  try { session.retire(); } catch { /* already retired */ }
  try { session.close(); } catch { /* already closed */ }
  console.log(JSON.stringify(payload));
  process.exit(code);
}

try {
  switch (command) {
    case 'capabilities':
      await finish({ status: 'ok', module: modulePath,
                     capabilities: session.capabilities, screen: session.screenState });
      break;
    case 'displays':
      await finish({ status: 'ok', displays: await session.listDisplays() });
      break;
    case 'windows':
      await finish({ status: 'ok', windows: await session.listWindows() });
      break;
    case 'screenshot': {
      const path = positional[0];
      const target = positional[1] || 'active';
      if (!path) await finish({ status: 'refused', reason: 'screenshot needs an output path' }, 2);
      const shot = await session.capture(target);
      const bytes = Buffer.from(shot?.data ?? []);
      if (!bytes.length) await finish({ status: 'failed', reason: 'capture returned no bytes', target }, 1);
      writeFileSync(path, bytes);
      await finish({ status: 'ok', path, target, bytes: bytes.length,
                     sha256: createHash('sha256').update(bytes).digest('hex'),
                     width: shot.width ?? null, height: shot.height ?? null,
                     source_width: shot.source_width ?? null, source_height: shot.source_height ?? null,
                     backend: shot.backend ?? null, display_server: shot.display_server ?? null,
                     screen_locked: shot.screen_locked ?? null });
      break;
    }
    case 'click': {
      const [x, y] = positional.map(Number);
      const target = positional[2] || 'display:active';
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        await finish({ status: 'refused', reason: 'click needs finite x y' }, 2);
      }
      await session.click(target, x, y, { takeover: true });
      await finish({ status: 'delivered', target, screen_point: [x, y] });
      break;
    }
    case 'keychord': {
      const keys = String(positional[0] || '').split('+').filter(Boolean);
      const target = positional[1] || 'display:active';
      if (!keys.length) await finish({ status: 'refused', reason: 'keychord needs keys' }, 2);
      await session.keyChord(target, keys, { takeover: true });
      await finish({ status: 'delivered', keys, target });
      break;
    }
    case 'capture-click': {
      const [x, y] = positional.map(Number);
      const target = positional[2] || 'display:active';
      const shot = await session.capture(target);
      await session.click(target, x, y, { takeover: true });
      await finish({ status: 'delivered', target, screen_point: [x, y],
                     frame_width: shot.width ?? null, frame_height: shot.height ?? null });
      break;
    }
    case 'control-state':
      await finish({ status: 'ok', control: session.controlState() });
      break;
    case 'control-acquire':
      await finish({ status: 'ok', control: await session.acquireControl() });
      break;
    case 'control-release':
      await finish({ status: 'ok', control: session.releaseControl() });
      break;
    default:
      await finish({ status: 'refused', reason: `unknown command: ${command ?? '(none)'}` }, 2);
  }
} catch (error) {
  await finish({ status: 'failed', command, reason: String(error) }, 1);
}
