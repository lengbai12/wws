#!/usr/bin/env python3
"""Image evidence -> expiring fire/aim intents. Does not inject input.

Screen extrapolation is usable only with a stable camera and a confirmed hull
anchor. Neither a nameplate nor a lock ring establishes that anchor.
"""
import argparse
import json
import math
from pathlib import Path
import time


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def point(value):
    return isinstance(value, (tuple, list)) and len(value)==2 and all(finite(v) for v in value)


class FireControl:
    def __init__(self, ttl_ns=150_000_000, max_lead_s=15.0):
        if not isinstance(ttl_ns,int) or isinstance(ttl_ns,bool) or ttl_ns<=0:
            raise ValueError('ttl_ns must be positive integer')
        if not finite(max_lead_s) or max_lead_s<=0:
            raise ValueError('max_lead_s must be positive finite')
        self.ttl_ns, self.max_lead_s = ttl_ns,max_lead_s
        self.epoch=None
        self.seq=-1
        self.captured=-1
        self.now=-1
        self.target=None
        self.last_fire_seq=-1
        self.target_valid_until=0

    def decide(self, observation, now_ns=None, focused=False, source_available=True):
        now_ns=time.monotonic_ns() if now_ns is None else now_ns
        out=dict(kind='fire_control',action='revoke',fire_authorized=False,
                 aim_pixel=None,relative_mouse=None,target=None,valid_until_ns=now_ns,
                 reason='invalid_observation',release_buttons=True)
        def stop(reason):
            out.update(action='revoke', fire_authorized=False, aim_pixel=None,
                       relative_mouse=None, target=None, valid_until_ns=now_ns,
                       reason=reason, release_buttons=True)
            out['valid_until_ns']=now_ns if isinstance(now_ns,int) and not isinstance(now_ns,bool) and now_ns>=0 else None
            self.target=None
            self.target_valid_until=0
            return out
        if not isinstance(now_ns,int) or isinstance(now_ns,bool) or now_ns<0:
            out['valid_until_ns']=None
            return stop('invalid_clock')
        if now_ns<self.now: return stop('clock_regressed')
        self.now=now_ns
        if not focused: return stop('focus_lost')
        if not source_available: return stop('source_lost')
        if not isinstance(observation,dict): return stop('invalid_observation')
        o=observation
        epoch,seq,captured,expiry=[o.get(k) for k in ('epoch','seq','captured_ns','valid_until_ns')]
        if any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in (epoch,seq,captured,expiry)):
            return stop('invalid_stamp')
        if self.epoch is not None and epoch<self.epoch: return stop('old_epoch')
        if self.epoch!=epoch:
            self.epoch,self.seq,self.captured=epoch,-1,-1
            self.target=None
            self.target_valid_until=0
            self.last_fire_seq=-1
        if seq<=self.seq or captured<=self.captured: return stop('out_of_order')
        self.seq,self.captured=seq,captured
        if o.get('fresh') is not True or captured>now_ns or expiry<=now_ns:
            return stop('expired_or_future_observation')
        out['valid_until_ns']=min(expiry,now_ns+self.ttl_ns)
        lock=o.get('lock',{})
        if lock.get('confirmed') is not True or lock.get('target_association')!='unique_healthbar':
            return stop('lock_target_unconfirmed')
        target=o.get('target')
        if not isinstance(target,dict) or target.get('team')!='enemy' or target.get('state')!='tracked':
            return stop('enemy_track_unconfirmed')
        if target.get('body_association')!='confirmed' or not point(target.get('anchor_px')):
            return stop('hull_anchor_unconfirmed')
        if target.get('id') is None or lock.get('target_id')!=target['id']:
            return stop('lock_track_identity_mismatch')
        hud=o.get('hud',{})
        if hud.get('weapon',{}).get('kind')!='gun' or hud['weapon'].get('status')!='observed':
            return stop('main_battery_unconfirmed')
        if hud.get('reload',{}).get('ready') is not True: return stop('guns_not_ready')
        flight=hud.get('flight_time',{})
        travel=flight.get('seconds')
        if flight.get('status')!='observed' or not finite(travel) or not 0<travel<=self.max_lead_s:
            return stop('flight_time_unknown')
        camera=o.get('camera',{})
        if camera.get('stable') is not True or camera.get('zoom_confirmed') is not True:
            return stop('camera_or_zoom_unconfirmed')
        velocity=target.get('velocity_px_s')
        if not point(velocity) or target.get('velocity_semantics')!='stable_camera_screen_pixels_per_second':
            return stop('target_motion_unknown')
        reticle=hud.get('reticle',{})
        if reticle.get('status')!='observed' or not point(reticle.get('center')):
            return stop('reticle_unknown')
        response=o.get('mouse_response',{})
        matrix=response.get('screen_px_per_mouse_unit')
        if response.get('status')!='measured' or response.get('camera_mode')!=camera.get('mode'):
            return stop('mouse_response_uncalibrated')
        if (not isinstance(matrix,list) or len(matrix)!=2 or not all(point(r) for r in matrix)):
            return stop('invalid_mouse_response')
        determinant=matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
        if abs(determinant)<1e-6: return stop('singular_mouse_response')
        horizon=travel+(now_ns-captured)/1e9
        aim=[p+v*horizon for p,v in zip(target['anchor_px'],velocity)]
        if not 0<=aim[0]<2560 or not 80<=aim[1]<1520: return stop('aim_out_of_view')
        dx,dy=[a-b for a,b in zip(aim,reticle['center'])]
        mouse=[(matrix[1][1]*dx-matrix[0][1]*dy)/determinant,
               (-matrix[1][0]*dx+matrix[0][0]*dy)/determinant]
        out.update(action='aim',reason='correcting_lead',aim_pixel=aim,relative_mouse=mouse,
                   target=target['id'],release_buttons=True)
        tolerance=response.get('settled_tolerance_px')
        if not finite(tolerance) or not 0<tolerance<=5: return stop('invalid_settling_tolerance')
        # A moved cursor is never treated as settled in the same decision.
        if math.hypot(dx,dy)<=tolerance and o.get('aim_settled') is True:
            if self.target==target['id'] and now_ns<self.target_valid_until and seq>self.last_fire_seq:
                out.update(action='fire_once',fire_authorized=True,reason='fresh_confirmed_solution',
                           relative_mouse=None,release_buttons=False)
                self.last_fire_seq=seq
                self.target=None
            else:
                out.update(action='hold',reason='confirming_settled_solution',relative_mouse=None)
                self.target=target['id']
                self.target_valid_until=out['valid_until_ns']
        else: self.target=None
        return out


def observe_image(rgb, stamp):
    from main_view_perception import perceive_frame
    from target_confirmation import confirm_target
    from fire_hud import FireHudReader
    view=perceive_frame(rgb)
    lock=confirm_target(rgb,view['detections'])
    return dict(stamp,view=view,lock=lock,hud=FireHudReader().detect(rgb),target=None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    from PIL import Image
    import numpy as np
    started=time.monotonic_ns()
    rgb=np.array(Image.open(args.image).convert('RGB'))
    observation=observe_image(rgb,dict(epoch=0,seq=0,captured_ns=started,
        valid_until_ns=started+750_000_000,fresh=True))
    # A file has no live capture timestamp. Exercise visual gates on a simulated
    # clock; processing latency must not hide the actual rejection reason.
    intent=FireControl().decide(observation,now_ns=started,focused=True)
    report=dict(mode='image_only_preview',input_injected=False,source=str(args.image),
                clock='simulated_offline_not_live_freshness',
                processing_elapsed_ns=time.monotonic_ns()-started,
                observation=observation,intent=intent)
    with args.out.open('x') as handle: json.dump(report,handle,indent=2,allow_nan=False)
    print(json.dumps(intent,indent=2,allow_nan=False))


if __name__=='__main__': main()
