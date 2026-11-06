import copy
import unittest
import numpy as np
import cv2
from fire_control import FireControl
from target_confirmation import confirm_target


def observation(seq=1):
    return dict(epoch=0,seq=seq,captured_ns=seq*100_000_000,valid_until_ns=seq*100_000_000+750_000_000,fresh=True,
        lock=dict(confirmed=True,target_association='unique_healthbar',target_id=8),
        target=dict(id=8,team='enemy',state='tracked',body_association='confirmed',anchor_px=[1200,800],
                    velocity_px_s=[10,0],velocity_semantics='stable_camera_screen_pixels_per_second'),
        hud=dict(weapon=dict(kind='gun',status='observed'),reload=dict(ready=True),
                 flight_time=dict(status='observed',seconds=2),reticle=dict(status='observed',center=[1220,800])),
        camera=dict(stable=True,zoom_confirmed=True,mode='binocular'),aim_settled=True,
        mouse_response=dict(status='measured',camera_mode='binocular',screen_px_per_mouse_unit=[[2,0],[0,2]],settled_tolerance_px=2))


class FireGates(unittest.TestCase):
    def test_requires_two_fresh_settled_observations(self):
        c=FireControl()
        self.assertEqual('hold',c.decide(observation(1),100_000_000,True)['action'])
        out=c.decide(observation(2),200_000_000,True)
        self.assertTrue(out['fire_authorized'])
        self.assertEqual('fire_once',out['action'])
        self.assertFalse(c.decide(observation(2),200_000_000,True)['fire_authorized'])

    def test_lead_correction_is_not_immediate_fire(self):
        o=observation();o['hud']['reticle']['center']=[1200,800]
        out=FireControl().decide(o,100_000_000,True)
        self.assertEqual([1220,800],out['aim_pixel'])
        self.assertEqual([10,0],out['relative_mouse'])
        self.assertFalse(out['fire_authorized'])

    def test_unknown_or_mismatched_inputs_revoke(self):
        cases=[('lock',dict(confirmed=False)),('target',None),('camera',dict(stable=False)),
               ('mouse_response',dict(status='unknown'))]
        for key,value in cases:
            with self.subTest(key=key):
                o=observation();o[key]=value
                result=FireControl().decide(o,100_000_000,True)
                self.assertEqual('revoke',result['action'])
                self.assertFalse(result['fire_authorized'])
        for key,value in [('body_association','unverified'),('velocity_px_s',[float('nan'),0]),('id',9)]:
            o=observation();o['target'][key]=value
            self.assertEqual('revoke',FireControl().decide(o,100_000_000,True)['action'])

    def test_stale_lost_and_focus_revoke(self):
        for now,focused,source in [(900_000_000,True,True),(100_000_000,False,True),(100_000_000,True,False)]:
            self.assertEqual('revoke',FireControl().decide(observation(),now,focused,source)['action'])

    def test_reload_and_flight_readout_required(self):
        for group,key,value in [('reload','ready',None),('reload','ready',False),('flight_time','seconds',float('nan'))]:
            o=observation();o['hud'][group][key]=value
            self.assertEqual('revoke',FireControl().decide(o,100_000_000,True)['action'])

    def test_source_change_clears_settled_confirmation(self):
        c=FireControl();c.decide(observation(1),100_000_000,True)
        o=observation(2);o['epoch']=1
        self.assertEqual('hold',c.decide(o,200_000_000,True)['action'])

    def test_late_rejection_clears_all_motion_payloads(self):
        o=observation()
        o['hud']['reticle']['center']=[1200,800]
        o['mouse_response']['settled_tolerance_px']=None
        out=FireControl().decide(o,100_000_000,True)
        self.assertEqual('revoke',out['action'])
        self.assertEqual('invalid_settling_tolerance',out['reason'])
        self.assertFalse(out['fire_authorized'])
        self.assertIsNone(out['relative_mouse'])
        self.assertIsNone(out['aim_pixel'])
        self.assertIsNone(out['target'])
        self.assertEqual(100_000_000,out['valid_until_ns'])
        self.assertTrue(out['release_buttons'])

    def test_new_source_can_fire_after_sequence_restarts(self):
        c=FireControl()
        self.assertEqual('hold',c.decide(observation(9),900_000_000,True)['action'])
        self.assertEqual('fire_once',c.decide(observation(10),1_000_000_000,True)['action'])
        for seq,clock,expected in [(1,1_100_000_000,'hold'),(2,1_200_000_000,'fire_once')]:
            o=observation(seq)
            o.update(epoch=1,captured_ns=clock,valid_until_ns=clock+750_000_000)
            self.assertEqual(expected,c.decide(o,clock,True)['action'])

    def test_expired_settling_does_not_authorize_a_later_shot(self):
        c=FireControl()
        self.assertEqual('hold',c.decide(observation(),100_000_000,True)['action'])
        o=observation(3)
        self.assertEqual('hold',c.decide(o,300_000_000,True)['action'])
        self.assertEqual('fire_once',c.decide(observation(4),400_000_000,True)['action'])

    def test_invalid_clock_cannot_leave_an_action_deadline(self):
        for now in (True,-1,float('nan')):
            with self.subTest(now=now):
                out=FireControl().decide(observation(),now,True)
                self.assertEqual('invalid_clock',out['reason'])
                self.assertIsNone(out['valid_until_ns'])
                self.assertIsNone(out['relative_mouse'])


class LockEvidence(unittest.TestCase):
    def frame(self):
        a=np.full((1600,2560,3),80,np.uint8)
        cv2.circle(a,(850,650),33,(245,245,245),2)
        return a

    def test_ring_requires_unique_healthbar(self):
        bar=dict(nameplate_bbox=[800,720,100,8])
        result=confirm_target(self.frame(),[bar])
        self.assertTrue(result['confirmed'])
        overlap=dict(nameplate_bbox=[870,730,80,6])
        result=confirm_target(self.frame(),[bar,overlap])
        self.assertFalse(result['confirmed'])
        self.assertEqual('ambiguous',result['state'])

    def test_absent_and_partial_rings_do_not_confirm(self):
        a=np.full((1600,2560,3),80,np.uint8)
        bar=dict(nameplate_bbox=[800,720,100,8])
        self.assertFalse(confirm_target(a,[bar])['confirmed'])
        cv2.ellipse(a,(850,650),(33,33),0,0,90,(245,245,245),2)
        self.assertFalse(confirm_target(a,[bar])['confirmed'])

if __name__=='__main__': unittest.main()
