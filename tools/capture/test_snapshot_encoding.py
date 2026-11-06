"""A PPM snapshot must skip PNG encoding, which costs about a second per frame."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_host import CaptureHost


class SnapshotEncodingTests(unittest.TestCase):
    def test_ppm_writes_raw_pixels(self):
        elements = CaptureHost.snapshot_elements('/tmp/frame.ppm')
        self.assertNotIn('pngenc', elements)
        self.assertEqual(elements[0], 'filesink')

    def test_png_still_encodes(self):
        elements = CaptureHost.snapshot_elements('/tmp/frame.png')
        self.assertIn('pngenc', elements)
        self.assertEqual(elements[-2], 'filesink')

    def test_other_extensions_default_to_png(self):
        self.assertIn('pngenc', CaptureHost.snapshot_elements('/tmp/frame'))


if __name__ == '__main__':
    unittest.main()
