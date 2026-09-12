import importlib.util
import sys
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('ram_probe', Path(__file__).resolve().parents[1] / 'tools/ram_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTest(unittest.TestCase):
    def test_allocation_and_exit(self):
        result = probe.measure([sys.executable, '-c',
                                'import time; x=bytearray(32*1024*1024); time.sleep(.1)'],
                               interval=.01, settle=0)
        self.assertEqual(result['returncode'], 0)
        self.assertGreater(len(result['samples']), 2)
        self.assertGreaterEqual(result['system_peak_delta_bytes'], 0)
        self.assertGreaterEqual(result['after_exit']['monotonic_s'], result['samples'][-1]['monotonic_s'])

    def test_failure_is_preserved(self):
        result = probe.measure([sys.executable, '-c', 'raise SystemExit(7)'], settle=0)
        self.assertEqual(result['returncode'], 7)

    def test_timeout_terminates_command_group(self):
        result = probe.measure(
            [sys.executable, '-c', 'import time; time.sleep(10)'],
            interval=.01, settle=0, timeout=.05)
        self.assertTrue(result['timed_out'])
        self.assertIsNotNone(result['returncode'])

    def test_reports_minimum_available_memory(self):
        result = probe.measure(
            [sys.executable, '-c',
             'import time; x=bytearray(64*1024*1024); time.sleep(.15)'],
            interval=.01, settle=0)
        observed = min(sample['available_bytes'] for sample in result['samples'])
        self.assertEqual(result['minimum_available_bytes'], observed)
        self.assertGreaterEqual(result['minimum_available_at_s'], 0)


if __name__ == '__main__':
    unittest.main()
