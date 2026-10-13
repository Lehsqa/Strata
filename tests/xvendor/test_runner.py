import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('runner', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_environments_both_orders(self):
        source = {'PATH': os.environ['PATH'], 'LD_LIBRARY_PATH': '/base',
                  'CUDA_VISIBLE_DEVICES': '9', 'CUDA_DEVICE_ORDER': 'FASTEST_FIRST',
                  'HIP_VISIBLE_DEVICES': '8', 'ROCR_VISIBLE_DEVICES': '7'}
        for order in [('cuda', 'hip'), ('hip', 'cuda')]:
            for vendor in order:
                env = runner.vendor_env(vendor, '2', '/hip', source)
                if vendor == 'cuda':
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2')
                    self.assertEqual(env['CUDA_DEVICE_ORDER'], 'PCI_BUS_ID')
                    self.assertNotIn('HIP_VISIBLE_DEVICES', env)
                    self.assertNotIn('ROCR_VISIBLE_DEVICES', env)
                    self.assertEqual(env['LD_LIBRARY_PATH'], '/base')
                else:
                    self.assertEqual(env['HIP_VISIBLE_DEVICES'], '2')
                    self.assertNotIn('CUDA_VISIBLE_DEVICES', env)
                    self.assertNotIn('CUDA_DEVICE_ORDER', env)
                    self.assertNotIn('ROCR_VISIBLE_DEVICES', env)
                    self.assertEqual(env['LD_LIBRARY_PATH'], '/hip:/base')
        self.assertEqual(source['CUDA_VISIBLE_DEVICES'], '9')

    def children(self, second):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        pid_paths = [Path(folder.name) / f'{i}.pid' for i in range(2)]
        fd = os.memfd_create('runner-test')
        self.addCleanup(os.close, fd)
        os.ftruncate(fd, 4096)
        programs = [
            "import os,sys,time; from pathlib import Path; "
            "Path(sys.argv[2]).write_text(str(os.getpid())); "
            "os.pwrite(int(sys.argv[1]),b'A',0); time.sleep(30)",
            "import os,sys,time; from pathlib import Path; "
            "Path(sys.argv[2]).write_text(str(os.getpid())); " + second,
        ]
        commands = [[sys.executable, '-c', code, str(fd), str(path)]
                    for code, path in zip(programs, pid_paths)]
        return fd, commands, pid_paths

    def assert_reaped(self, paths):
        for path in paths:
            if path.exists():
                pid = int(path.read_text())
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_fd_inherited_both_execs_and_failure_cleanup(self):
        fd, commands, paths = self.children("os.pwrite(int(sys.argv[1]),b'B',1); time.sleep(.15); sys.exit(3)")
        begin = time.monotonic()
        self.assertNotEqual(runner.run_pair(commands, [os.environ.copy()] * 2, fd, 3), 0)
        self.assertLess(time.monotonic() - begin, 2)
        self.assertEqual(os.pread(fd, 2, 0), b'AB')
        self.assert_reaped(paths)

    def test_timeout_reaps_both(self):
        fd, commands, paths = self.children('time.sleep(30)')
        self.assertNotEqual(runner.run_pair(commands, [os.environ.copy()] * 2, fd, .3), 0)
        self.assert_reaped(paths)

    def test_second_spawn_failure_reaps_first(self):
        fd, commands, paths = self.children('time.sleep(30)')
        commands[1] = ['/does/not/exist']
        self.assertNotEqual(runner.run_pair(commands, [os.environ.copy()] * 2, fd, 2), 0)
        self.assert_reaped(paths)

    def test_bounded_arguments(self):
        for args in [['--rounds', '0'], ['--rounds', 'nan'], ['--timeout', '-1'], ['--arena-mib', '0']]:
            with self.assertRaises(SystemExit):
                runner.parser().parse_args(['--cuda', '/cuda', '--hip', '/hip'] + args)


if __name__ == '__main__':
    unittest.main()
