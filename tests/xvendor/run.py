#!/usr/bin/env python3
"""Manual Linux GPU feasibility launcher; CPU tests never clear the hardware gate."""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time


def vendor_env(vendor, device, hip_libdir='', source=None):
    env = dict(os.environ if source is None else source)
    for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES'):
        env.pop(key, None)
    if vendor == 'cuda':
        env.update(CUDA_VISIBLE_DEVICES=device, CUDA_DEVICE_ORDER='PCI_BUS_ID')
    else:
        env['HIP_VISIBLE_DEVICES'] = device
        if hip_libdir:
            env['LD_LIBRARY_PATH'] = hip_libdir + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env


def run_pair(commands, environments, fd, timeout):
    children, logs = [], []
    result = 1
    try:
        for command, env in zip(commands, environments):
            log = tempfile.TemporaryFile()
            logs.append(log)
            children.append(subprocess.Popen(command, env=env, pass_fds=(fd,), stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + timeout
        while True:
            codes = [child.poll() for child in children]
            if any(code not in (None, 0) for code in codes):
                print('FAIL: child exited; stopping peer', file=sys.stderr)
                break
            if all(code == 0 for code in codes):
                result = 0
                break
            if time.monotonic() >= deadline:
                print('FAIL: deadline exceeded; killing both GPU processes', file=sys.stderr)
                break
            time.sleep(.01)
    except (OSError, KeyboardInterrupt) as error:
        print(f'FAIL: {error}', file=sys.stderr)
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
        for child in children:
            child.wait()
        for index, log in enumerate(logs):
            log.seek(0, os.SEEK_END)
            size = log.tell()
            log.seek(max(0, size - 65536))
            print(f'child {index} log ({size} bytes; last 64 KiB):', flush=True)
            print(log.read().decode('utf-8', errors='replace'), end='', flush=True)
            log.close()
    return result


def bounded_int(low, high):
    def parse(value):
        number = int(value)
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f'must be {low}..{high}')
        return number
    return parse


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--cuda', required=True, help='CUDA handoff executable')
    cli.add_argument('--hip', required=True, help='HIP handoff executable')
    cli.add_argument('--main', choices=['cuda', 'hip'], default='cuda')
    cli.add_argument('--cuda-device', type=bounded_int(0, 1023), default=0)
    cli.add_argument('--hip-device', type=bounded_int(0, 1023), default=0)
    cli.add_argument('--hip-libdir', default='')
    cli.add_argument('--mode', choices=['spin', 'futex', 'doorbell', 'large-registration'], default='futex')
    cli.add_argument('--rounds', type=bounded_int(1, (1 << 24) - 32), default=10000)
    cli.add_argument('--registration-mib', type=bounded_int(1, 262144), default=32768)
    cli.add_argument('--arena-mib', type=bounded_int(1, 262144), default=16384)
    cli.add_argument('--delay-us', type=bounded_int(0, 1000000), default=50)
    cli.add_argument('--timeout', type=bounded_int(1, 86400), default=180)
    return cli


def main():
    args = parser().parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Linux memfd required')
    executables = {'cuda': str(Path(args.cuda).resolve()), 'hip': str(Path(args.hip).resolve())}
    envs = {vendor: vendor_env(vendor, str(getattr(args, vendor + '_device')), args.hip_libdir)
            for vendor in executables}
    common = ['--mode', args.mode, '--rounds', str(args.rounds), '--registration-mib', str(args.registration_mib),
              '--arena-mib', str(args.arena_mib), '--delay-us', str(args.delay_us),
              '--timeout-ms', str(min(args.timeout * 1000, 86400000))]
    sizes = []
    for vendor in executables:
        query = subprocess.run([executables[vendor], '--region-bytes'] + common, env=envs[vendor],
                               capture_output=True, text=True, timeout=15, check=True)
        sizes.append(int(query.stdout.strip()))
    if sizes[0] != sizes[1] or sizes[0] < 4096 or sizes[0] % os.sysconf('SC_PAGE_SIZE'):
        raise RuntimeError(f'incompatible region sizes: {sizes}')
    print(f'host={platform.platform()} mode={args.mode} main={args.main} rounds={args.rounds} bytes={sizes[0]}', flush=True)
    for vendor in executables:
        print(f'{vendor}: executable={executables[vendor]} device={getattr(args, vendor + "_device")} '
              f'LD_LIBRARY_PATH={envs[vendor].get("LD_LIBRARY_PATH", "")}', flush=True)
    fd = os.memfd_create('strata-xvendor', os.MFD_CLOEXEC)
    try:
        os.ftruncate(fd, sizes[0])
        vendors = [args.main, 'hip' if args.main == 'cuda' else 'cuda']
        commands = [[executables[vendor], '--role', role, '--fd', str(fd)] + common
                    for vendor, role in zip(vendors, ['main', 'worker'])]
        return run_pair(commands, [envs[vendor] for vendor in vendors], fd, args.timeout)
    finally:
        os.close(fd)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f'FAIL: {error}', file=sys.stderr)
        sys.exit(1)
