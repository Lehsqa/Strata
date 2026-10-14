#!/usr/bin/env python3
"""Run token files through one local Strata --serve process; write JSON and two logs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('exe', 'pack', 'native', 'ple-gguf', 'profile', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--tokens-file', action='append', required=True,
                        help='Repeat for sequential GEN requests in the same process.')
    parser.add_argument('--backend', choices=('cuda', 'hip'), default='cuda')
    parser.add_argument('--worker')
    parser.add_argument('--worker-backend', choices=('cuda', 'hip'), default='hip')
    parser.add_argument('--split', type=int, default=12)
    parser.add_argument('--context', type=int, default=150000)
    parser.add_argument('--kv', default='q4_0')
    parser.add_argument('--prefill', type=int, default=1024)
    parser.add_argument('--cache', default='auto')
    parser.add_argument('--max-new', type=int, default=40)
    parser.add_argument('--timeout', type=float, default=1800,
                        help='Maximum seconds for the complete engine run, including loading.')
    parser.add_argument('--engine-arg', action='append', default=[],
                        help='Append one engine argument; use --engine-arg=--flag for flags.')
    args = parser.parse_args()
    if (min(args.context, args.prefill, args.max_new, args.split) <= 0
            or not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('context, prefill, max-new, split and timeout must be positive')
    if args.cache != 'auto' and not re.fullmatch(r'\d+', args.cache):
        parser.error('cache must be auto or a nonnegative integer')

    requests = []
    try:
        for filename in args.tokens_file:
            content = Path(filename).read_text()
            fields = re.split(r'[\s,]+', content.strip())
            if not fields or any(not re.fullmatch(r'\d+', field) for field in fields):
                raise ValueError(f'{filename}: expected nonnegative token IDs separated by commas or whitespace')
            ids = [int(field) for field in fields]
            tokens = ','.join(map(str, ids))
            requests.append({'tokens_file': filename, 'token_count': len(ids),
                             'token_sha256': hashlib.sha256(tokens.encode('ascii')).hexdigest(),
                             'output_ids': [], 'status': 'pending', '_tokens': tokens})
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    argv = [args.exe, '--pack', args.pack, '--native', args.native,
            '--ple-gguf', args.ple_gguf, '--gpu', '0', '--serve', '--spec', '8',
            '--greedy', '--mmap-experts', '--no-prefill-borrow',
            '--expert-profile', args.profile, '--expert-cache', args.cache,
            '--kv', args.kv, '--max-context', str(args.context),
            '--prefill', str(args.prefill), '--adapt-every', '0', '--adapt-swaps', '0',
            '--prompt-cache', '0', '--conversation-cache-mib', '0',
            '--conversation-cache-slots', '0', '--suffix-draft', '0', '--stats']
    if args.worker:
        argv += ['--stage-remote', f'{args.worker_backend}:{args.worker}',
                 '--stage-remote-devices', '0', '--layer-split', str(args.split)]
    argv += args.engine_arg
    env = os.environ.copy()
    if args.backend == 'hip':
        env.pop('CUDA_VISIBLE_DEVICES', None)
        env.pop('CUDA_DEVICE_ORDER', None)
        env['HIP_VISIBLE_DEVICES'] = '0'
    else:
        env['CUDA_VISIBLE_DEVICES'] = '0'
    environment_keys = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'HIP_VISIBLE_DEVICES',
                        'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL',
                        'LD_LIBRARY_PATH', 'STRATA_STAGE_LD_LIBRARY_PATH',
                        'STRATA_ROUTE_TAIL_SKIP', 'HSA_OVERRIDE_GFX_VERSION',
                        'HIP_LAUNCH_BLOCKING', 'CUDA_LAUNCH_BLOCKING')
    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    result = {'started_at': datetime.now(timezone.utc).isoformat(), 'cwd': os.getcwd(),
              'argv': argv, 'environment': {key: env[key] for key in environment_keys if key in env},
              'requested_context': args.context, 'requested_kv': args.kv,
              'timeout_seconds': args.timeout, 'token_hash_encoding': 'ASCII comma-separated canonical IDs',
              'info': {}, 'info_lines': [], 'ready': None, 'requests': requests,
              'stdout_log': str(prefix) + '.stdout.log', 'stderr_log': str(prefix) + '.stderr.log',
              'status': 'starting', 'process_exit_code': None}
    started = time.monotonic()
    deadline = started + args.timeout
    proc = None
    current = None
    buffer = b''
    done_names = ('generated', 'prompt_tokens', 'prefill_ms', 'decode_ms', 'finish',
                  'drafts_accepted', 'drafts_offered', 'reused')

    with open(result['stdout_log'], 'wb') as stdout_log, \
            open(result['stderr_log'], 'wb') as stderr_log, selectors.DefaultSelector() as selector:
        def read_until(accept, payload=None, allow_eof=False):
            nonlocal buffer
            pending = memoryview(payload or b'')
            if pending:
                selector.register(proc.stdin, selectors.EVENT_WRITE)
            try:
                while True:
                    while b'\n' in buffer:
                        raw, buffer = buffer.split(b'\n', 1)
                        line = raw.decode('utf-8', errors='replace').rstrip('\r')
                        if line.startswith('INFO '):
                            result['info_lines'].append(line)
                            result['info'].update(field.split('=', 1) for field in line.split()[1:] if '=' in field)
                        if line.startswith('ERR '):
                            raise RuntimeError(line)
                        if accept(line):
                            return line
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(f'engine exceeded {args.timeout:g} seconds')
                    for key, _ in selector.select(remaining):
                        if key.fileobj is proc.stdin:
                            written = os.write(proc.stdin.fileno(), pending)
                            pending = pending[written:]
                            if not pending:
                                selector.unregister(proc.stdin)
                        else:
                            chunk = os.read(proc.stdout.fileno(), 65536)
                            if not chunk:
                                if allow_eof and not pending:
                                    return None
                                raise RuntimeError('engine stdout closed before the expected protocol line')
                            stdout_log.write(chunk)
                            stdout_log.flush()
                            buffer += chunk
            finally:
                if pending:
                    selector.unregister(proc.stdin)

        try:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=stderr_log, env=env, start_new_session=True, bufsize=0)
            os.set_blocking(proc.stdin.fileno(), False)
            os.set_blocking(proc.stdout.fileno(), False)
            selector.register(proc.stdout, selectors.EVENT_READ)
            result['ready'] = read_until(lambda line: line.startswith('READY '))
            result['startup_seconds'] = time.monotonic() - started
            for current in requests:
                request_started = time.monotonic()
                current['status'] = 'running'

                def request_line(line):
                    if line.startswith('T '):
                        current['output_ids'].append(int(line.split()[1]))
                        if 'first_token_seconds' not in current:
                            current['first_token_seconds'] = time.monotonic() - request_started
                    return line.startswith('DONE ')

                done = read_until(request_line,
                                  f"GEN {args.max_new} {current['_tokens']}\n".encode('ascii'))
                fields = done.split()[1:]
                if len(fields) < 5:
                    raise ValueError(f'incomplete DONE line: {done}')
                current['done_line'] = done
                current['done_fields'] = fields
                current['done'] = {name: (value if name == 'finish' else
                                         float(value) if name.endswith('_ms') else int(value))
                                   for name, value in zip(done_names, fields)}
                current['duration_seconds'] = time.monotonic() - request_started
                current['status'] = ('ok' if current['done']['finish'] in ('stop', 'length')
                                     else current['done']['finish'])
                if current['status'] != 'ok':
                    raise RuntimeError(f"engine ended generation with {current['done']['finish']}")
                if current['done']['generated'] != len(current['output_ids']):
                    raise RuntimeError('DONE generated count differs from received token count')
                if current['done'].get('reused', 0) != 0:
                    raise RuntimeError('engine reused a prefix despite caches being disabled')
            current = None
            read_until(lambda line: False, b'QUIT\n', allow_eof=True)
            proc.wait(timeout=max(0.001, deadline - time.monotonic()))
            if proc.returncode:
                raise RuntimeError(f'engine exited with code {proc.returncode}')
            result['status'] = 'ok'
        except (OSError, RuntimeError, ValueError, TimeoutError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            result['status'] = ('timeout' if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired))
                                else 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error')
            result['error'] = str(exc) or type(exc).__name__
            if current is not None:
                current['duration_seconds'] = time.monotonic() - request_started
                current['status'] = result['status']
        finally:
            if proc is not None:
                if result['status'] != 'ok':
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                    # Workers can outlive the parent; kill the remaining process group as well.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                proc.wait()
                result['process_exit_code'] = proc.returncode
                proc.stdin.close()
                proc.stdout.close()
    for request in requests:
        del request['_tokens']
    result['duration_seconds'] = time.monotonic() - started
    Path(str(prefix) + '.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"{result['status']}: {prefix}.json")
    return 0 if result['status'] == 'ok' else 1


if __name__ == '__main__':
    sys.exit(main())
