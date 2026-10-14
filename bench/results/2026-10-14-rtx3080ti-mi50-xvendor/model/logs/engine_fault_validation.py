#!/usr/bin/env python3
"""Replay the mixed 4K fixture, injecting only PID-scoped process faults."""
import hashlib,json,os,selectors,signal,subprocess,time
from pathlib import Path
ROOT=Path.cwd()
BASE=ROOT/'bench-local/xvendor-2026-10-14'
prior=json.loads((BASE/'mixed-cuda-hip-context-ladder.json').read_text())
argv=prior['argv']
tokens=(BASE/'iq3s-context-fixtures/iq3s-q4-context150000-prompt4096.tokens').read_text().strip()
payload=('GEN 40 '+tokens+'\n').encode()
env=os.environ.copy()
env.update(STRATA_STAGE_LD_LIBRARY_PATH='/opt/rocm/lib',STRATA_ROUTE_TAIL_SKIP='0',STRATA_PREFILL_CPU_SHARE='0',STRATA_LOOKAHEAD='0',STRATA_NAN_GUARD='1',CUDA_VISIBLE_DEVICES='0')
def state(pid):
 try:
  s=Path(f'/proc/{pid}/stat').read_text();return s[s.rfind(')')+2:].split()[0]
 except FileNotFoundError:return None
def children(pid):
 try:return list(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
 except FileNotFoundError:return []
def active(pid):return state(pid) not in (None,'Z','X')
results=[]
for mode in ('kill-worker','kill-parent'):
 prefix=BASE/('engine-fault-'+mode)
 record={'case':mode,'argv':argv,'requested_context':150000,'requested_kv':'q4_0','prompt_tokens':4096,'fault_delay_after_gen_seconds':5,'status':'starting'}
 proc=None;worker=None;buffer=b'';lines=[];selector=selectors.DefaultSelector();pending=memoryview(b'')
 out=open(str(prefix)+'.stdout.log','wb');err=open(str(prefix)+'.stderr.log','wb')
 def pump(seconds):
  global buffer,pending
  for key,events in selector.select(seconds):
   if key.fileobj==proc.stdin:
    n=os.write(proc.stdin.fileno(),pending);pending=pending[n:]
    if not pending:selector.unregister(proc.stdin)
   else:
    chunk=os.read(proc.stdout.fileno(),65536)
    if not chunk:selector.unregister(proc.stdout);continue
    out.write(chunk);out.flush();buffer+=chunk
    while b'\n' in buffer:
     line,buffer=buffer.split(b'\n',1);lines.append(line.decode(errors='replace'))
 try:
  proc=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,env=env,start_new_session=True,bufsize=0)
  record['parent_pid']=proc.pid
  os.set_blocking(proc.stdout.fileno(),False);os.set_blocking(proc.stdin.fileno(),False)
  selector.register(proc.stdout,selectors.EVENT_READ)
  deadline=time.monotonic()+300
  while not any(x.startswith('READY ') for x in lines):
   if time.monotonic()>deadline:raise RuntimeError('startup timeout')
   if proc.poll() is not None:raise RuntimeError('engine exited before READY')
   pump(.25)
  cs=children(proc.pid)
  if len(cs)!=1:raise RuntimeError('expected exactly one direct HIP worker: '+str(cs))
  worker=cs[0];record['worker_pid']=worker
  cmd=Path(f'/proc/{worker}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  if 'build-xvendor-hip/strata' not in cmd:raise RuntimeError('worker identity mismatch: '+cmd)
  record['worker_cmdline']=cmd
  pending=memoryview(payload);selector.register(proc.stdin,selectors.EVENT_WRITE)
  while pending:
   if proc.poll() is not None:raise RuntimeError('engine exited while sending GEN')
   pump(.25)
  gen_sent=time.monotonic();record['gen_payload_sha256']=hashlib.sha256(payload).hexdigest()
  while time.monotonic()-gen_sent<5:pump(.25)
  before=[x for x in lines if x.startswith(('T ','DONE '))]
  if before:raise RuntimeError('request completed or generated tokens before fault')
  record['protocol_before_fault']=before;record['parent_state_before_fault']=state(proc.pid);record['worker_state_before_fault']=state(worker)
  if not active(proc.pid) or not active(worker):raise RuntimeError('fault target not active')
  fault=time.monotonic();victim=worker if mode=='kill-worker' else proc.pid
  os.kill(victim,signal.SIGKILL);record['killed_pid']=victim
  deadline=fault+30
  while time.monotonic()<deadline and (proc.poll() is None or active(worker)):pump(.1)
  proc.poll();record['parent_exit_code']=proc.returncode;record['elapsed_after_fault_seconds']=round(time.monotonic()-fault,6);record['worker_state_after_fault']=state(worker)
  record['protocol_after_fault']=[x for x in lines if x.startswith(('T ','DONE ','ERR '))]
  record['done_after_fault']=any(x.startswith('DONE ') for x in record['protocol_after_fault'])
  record['parent_exited']=proc.returncode is not None;record['worker_active']=active(worker)
  passed=(record['parent_exited'] and proc.returncode!=0 and not record['worker_active'] and not record['done_after_fault'] and not any(x.startswith('T ') for x in record['protocol_after_fault']))
  if mode=='kill-parent':passed=passed and proc.returncode==-signal.SIGKILL
  record['status']='pass' if passed else 'fail'
 except Exception as e:
  record['status']='error';record['error']=str(e)
 finally:
  record['cleanup_needed']=bool(proc and (proc.poll() is None or (worker and active(worker))))
  if record['cleanup_needed']:
   try:os.killpg(proc.pid,signal.SIGKILL)
   except ProcessLookupError:pass
  if proc:
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:pass
  out.close();err.close();selector.close()
  Path(str(prefix)+'.json').write_text(json.dumps(record,indent=2)+'\n')
  results.append(record)
  print(mode,record['status'],'elapsed',record.get('elapsed_after_fault_seconds'),'parent',record.get('parent_exit_code'),'worker',record.get('worker_state_after_fault'),'cleanup',record['cleanup_needed'],flush=True)
(BASE/'engine-fault-results.json').write_text(json.dumps(results,indent=2)+'\n')
raise SystemExit(0 if all(x['status']=='pass' for x in results) else 1)
