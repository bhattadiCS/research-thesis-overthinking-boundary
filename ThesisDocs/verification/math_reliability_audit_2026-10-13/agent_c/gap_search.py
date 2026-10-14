"""Inventory local/package evidence without loading pickles or fitting models."""
from pathlib import Path
import collections,hashlib,io,json,pickletools,re,subprocess,zipfile
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[3];DEST=OUT/'gap_zip_inventories';DEST.mkdir(exist_ok=True)
patterns={
 'n1':r'(?i)(algov2|n1n4|n1_|loco|lomo|predicted.?delta|held.?out.?pred|ridge)',
 'stack':r'(?i)(blackwell|stack|hybrid|advanced_tournament|oof_prediction|checkpoint)',
 'replay':r'(?i)(active_stopping|offline_replay|stopping_simulation|replay.*result|replay.*paired)',
}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,indent=2,default=str)+'\n',encoding='utf-8')
def pickle_metadata(raw):
 # Disassembly only. Never pickle.load/torch.load or evaluate GLOBAL/REDUCE.
 strings=[];globals=[];errors=[];opcounts=collections.Counter()
 try:
  for op,arg,pos in pickletools.genops(raw):
   opcounts[op.name]+=1
   if op.name in ('UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','STRING') and isinstance(arg,str) and len(arg)<=300:strings.append((pos,arg))
   if op.name in ('GLOBAL','STACK_GLOBAL'):globals.append((pos,str(arg)))
 except Exception as e:errors.append(type(e).__name__+': '+str(e))
 return dict(strings=strings,globals=globals,opcode_counts=dict(opcounts),errors=errors,method='pickletools.genops disassembly only; no object construction/execution')
def checkpoint_metadata(raw):
 if zipfile.is_zipfile(io.BytesIO(raw)):
  with zipfile.ZipFile(io.BytesIO(raw)) as z:
   entries=[dict(name=x.filename,bytes=x.file_size,crc=x.CRC) for x in z.infolist()]
   data={n:pickle_metadata(z.read(n)) for n in z.namelist() if n.endswith('data.pkl')}
   return dict(container='ZIP',entries=entries,pickle_disassembly=data)
 return dict(container='legacy pickle',pickle_disassembly=pickle_metadata(raw))
def npz_metadata(raw):
 data=[]
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  for n in z.namelist():
   if not n.endswith('.npy'):continue
   with z.open(n) as f:
    version=np.lib.format.read_magic(f)
    if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
    else:shape,fortran,dtype=np.lib.format.read_array_header_2_0(f)
   data.append(dict(name=n,shape=shape,dtype=str(dtype),object_dtype=bool(dtype.hasobject)))
 return data

allpaths=(OUT/'gap_workspace_paths.txt').read_text(encoding='utf-16').splitlines()
archives=[ROOT/p for p in allpaths if p.lower().endswith(('.zip','.7z','.tar','.tar.gz','.tgz'))]
result=dict(search_roots=['entire workspace via rg --files --hidden --no-ignore'],query_patterns=patterns,commands=["rg --files --hidden --no-ignore -g '*.zip' -g '*.7z' -g '*.tar' -g '*.tar.gz' -g '*.tgz' -g '*.pth' -g '*.pt' -g '*active*stopping*' -g '*n1*' -g '*blackwell*' -g '*replay*' -g '*hybrid*' -g '*stack*'",'read ZIP central directories; inspect checkpoint data.pkl via pickletools only; inspect NPZ NPY headers with allow_pickle never invoked'],archives=[],local_checkpoints=[],local_arrays=[],package_payload_candidates=[])
seen={}
for p in archives:
 rel=p.relative_to(ROOT).as_posix();record=dict(path=rel,bytes=p.stat().st_size,sha256=sha(p))
 if not zipfile.is_zipfile(p):record['scope']='non-ZIP tool/environment archive; no research-package suffix';result['archives'].append(record);continue
 with zipfile.ZipFile(p) as z:
  entries=[dict(path=x.filename,bytes=x.file_size,compressed_bytes=x.compress_size,crc32=x.CRC) for x in z.infolist()]
  file=DEST/(str(len(result['archives'])).zfill(2)+'_'+p.stem+'.json');file.write_text(json.dumps(entries,indent=2)+'\n',encoding='utf-8')
  matches={key:[x for x in entries if re.search(pattern,x['path'])] for key,pattern in patterns.items()};record.update(entry_count=len(entries),inventory=file.relative_to(ROOT).as_posix(),pattern_matches=matches,nested_zip_entries=[x for x in entries if x['path'].lower().endswith('.zip')])
  if record['sha256'] in seen:record['identical_to']=seen[record['sha256']]
  else:
   seen[record['sha256']]=rel
   for x in entries:
    n=x['path'];low=n.lower()
    if low.endswith(('.pth','.pt')) and re.search(patterns['stack'],n):
     data=z.read(n);result['package_payload_candidates'].append(dict(archive=rel,entry=n,sha256=hashlib.sha256(data).hexdigest(),metadata=checkpoint_metadata(data)))
    elif low.endswith('.npz') and any(re.search(q,n) for q in patterns.values()):
     data=z.read(n);result['package_payload_candidates'].append(dict(archive=rel,entry=n,sha256=hashlib.sha256(data).hexdigest(),npz_headers=npz_metadata(data)))
 result['archives'].append(record);print('archive',rel,record.get('entry_count'),flush=True)
# All retained checkpoint paths discovered throughout the workspace.
for s in allpaths:
 p=ROOT/s
 if p.suffix.lower() not in ('.pth','.pt') or '.venv/' in p.as_posix():continue
 raw=p.read_bytes();result['local_checkpoints'].append(dict(path=p.relative_to(ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),metadata=checkpoint_metadata(raw)));print('checkpoint',s,flush=True)
# Header-only pass over every NPZ/NPY/Pickle path in research/evidence roots.
candidatepaths=(OUT/'gap_candidate_paths.txt').read_text(encoding='utf-16').splitlines()
for s in candidatepaths:
 p=ROOT/s
 if p.suffix.lower()=='.npz':
  raw=p.read_bytes();result['local_arrays'].append(dict(path=p.relative_to(ROOT).as_posix(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),npz_headers=npz_metadata(raw)))
dump('gap_search.json',result)
print('done archives',len(result['archives']),'checkpoints',len(result['local_checkpoints']),'array files',len(result['local_arrays']),'package candidates',len(result['package_payload_candidates']),flush=True)
