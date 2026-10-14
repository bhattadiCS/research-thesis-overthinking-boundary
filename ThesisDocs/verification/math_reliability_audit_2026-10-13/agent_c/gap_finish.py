"""Finish read-only gap classification from recorded inventories and source."""
from pathlib import Path
import collections,hashlib,json,pickletools,re,subprocess,zipfile
import pandas as pd
out=Path(__file__).resolve().parent;root=out.parents[3];p=out/'gap_search.json';d=json.loads(p.read_text(encoding='utf-8'))
def sha(b):return hashlib.sha256(b).hexdigest()
keypat=re.compile(r'(?i)(loco|lomo|oof_control|oof_hybrid|held.?out.?pred|predicted.?delta|stop_step|stopped_tokens|stopped_correct|delta_full|dU_profile|coefficient|intercept)')
d['package_content_metadata']=[];seen=set()
for a in d['archives']:
 if 'entry_count' not in a or a['sha256'] in seen:continue
 seen.add(a['sha256']);path=root/a['path'];record=dict(archive=a['path'],json_key_hits=[],csv_headers=[],binary_array_entries=[])
 with zipfile.ZipFile(path) as z:
  for info in z.infolist():
   name=info.filename;low=name.lower()
   if low.endswith(('.npz','.npy','.pkl','.pickle','.pth','.pt')):record['binary_array_entries'].append(name)
   if low.endswith(('.csv','.tsv')):
    with z.open(name) as f:header=f.readline().decode('utf-8-sig',errors='replace').strip()
    record['csv_headers'].append(dict(entry=name,header=header))
   if low.endswith('.json'):
    try:j=json.loads(z.read(name).decode('utf-8-sig'))
    except Exception:continue
    hits=[]
    def walk(v,prefix=''):
     if isinstance(v,dict):
      for k,x in v.items():
       field=prefix+'/'+str(k)
       if keypat.search(str(k)):hits.append(dict(key=field,type=type(x).__name__,length=len(x) if isinstance(x,(list,dict)) else None,scalar=x if isinstance(x,(str,int,float,bool)) else None))
       walk(x,field)
     elif isinstance(v,list):
      for i,x in enumerate(v):
       if isinstance(x,(dict,list)):walk(x,prefix+'/'+str(i))
    walk(j)
    if hits:record['json_key_hits'].append(dict(entry=name,hits=hits))
  # Match published package manifests to actual archive contents, not merely names.
  manifest=path.with_name(path.stem+'_manifest.json')
  if manifest.exists():
   m=json.loads(manifest.read_text(encoding='utf-8-sig'));files=m.get('files',[])
   if isinstance(files,list) and all(isinstance(r,dict) and 'path' in r for r in files):
    expected={r['path']:r for r in files};actual={x.filename for x in z.infolist() if not x.is_dir()};mismatches=[]
    for name,r in expected.items():
     if name in actual and 'sha256' in r:
      h=sha(z.read(name))
      if h!=r['sha256']:mismatches.append(dict(entry=name,expected=r['sha256'],actual=h))
    record['manifest_check']=dict(path=manifest.relative_to(root).as_posix(),sha256=sha(manifest.read_bytes()),declared_files=len(expected),actual_files=len(actual),missing=sorted(set(expected)-actual),unlisted=sorted(actual-set(expected)),payload_sha256_mismatches=mismatches)
 d['package_content_metadata'].append(record)
# Disassemble prepared data pickle metadata only. No reducers/globals executed.
d['prepared_pickle_metadata']=[]
for s in (out/'gap_candidate_paths.txt').read_text(encoding='utf-16').splitlines():
 path=root/s
 if path.suffix.lower() not in ('.pkl','.pickle'):continue
 raw=path.read_bytes();strings=[];globals=[];error=None
 try:
  for op,arg,pos in pickletools.genops(raw):
   if op.name in ('BINUNICODE','SHORT_BINUNICODE','UNICODE','BINUNICODE8') and isinstance(arg,str) and len(arg)<250 and keypat.search(arg):strings.append(dict(position=pos,value=arg))
   if op.name in ('GLOBAL','STACK_GLOBAL'):globals.append(dict(position=pos,value=str(arg)))
 except Exception as e:error=repr(e)
 d['prepared_pickle_metadata'].append(dict(path=path.relative_to(root).as_posix(),bytes=len(raw),sha256=sha(raw),relevant_strings=strings,globals=globals,error=error,method='pickletools.genops only; no object construction'))
d['npz_summary']=dict(files=len(d['local_arrays']),distinct_sha256=len({r['sha256'] for r in d['local_arrays']}),target_key_hits=[dict(path=r['path'],arrays=[x for x in r['npz_headers'] if keypat.search(x['name'])]) for r in d['local_arrays'] if any(keypat.search(x['name']) for x in r['npz_headers'])])
def source_record(path,needles,revision=None):
 raw=subprocess.check_output(['git','show',revision+':'+path],cwd=root) if revision else (root/path).read_bytes();lines=raw.decode('utf-8-sig').splitlines()
 return dict(path=path,revision=revision,sha256=sha(raw),anchors=[dict(needle=n,line=next((i+1 for i,s in enumerate(lines) if n in s),None)) for n in needles])
d['source_save_logic']=[source_record('research/algorithm_v2_experiments.py',['def held_out_dU','return tot','LOCO predicted-delta dU']),source_record('tools/compute_progress_report_review_metrics.py',['def n1_contributions','algorithm_v2_normalized_effects.csv','def write_offline_replay_output']),source_record('research/run_ultimate_blackwell_5day_tournament.py',['def save_checkpoint_and_git_push','oof_control =','oof_hybrid =','json.dump(report, f','blackwell_tournament_report.md']),source_record('research/run_active_stopping.py',['FEATURES =','return {','clf.intercept_[0]:.4f','REPORT_OUTPUT.write_text'],revision='5955600539f6a9a27b417872b86d3da184128935'),source_record('research/run_advanced_hyper_optimization.py',['checkpoint_data =','"oof_predictions": oof_predictions','torch.save(checkpoint_data']),source_record('research/run_ultimate_multi_day_tournament.py',['def save_fold_checkpoint','"oof_q": oof_q'])]
cache=root/'research/outputs/experiments_v2/algov2_cache';d['n1_cache_schema']=dict(files=len(list(cache.glob('*.json'))),key_sets=sorted({tuple(sorted(json.loads(x.read_text(encoding='utf-8')).keys())) for x in cache.glob('*.json')}))
d['rounded_7b_assessment']=json.loads((out/'rounded_replay_bounds.json').read_text(encoding='utf-8'))
d['gap_conclusions']=[dict(id='C-018',outcome='NOT ESTABLISHED',scope='Workspace named-candidate inventory; published ZIP complete entry/payload-manifest inventories;3970 NPZ header dictionaries;prepared pickle metadata;52-cell cache schema and historical save logic',finding='No exact N1 per-heldout selected thresholds/predictions or per-cell contributions found. Cache retains features, targets and profiles; source computes heldout predictions in memory and prints only aggregate totals. Exact effect/CIs require refit or new full-precision per-heldout evidence.'),dict(id='C-014',outcome='NOT ESTABLISHED',finding='No exact Blackwell oof_control/oof_hybrid per-row arrays found in available package inventories/local NPZ keys/checkpoint metadata. Blackwell save function only invokes git; main writes aggregate JSON/Markdown. Advanced checkpoint does retain other experiment OOF arrays (202440 float64 entries for first key), not the144440-row Blackwell stack; ultimate checkpoint is another linear/moe experiment. No arbitrary pickle execution.'),dict(id='C-015',outcome='NOT ESTABLISHED',finding='Original7B report retains rounded4-decimal heads and complete raw feature order,no scaler. Raw baseline827804 tokens/1058 correct is independently verified. Rounding uncertainty leaves97/1500 stop decisions uncertified; rounded point tokens377774 differ from headline377960, with conservative possible token range374891..380891. This is insufficient precision, not contradiction of full-precision historical endpoint. No full-precision fitted parameters or per-run original stops found in available package/local named inventories.')]
d['limitations']=['This is an inventory of accessible local workspace/package evidence, not proof that missing files never existed or are absent from external/remote/unreachable Git storage.','NPZ headers and pickle opcode metadata inspected without object deserialization. Exact contents of arbitrary binary blobs not interpreted as unknown custom formats.','Source-only reading of historical scripts: never invoked collection/training/git-write helper functions. No model refit or new-generation execution.','No automatic identification by merely similar AUC or array size: advanced/ultimate,peer,portable,heuristic replay outputs remain distinct experiments.','Rounded coefficient interval analysis assumes decimal table values are nearest4-decimal representations of originally applied coefficients and historical raw source features are exact. Dependencies across coefficient effects make stated aggregate bounds conservative.']
p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
summary=['# Final evidence-gap search','', 'Read-only follow-up,2026-10-13. No model fit,generation,regrading or pickle object loading.','',f"Searched20 workspace archives(19 ZIPs and one unrelated environment tar),five retained checkpoint paths,3970 NPZ files({d['npz_summary']['distinct_sha256']} distinct hashes),13 prepared pickle metadata files,and52 N1 cache records. The ZIP entry inventories are saved under gap_zip_inventories/. gap_search.json records every path,hash,query pattern,CSV header,matching JSON key,checkpoint opcode metadata and source-save anchor.",'','## Archive inventories','', '| Archive | Entries | Declared-manifest check |','| --- | ---: | --- |']
for r in d['archives']:
 if 'entry_count' not in r:continue
 c=next((x.get('manifest_check') for x in d['package_content_metadata'] if x['archive']==r['path']),None)
 check='duplicate package hash' if 'identical_to' in r else ('not a research manifest' if not c else f"{c['declared_files']} declared;{len(c['missing'])} missing;{len(c['unlisted'])} unlisted;{len(c['payload_sha256_mismatches'])} payload hash mismatches")
 summary.append(f"| {r['path']} | {r['entry_count']} | {check} |")
summary+=['','The v4–v7 publication packages and audited research supplement contain no NPZ,NPY,PKL,PTH or PT payloads. Their saved peer/selected-answer CSVs are inventoried; they are different endpoints. The NeurIPS supplement contains paired portable/heuristic replay records,not the original7B fitted replay. No nested ZIPs were found.','', '## Exact gap scopes','']
for g in d['gap_conclusions']:summary += [f"**{g['id']}: {g['outcome']}.** {g['finding']}",'']
summary+=['## Rounded7B precision assessment','', 'The retained report provides all three heads at four decimals and the historical source provides the complete unscaled feature order. Each coefficient/intercept rounding cell has half-width0.00005. For a raw feature vector x augmented with an intercept1,the unknown logit is within0.00005*sum(abs(x)) of the reported logit. Sigmoid monotonicity gives q/repair/corruption intervals. The drift is decreasing in q and corruption and increasing in repair,so opposite interval corners provide valid drift bounds. A stop is certified only if all earlier possible decisions and that stop have definite signs.','', 'This certifies1403 trajectory stop decisions;97 remain ambiguous. Reported-rounded weights yield377774 tokens and963 correct;full-precision historical headline is377960 tokens and963 correct. Conservative rounding-compatible aggregate bounds are374891–380891 tokens and958–964 correct. Full coefficient precision or original stopped rows is still needed to certify the exact stopped endpoint. The baseline endpoint is separately reproduced directly from all15000 raw rows:827804 tokens and1058/1500 correct.','', '## Retained checkpoints are different experiments','', 'The33,641,015-byte advanced checkpoint has SHA2567405818cdec82813b38317547475bd4d5f5f3d25bf67ca2f695ff54550c4c036. Pickle disassembly shows fold3,smoke_testFalse,and five named OOF arrays for linear,dynamic,BiGRU,TCN and BetaLikelihood models;the first array has202440 float64 entries. It also contains per-policy results dataframes. This is the advanced-hyper-optimization experiment,not the144440-row Blackwell stack or the1500-run7B replay. Current ultimate checkpoint keys instead contain linear/moe q,repair and corruption dictionaries. All these objects were inspected by pickletools only;no GLOBAL/REDUCE opcode was executed.','', '## Limits','']
summary += ['- '+s for s in d['limitations']]
(out/'gap_search.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
print('Finished',len(d['package_content_metadata']),'unique ZIP content scans',len(d['prepared_pickle_metadata']),'prepared pickle scans','npz target-key hits',len(d['npz_summary']['target_key_hits']))
