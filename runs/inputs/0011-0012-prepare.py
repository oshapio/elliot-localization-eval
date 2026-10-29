"""Prepare runs 0011/0012 from the pinned full manifest and audited baseline outputs.

Run-specific data selection recipe; refuses existing output directories.
"""
from pathlib import Path
import json,hashlib,sys,shutil
repo=Path('/leonardo_work/EUHPC_E07_125/nas/elliot-localization-eval-full-preflight')
sys.path.insert(0,str(repo))
from evals_localization.tasks import build_prompt_for_entry
base=Path('/leonardo_scratch/large/userexternal/auselis0/elliot-localization/evaluations')
full=base/'full/full_v022_matched_step77938_228k_32gpu_debug_20260915/inputs/manifest.jsonl'
oldroot=Path('/leonardo_work/EUHPC_E07_125/nas/elliot-vl-eval-vllm/outputs')
for number,name,oldname,family,configprefix in [('0011','qwen35-full-backfill','full_qwen35_vllm_20260905','qwen','0009'),('0012','rex-full-backfill','full_rex_vllm_20260905','rex-omni','0010')]:
 out=base/(number+'-'+name);assert not out.exists(),out
 out.mkdir();(out/'inputs').mkdir();(out/'logs').mkdir();(out/'source').symlink_to(repo,target_is_directory=True)
 config=next((repo/'runs/inputs').glob(configprefix+'-*.json'));shutil.copyfile(config,out/'inputs/config.json')
 oldfile=oldroot/oldname/'predictions/predictions.jsonl';kept=set();oldtotal=oldbad=0;oldhash=hashlib.sha256()
 with oldfile.open('rb') as f,(out/'inputs/reused_predictions.jsonl').open('wb') as dest:
  for line in f:
   oldhash.update(line);r=json.loads(line);oldtotal+=1
   assert r['max_new_tokens']==4096 and not r.get('retried_after_token_cap')
   assert build_prompt_for_entry(r,family)==r['prompt'],r['sample_id']
   if r['status']=='error':oldbad+=1;continue
   assert r['sample_id'] not in kept;kept.add(r['sample_id']);dest.write(line)
 fullhash=hashlib.sha256();backhash=hashlib.sha256();fullids=set();count=0;groups=set()
 with full.open('rb') as f,(out/'inputs/manifest.jsonl').open('wb') as dest:
  for line in f:
   fullhash.update(line);r=json.loads(line);sid=r['sample_id'];assert sid not in fullids;fullids.add(sid)
   if sid not in kept:
    dest.write(line);backhash.update(line);count+=1;groups.add((r['benchmark_dataset'],r['benchmark_protocol']))
 assert len(fullids)==228133 and fullhash.hexdigest()=='08a7f24ddb75d87285813366e54deb6872f84b4250cce61ce588f73b72d2f960'
 assert kept<=fullids and len(kept)+count==228133
 assert count==(48174 if number=='0011' else 48196)
 audit={'old_predictions':str(oldfile),'old_sha256':oldhash.hexdigest(),'old_rows':oldtotal,'reused_rows':len(kept),'excluded_error_rows':oldbad,'backfill_rows':count,'backfill_groups':len(groups),'backfill_sha256':backhash.hexdigest(),'full_manifest':str(full),'full_manifest_sha256':fullhash.hexdigest(),'groups':sorted(groups),'source_commit':'f4cb946','config_source':str(config),'compatibility':'all reused rendered prompts match; 4096 output cap, no token-cap retry; same historical weights/tokenizer verified separately; legacy metadata retained'}
 (out/'preparation.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps({'run':number,'rows':count,'groups':len(groups),'reused':len(kept),'sha':backhash.hexdigest()}),flush=True)
