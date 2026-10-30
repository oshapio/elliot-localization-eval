#!/usr/bin/env python3
"""Render checked-in metric/provenance snapshots; never launches evaluations."""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def specifications(models):
    rows = []
    def add(label, group, metric):
        rows.append({'label': label, 'group': group, 'metric': metric,
                     'key': group + '|' + metric})
    for dataset in ['COCO', 'Dense200', 'LVIS', 'VisDrone']:
        add(dataset + ' / boxes', 'box/' + dataset, 'F1@0.5')
    add('ODinW13 overall', 'suite/ODinW13', 'mean_dataset_mAP')
    add('HumanRef / boxes', 'box/HumanRef', 'F1@0.5')
    add('HumanRef / boxes (IoU sweep)', 'box/HumanRef', 'F1@mean')
    for dataset in ['RefCOCO+_testA', 'RefCOCO+_testB', 'RefCOCO+_val',
                    'RefCOCO_testA', 'RefCOCO_testB', 'RefCOCO_val',
                    'RefCOCOg_test', 'RefCOCOg_val']:
        add(dataset, 'box/' + dataset, 'Acc@IoU0.5')
    for dataset in ['DocLayNet', 'M6Doc', 'HierText', 'IC15', 'SROIE', 'TotalText']:
        add(dataset + ' / boxes', 'box/' + dataset, 'F1@0.5')
    for label, dataset in [('ScreenSpot v2', 'screenspot_v2'), ('ScreenSpot Pro', 'sspro')]:
        add(label, 'gui/' + dataset, 'accuracy')
    for dataset in ['COCO', 'Dense200', 'HumanRef', 'LVIS', 'RefCOCOg_test',
                    'RefCOCOg_val', 'VisDrone']:
        add(dataset + ' / pointing', 'point/' + dataset, 'F1@0.5')
    for split in ['Location', 'Placement', 'Unseen']:
        add('RefSpatialBench / ' + split, 'point/RefSpatialBench_' + split, 'accuracy')
    add('RefSpatialBench overall', 'suite/RefSpatialBench', 'query_weighted_accuracy')
    odinw = sorted({key.split('|')[0] for model in models for key in model['metrics']
                    if key.startswith('box/ODinW13_') and key.endswith('|mAP')})
    for group in odinw:
        add(group.removeprefix('box/'), group, 'mAP')
    return rows


def render(data):
    models = data['models']
    assert len({m['id'] for m in models}) == len(models)
    specs = specifications(models)
    for model in models:
        for spec in specs:
            if spec['key'] in model['metrics']:
                value = model['metrics'][spec['key']]
                assert type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, (model['id'], spec['key'], value)
    with (ROOT / 'comparison.csv').open('w', newline='') as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(['benchmark', 'metric', *[m['label'] for m in models]])
        for spec in specs:
            writer.writerow([spec['label'], spec['metric'], *[
                format(100 * m['metrics'][spec['key']], '.6f')
                if spec['key'] in m['metrics'] else '' for m in models]])
    payload = json.dumps({'models': models, 'rows': specs}, separators=(',', ':')).replace('<', '\\u003c')
    template = '''<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Localization checkpoint comparison</title>
<style>
body{font:15px system-ui;margin:24px;color:#18212f;background:#f6f7fa}h1{font-size:25px}p{max-width:1000px;line-height:1.5}
button,input{font:inherit;padding:7px;border:1px solid #bac3d0;border-radius:5px;background:white}button{cursor:pointer}
#models{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0}label{background:white;padding:7px;border-radius:5px;border:1px solid #dde2ea}
.scroll{overflow:auto;max-height:73vh;border:1px solid #cbd2de;background:white}table{border-collapse:separate;border-spacing:0;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:10px;border-bottom:1px solid #e2e6ed;text-align:right;white-space:nowrap}thead th{position:sticky;top:0;background:#e9eef6;z-index:3;max-width:230px;white-space:normal}
th:first-child,td:first-child{position:sticky;left:0;text-align:left;background:#fff;z-index:2;min-width:230px}thead th:first-child{z-index:4;background:#e9eef6}
small{display:block;color:#5e6878;font-weight:400}.missing{color:#8b94a3}details{margin:12px 0}pre{white-space:pre-wrap;background:white;padding:12px;font-size:12px}
</style>
<h1>Localization checkpoint comparison</h1>
<p>Values are percentages (raw fractions ×100). Each cell uses the named metric. Blank cells mean the metric has not been imported, never zero. Archived numbers are not a controlled ablation: prompts, formats and training branches can differ.</p>
<p><b>HumanRef:</b> F1@0.5 and mean F1 over IoU 0.5–0.95 are separate rows. RefSpatial overall is weighted over 277 queries, not an unweighted mean of its three splits.</p>
<input id="search" placeholder="Filter benchmarks…" aria-label="Filter benchmarks">
<button id="recent">Elliot 7145 + baselines</button> <button id="all">All checkpoints</button>
<button id="csv">Download CSV</button> · <a href="README.md">Run flow and provenance notes</a>
<div id="models"></div><div class="scroll"><table id="table"></table></div>
<h2>Source and compatibility notes</h2><div id="sources"></div>
<script id="data" type="application/json">PAYLOAD</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent), chosen=new Set(data.models.map(m=>m.id));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function draw(){
 const ms=data.models.filter(m=>chosen.has(m.id)),q=document.getElementById('search').value.toLowerCase();
 document.getElementById('table').innerHTML='<thead><tr><th>Benchmark / metric</th>'+ms.map(m=>'<th>'+esc(m.label)+'<small>'+esc(m.kind)+'</small></th>').join('')+'</tr></thead><tbody>'+data.rows.filter(r=>(r.label+' '+r.metric).toLowerCase().includes(q)).map(r=>'<tr><td>'+esc(r.label)+'<small>'+esc(r.metric)+'</small></td>'+ms.map(m=>Object.hasOwn(m.metrics,r.key)?'<td title="Raw fraction: '+m.metrics[r.key]+'">'+(100*m.metrics[r.key]).toFixed(2)+'</td>':'<td class="missing">—</td>').join('')+'</tr>').join('')+'</tbody>';
 document.querySelectorAll('#models input').forEach(e=>e.checked=chosen.has(e.value));
}
document.getElementById('models').innerHTML=data.models.map(m=>'<label><input type="checkbox" value="'+esc(m.id)+'" checked> '+esc(m.label)+'</label>').join('');
document.getElementById('models').addEventListener('change',e=>{if(e.target.checked)chosen.add(e.target.value);else chosen.delete(e.target.value);draw()});
document.getElementById('search').oninput=draw;
document.getElementById('csv').onclick=()=>{
 const cell=v=>'"'+String(v).replaceAll('"','""')+'"';
 const lines=[['benchmark','metric',...data.models.map(m=>m.label)],...data.rows.map(r=>[r.label,r.metric,...data.models.map(m=>Object.hasOwn(m.metrics,r.key)?(100*m.metrics[r.key]).toFixed(6):'')])];
 const url=URL.createObjectURL(new Blob([lines.map(r=>r.map(cell).join(',')).join('\\n')],{type:'text/csv;charset=utf-8'}));
 const a=document.createElement('a');a.href=url;a.download='comparison.csv';a.click();URL.revokeObjectURL(url);
};
document.getElementById('recent').onclick=()=>{chosen.clear();['worker-safe-local7145','qwen35-2b','rex-omni'].forEach(id=>chosen.add(id));draw()};
document.getElementById('all').onclick=()=>{data.models.forEach(m=>chosen.add(m.id));draw()};
document.getElementById('sources').innerHTML=data.models.map(m=>'<details><summary>'+esc(m.label)+'</summary><p>'+esc(m.audit_status)+'</p><pre>'+esc(JSON.stringify({coverage:m.coverage,source:m.source},null,2))+'</pre></details>').join('');draw();
</script>'''
    (ROOT / 'comparison.html').write_text(template.replace('PAYLOAD', payload))
    write_readme(data, specs)
    return specs


def write_readme(data, specs):
    byid = {m['id']: m for m in data['models']}
    current = [byid[k] for k in ['worker-safe-local7145', 'qwen35-2b', 'rex-omni']]
    selected = [r for r in specs if r['group'].startswith('box/RefCOCO')
                or r['group'] == 'box/HumanRef' or 'RefSpatialBench' in r['group']
                or r['group'] == 'suite/ODinW13']
    def value(model, key):
        return f"{model['metrics'][key] * 100:.2f}" if key in model['metrics'] else '—'
    text = """# Localization results and checkpoint comparison

[Open the standalone comparison](comparison.html) · [Download the full CSV](comparison.csv)

The HTML has model toggles, a benchmark filter and source details. Download/open it
locally to use the interactive controls; GitHub displays HTML as source. The CSV
has all imported checkpoint columns side by side, including individual ODinW datasets.

## Current comparison

All values are raw fractions ×100. Each row names its metric. HumanRef F1@0.5
and mean F1 over IoU 0.5–0.95 are different rows. RefSpatial overall is weighted
across 277 queries (100 Location, 100 Placement, 77 Unseen).

| Benchmark | Metric | Elliot SFT7145 | Qwen3.5-2B | Rex-Omni |
| --- | --- | ---: | ---: | ---: |
"""
    for row in selected:
        text += '| ' + ' | '.join([row['label'], row['metric'],
                                    *[value(m, row['key']) for m in current]]) + ' |\n'
    text += """
Elliot here is the **worker-safe local-7145 pre-RL parent**, not the older local-500
smoke target. Qwen is a completed combined 228,133-query / 44-group result.
Rex's selected rows were verified from completed scoring groups; its merged
prediction artifact has 19 error rows and five wrong-dimension box answers.
Rex’s login-node full rescore was killed with exit 137; its full scoring is
incomplete. The selected rows shown above were completed before that failure.
Neither baseline is represented as a fresh generation of every historical row.

## Archived full checkpoint evaluations

These 13 local score files each contain 228,133 samples and 44 groups. Their
SHA-256 values match archived publication receipts. This verifies the stored
numbers, not every historical inference command or per-query protocol. Labels
identify distinct artifacts; row order is not a claim of linear ancestry. The
full CSV/HTML includes all their detailed scores, not only this overview.

| Checkpoint artifact | HumanRef F1@0.5 | RefCOCO val Acc@0.5 | RefSpatial overall | ODinW13 mean mAP | Prompt protocol in imported receipt |
| --- | ---: | ---: | ---: | ---: | --- |
"""
    for model in data['models']:
        if model['kind'] != 'archived-full':
            continue
        protocol = ', '.join(model['source']['metadata'].get('prompt_protocol', [])) or 'Not imported'
        keys = ['box/HumanRef|F1@0.5', 'box/RefCOCO_val|Acc@IoU0.5',
                'suite/RefSpatialBench|query_weighted_accuracy', 'suite/ODinW13|mean_dataset_mAP']
        text += '| ' + ' | '.join([model['label'], *[value(model, k) for k in keys], protocol]) + ' |\n'
    text += """
## What ran, in plain terms

1. A numbered record under [`runs/`](../runs/README.md) identifies the model,
   committed code, config, manifest, command, environment, job and output paths.
2. `orchestration/vllm_full.sbatch` allocates inference shards through
   `orchestration/run_shard.sh`. The wrapper calls `evaluator/run.py`; it does not
   implement a second evaluator. `ADAPTER` and `PROMPT_FAMILY` select model execution
   and the task templates. Ground-truth coordinates/counts are not used in prompts.
3. Each adapter loads that model's tokenizer/processor and chat template.
   Elliot uses its custom vLLM integration and selected REF/BOX or mixed decoder.
   Rex uses its vendor protocol and 0–999 coordinates; Qwen uses its native
   box/point response format and 0–1000 coordinates. Decoders scale to original
   image pixels, not resized input pixels. Raw text and token IDs remain saved.
4. For the baseline coverage repair, 48,174 IDs were missing from each old
   179,959-query result. We checked all old prompts and model assets for compatibility.
   Qwen reuses 179,959 rows and generates 48,174. Rex reuses 179,937 successful rows
   and generates 48,196, including replacements for 22 old error rows. Historical
   row metadata remains intact, rather than being relabelled as new inference.
5. `evaluator/merge.py` checks exact IDs/uniqueness against the full manifest.
   `evaluator/score.py` scores each group. The complete baseline result is under
   `combined/...` (or Rex's separate `combined-repair-v1/...`), not the subset scores.
6. Rex emitted points for five box questions. The scoring repair counts each as
   a miss without removing the query, changing its answer or selecting a later box.
   Nineteen separate error rows remain disclosed. Recovery used saved predictions
   on the login node; no new inference was needed.

## Commands, evidence and remaining gaps

- [Qwen full coverage and command](../runs/0011-qwen35-full-backfill.md)
- [Rex full coverage, command and scoring recovery](../runs/0012-rex-full-backfill.md)
- [All numbered runs, including smoke and failed attempts](../runs/README.md)
- [Machine-readable values and provenance](data.json)

This snapshot covers **13 archived full checkpoint artifacts plus selected Qwen/Rex
readbacks**. It is not every saved training checkpoint. The seven-query v022-77938,
family-balanced-119000, cardinality-repair-500 and recovery-10000 smoke results are
not inserted as full accuracy columns. Their other historical full results have
not been imported here. RL dependencies/results remain separately unresolved.
Historical commands absent from the imported receipt are not invented.

Remote SSH access expired during this import. Baseline raw full-score files and
additional remote-only checkpoint results need the renewed connection; missing
cells stay blank. Per-model source paths, available checkpoint/protocol hashes,
audit limits and archive checksums are in data.json and the HTML source details.

Regenerate the CSV, HTML and this page from the checked-in snapshot:

```bash
python3 results/build.py
```

On the original workspace, `python3 results/build.py --verify-archives` additionally
checks all 13 external archived score files against their recorded checksums.
"""
    (ROOT / 'README.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-archives', action='store_true')
    args = parser.parse_args()
    data = json.loads((ROOT / 'data.json').read_text())
    if args.verify_archives:
        for model in data['models']:
            source = model['source']
            if model['kind'] != 'archived-full':
                continue
            path = Path(source['local_archive'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source['score_sha256'], path
            original = json.loads(path.read_text())
            for key, value in model['metrics'].items():
                group, metric = key.split('|', 1)
                if group == 'suite/ODinW13':
                    parts = [v['metrics']['mAP'] for k, v in original['datasets'].items()
                             if k.startswith('box/ODinW13_')]
                    assert len(parts) == 13
                    actual = sum(parts) / 13
                elif group == 'suite/RefSpatialBench':
                    parts = [original['datasets']['point/RefSpatialBench_' + split]
                             for split in ['Location', 'Placement', 'Unseen']]
                    actual = sum(v['metrics']['accuracy'] * v['diagnostics']['samples']
                                 for v in parts) / sum(v['diagnostics']['samples'] for v in parts)
                else:
                    actual = original['datasets'][group]['metrics'][metric]
                assert abs(actual - value) < 1e-12, (model['id'], key)
    specs = render(data)
    print(f"Rendered {len(data['models'])} model columns × {len(specs)} metric rows.")


if __name__ == '__main__':
    main()
