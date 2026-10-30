"""Validate committed evidence without pretending to retrain external-data models."""
from pathlib import Path
import ast, csv, json, math
ROOT=Path(__file__).resolve().parents[1]
for p in ROOT.rglob('*.py'):
    if not any(x in p.parts for x in ['.git','.venv','node_modules']): ast.parse(p.read_text())
for p in ROOT.rglob('*.ipynb'):
    n=json.loads(p.read_text()); assert n['nbformat']==4
    for c in n['cells']:
        assert c['cell_type'] in ['code','markdown','raw']
        assert not any(o.get('output_type')=='error' for o in c.get('outputs',[])), str(p)
def table(path):
    with (ROOT/path).open(newline='',encoding='latin1' if str(path).startswith('data/') else 'utf-8') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-6): assert math.isclose(float(a),float(b),abs_tol=tol,rel_tol=tol),(a,b)
v=json.loads((ROOT/'outputs/validation.json').read_text())
assert sum(int(r['observations']) for r in table('outputs/indicator_coverage.csv'))==v['observations']
assert v['duplicate_grain_rows']==0 and not v['foreign_key_violations']

f=json.loads((ROOT/'outputs/structure_findings.json').read_text())
nuclear=f['patterns']['nuclear']
assert 0<nuclear['countries_with_nuclear']<=nuclear['countries_with_data']
assert sum(nuclear['by_region'].values())==nuclear['countries_with_nuclear']
auc=nuclear['gdp_predicts_presence'];assert 0<=auc['ci_low']<=auc['auc']<=auc['ci_high']<=1
assert len(f['patterns']['renewable_consumption']['median_by_gdp_quintile'])==5
assert f['nmf']['seed_stability_rank2']['min']>0.99
for rank,row in f['nmf']['null_comparison'].items(): assert 0<row['null_min']<=row['null_mean']<=row['null_max']<1
errors=f['nmf']['relative_error_by_rank'];assert all(errors[str(k)]>errors[str(k+1)] for k in range(1,5))
assert f['parafac']['baselines']['country_indicator_mean']<f['parafac']['baselines']['indicator_mean']
for row in f['parafac']['parafac'].values(): assert row['min']<=row['mean']<=row['max']

print('Committed evidence and syntax checks passed; see VALIDATION.md for rerun scope.')
