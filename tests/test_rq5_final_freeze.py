from __future__ import annotations
import ast
import hashlib
import json
import itertools
from pathlib import Path
import pandas as pd
import pytest
import yaml
from src.evaluation.rq5_freeze_metrics import parse_ranking, mostpop, beyond_utility
from src.analysis.rq5_final_freeze import Evidence, NS, validate_request_pair, verify_hashes
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("text",['', '{}', '{"ranked_candidate_indices":[1,1,2,3,4,5,6,7,8,9]}', '{"ranked_candidate_indices":[1,2]}', '{"ranked_candidate_indices":[true,2,3,4,5,6,7,8,9,10]}', '{"ranked_candidate_indices":[1,2,3,4,5,6,7,8,9,21]}'])
def test_invalid_zero(text):
    result=parse_ranking(text,[str(i) for i in range(20)],'0')
    assert result['strict_valid']==0
    assert result['hr_at_5']==result['ndcg_at_5']==0
    assert result['top5']==[]

def test_known_utility_and_empty_predictions():
    result=parse_ranking(json.dumps({'ranked_candidate_indices':list(range(1,11))}),[str(i) for i in range(20)],'1')
    assert result['hr_at_5']==1
    assert result['ndcg_at_5']==pytest.approx(1/1.584962500721156)
    assert parse_ranking('{}',[],'1')['hr_at_5']==0

def test_aplt_terminal_invalid_regression():
    # Regression: old code used retained parsed IDs from an invalid response.
    parsed={'strict_valid':0,'hr_at_5':0.,'top5':['1','2','3','4','5']}
    actual=beyond_utility(parsed,['1','2','3','4','5'],'1',{'1':1,'2':10},{'1'})
    assert actual['aplt_at_5']==actual['serendipity_at_5']==0.

def test_aplt_valid_known_value_and_duplicate_items():
    actual=beyond_utility({'strict_valid':1,'hr_at_5':1.,'top5':['1','1','2','3','4']},['1','2','3','4','5','6'],'1',{'1':1,'2':10,'3':9,'4':8,'5':7,'6':6},{'1'})
    assert actual['aplt_at_5']==.2
    assert actual['serendipity_at_5']==1.

def test_missing_popularity_is_not_zero():
    with pytest.raises(ValueError):
        beyond_utility({'strict_valid':1,'hr_at_5':1.,'top5':[]},['1'],'1',{},set())

def test_mostpop_ties_are_canonical_and_order_invariant():
    ids=['2','10','3','01'];counts={'2':4,'10':4,'3':8,'01':4}
    for order in itertools.permutations(ids):
        assert mostpop(order,counts)==['3','01','10','2']
    assert mostpop(['2','2','10'],{'2':0,'10':0})==['10','2']
    assert mostpop([],counts)==[]

def test_corrected_request_identity_and_intervention():
    e=Evidence(ROOT)
    try:
        u=e.table('submission/results/tables/rq5_corrected_full_request_universe.csv',True)
        base=u.loc[u.condition.eq('history_only')].set_index(['dataset','model_name','user_id'])
        for row in u.loc[~u.condition.eq('history_only')].to_dict('records'):
            control=base.loc[(row['dataset'],row['model_name'],row['user_id'])].to_dict()
            control.update(dataset=row['dataset'],model_name=row['model_name'],user_id=row['user_id'])
            validate_request_pair(control,row)
        bad=dict(row);bad['seed']='1'
        with pytest.raises(AssertionError):validate_request_pair(control,bad)
    finally:e.close()

def test_frozen_accounting_and_statistics():
    for family,n,v,b in [('corrected',4000,3996,32),('systematic',9000,8997,60),('history_v2',4800,4794,64)]:
        per=pd.read_csv(ROOT/NS/family/'per_user.csv');stats=pd.read_csv(ROOT/NS/family/'bootstrap_results.csv',keep_default_na=False)
        assert len(per)==n and per.strict_valid.sum()==v and len(stats)==b
        assert stats.paired_n.eq(200).all() and stats.bootstrap_resamples.eq(10000).all() and stats.bootstrap_seed.eq(20260901).all()
        assert stats.p_value.eq('').all()
        for metric in ['hr_at_5','ndcg_at_5']+(['aplt_at_5','serendipity_at_5'] if family=='history_v2' else []):assert per.loc[per.strict_valid.eq(0),metric].eq(0).all()

def test_history_v2_migration_and_serendipity_source_hash():
    m=json.loads((ROOT/NS/'history_v2/migration_manifest.json').read_text(encoding="utf-8"))
    e=Evidence(ROOT)
    try:assert hashlib.sha256(e.read(m['source'])).hexdigest()==m['source_sha256']
    finally:e.close()
    assert m['raw_inference_unchanged'] and len(m['aplt_affected_requests'])==3
    assert m['old_hit_count']==47 and m['audit_observed_historical_hit_count']==56
    assert m['new_per_user_sha256']==hashlib.sha256((ROOT/NS/'history_v2/per_user.csv').read_bytes()).hexdigest()
    p=pd.read_csv(ROOT/NS/'history_v2/per_user.csv')
    assert (p.serendipity_at_5==p.hr_at_5*(1-p.mostpop_hit_at_5)).all()

def test_systematic_profile_reuse_audit():
    p=pd.read_csv(ROOT/NS/'rq5_primary_systematic_profile_reuse_audit.csv',keep_default_na=False)
    assert len(p)==1000 and p.reused.astype(str).str.lower().eq('true').sum()==400
    assert p.status.eq('PASS').all()
    assert p.loc[p.reused.astype(str).str.lower().eq('true'),'source_sha256'].str.len().eq(64).all()

def test_figure_source_hashes():
    figures=json.loads((ROOT/NS/'figures/figure_index.json').read_text(encoding="utf-8"))
    assert len(figures)==5
    for m in figures:
        for rel,h in {**m['source_artifacts'],**m['figure_sha256'],m['source_csv']:m['source_csv_sha256']}.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h
        rows=pd.read_csv(ROOT/m['source_csv'])
        for value in rows.source_artifacts:
            for rel,h in json.loads(value).items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h

def test_no_foreground_f6l_statistics_or_network_execution():
    source=(ROOT/'src/analysis/rq5_final_freeze.py').read_text(encoding="utf-8")
    assert 'rq5_profile_statistical_results.csv' not in source
    tree=ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or '']
            assert not any(n.startswith(('requests','httpx','openai','ollama','urllib')) for n in names)
    synthesis=json.loads((ROOT/NS/'scientific_synthesis.json').read_text(encoding="utf-8"))
    assert synthesis['semantic_preference_preservation_validated'] is False
    assert synthesis['independent_replications'] is False
    assert synthesis['authority']['f6l_statistics']=='FORENSIC'

def test_operational_paths_are_repository_relative():
    import re
    for rel in ['configs/rq5_final_freeze.yaml','src/analysis/rq5_final_freeze.py','src/evaluation/rq5_freeze_metrics.py']:
        assert not re.search(r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/]',(ROOT/rel).read_text(encoding="utf-8"))
    for family in ['corrected','systematic','history_v2']:
        assert not re.search(r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/]',(ROOT/NS/family/'request_universe.csv').read_text(encoding="utf-8"))

def test_freeze_manifest_completeness():
    manifest=json.loads((ROOT/NS/'freeze_manifest.json').read_text(encoding="utf-8"))
    expected=['configs/rq5_final_freeze.yaml','src/analysis/rq5_final_freeze.py','src/evaluation/rq5_freeze_metrics.py','tests/test_rq5_final_freeze.py',NS+'/package_inventory.csv']
    assert all(p in manifest['artifacts'] for p in expected)
    assert manifest['base_commit']!=''
    assert manifest['freeze_commit']['resolution']=='commit_provenance.json'
    assert manifest['external_bundles_required']==[]
    for name in ['corrected','systematic','history_v2','r5d']:
        assert NS+'/'+name+'/metrics.csv' in manifest['artifacts']
    assert verify_hashes(ROOT)['status']=='PASS'
