from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.analysis.final_figures.rq4 import (
    ROOT, FIG1, FIG2, FIG3, IDS, MOUT, METRIC, STAT, R5_CAND, R5_METRIC,
    PAIRS, HASHES, bootstrap_values, utility_values, r5_values,
    generate_all, validate_all, validate_sources,
)

STARTING_COMMIT='bc85bd0dfe2bf71e361eff2cdf780ccf98ec1f48'

def _hashes(root:Path)->dict[str,str]:
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}

def test_rq4_hashes_pairing_and_g3_configuration()->None:
    assert validate_sources(ROOT)==HASHES
    pairs=pd.read_csv(ROOT/PAIRS,dtype=str,keep_default_na=False)
    assert len(pairs)==6000
    assert pairs.same_candidate_membership.eq('true').all()
    assert pairs.only_candidate_order_differs.eq('true').all()
    assert pairs.first_positive_position.eq('1').all()
    values=utility_values(ROOT)
    assert len(values)==90
    assert values.groupby('metric_id').size().eq(30).all()
    assert values.planned_applicable_users.eq('200').all()
    assert np.allclose(values.first_value.astype(float)-values.random_value.astype(float),values.first_minus_random_delta.astype(float),atol=1e-12)
    assert values.condition_contract.eq('SAME_CANDIDATE_SET_FIRST_TARGET_INFORMED_ORACLE').all()


def test_rq4_frozen_ci_decisions_and_unique_exception()->None:
    values=bootstrap_values(ROOT)
    source=pd.read_csv(ROOT/STAT,dtype=str,keep_default_na=False)
    assert len(values)==len(source)==90
    assert {x:int(values.decision.eq(x).sum()) for x in ['POSITIVE','NEGATIVE','INCONCLUSIVE']}=={'POSITIVE':85,'NEGATIVE':1,'INCONCLUSIVE':4}
    assert values.orientation.eq('FIRST_MINUS_RANDOM').all()
    assert values.n_pairs.eq('200').all()
    assert not values.duplicated(['dataset','model','prompt_strategy','metric_id']).any()
    exception=values.loc[values.decision.eq('NEGATIVE')]
    assert len(exception)==1
    row=exception.iloc[0]
    assert [row.dataset,row.model,row.prompt_strategy,row.metric_id]==['ml1m','mistral:7b-instruct','LLMRANK','hr_at_5']
    assert np.allclose([float(row.random_mean),float(row.first_mean),float(row.first_minus_random_delta),float(row.ci_low),float(row.ci_high)],[.410,.250,-.160,-.250,-.075],atol=1e-12)
    joined=values.merge(source,on='result_id',validate='one_to_one')
    assert len(joined)==90
    assert joined.first_minus_random_delta_x.eq(joined.first_minus_random_delta_y).all()
    assert joined.ci_low_x.eq(joined.ci_low_y).all()
    assert joined.ci_high_x.eq(joined.ci_high_y).all()


def test_rq4_r5b_corrected_supporting_only()->None:
    values=r5_values(ROOT)
    assert len(values)==12
    assert values.role.eq('SUPPORTING_R5B_CORRECTED_NO_G3_POOLING').all()
    assert set(values.dataset)=={'Beauty','Sports','LastFM'}
    assert int((values.loc[values.metric_id.eq('hr_at_5'),'canddif_value'].astype(float)>0).sum())==5
    assert int((values.loc[values.metric_id.eq('ndcg_at_5'),'canddif_value'].astype(float)>0).sum())==6
    assert values.n_users.eq('100').all()
    assert not values.duplicated(['dataset','model_name','metric_id']).any()
    assert (ROOT/R5_CAND).is_file() and (ROOT/R5_METRIC).is_file()


def test_rq4_deterministic_regeneration_and_portability(tmp_path:Path)->None:
    a,b=tmp_path/'a',tmp_path/'b'
    generate_all(ROOT,artifact_root=a,created_from_commit='a'*40)
    generate_all(ROOT,artifact_root=b,created_from_commit='a'*40)
    assert _hashes(a)==_hashes(b)
    assert len(_hashes(a))==15
    assert {m['figure_id'] for m in validate_all(ROOT,artifact_root=a)}==set(IDS)
    for figure_id in IDS:
        m=json.loads((a/MOUT/f'{figure_id}_manifest.json').read_text(encoding='utf-8'))
        assert m['created_from_commit']=='a'*40
        assert m['display_contract']['no_r5b_g3_pooling'] is True
        assert m['display_contract']['no_deployability_claim'] is True
        assert m['evidence_role']==('SUPPORTING_R5B_TARGETED' if figure_id==FIG3 else 'PRIMARY_G3_SYSTEMATIC')
        for key in ['canonical_source_artifacts','figure_values_csv','generator','output_png','output_pdf','output_svg']:
            paths=m[key] if isinstance(m[key],list) else [m[key]]
            assert all(not Path(p).is_absolute() and ':' not in p for p in paths)


def test_rq4_committed_outputs_match_manifests()->None:
    manifests=validate_all(ROOT)
    assert {m['figure_id'] for m in manifests}=={FIG1,FIG2,FIG3}
    assert all(m['created_from_commit']==STARTING_COMMIT for m in manifests)
