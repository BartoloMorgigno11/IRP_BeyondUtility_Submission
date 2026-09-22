"""Final RQ4 figures from frozen G3 and corrected R5B evidence."""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.analysis.final_figures.common import (FigureValidationError,git_head,repository_path,sha256_file,validate_figure_manifest,write_dataframe,write_json)

METRIC='results/tables/g3_candidate_position_metric_results.csv'
CANDDIF='results/tables/g3_candidate_position_canddif.csv'
STAT='results/tables/g3_candidate_position_statistical_results.csv'
SYN='results/tables/g3_candidate_position_synthesis_statistical_comparisons.csv'
BREAK='results/tables/g3_candidate_position_synthesis_decision_breakdowns.csv'
RELIABILITY='results/tables/g3_candidate_position_reliability.csv'
PAIRS='results/tables/g3_candidate_position_pair_integrity.csv'
R5_CAND='results/tables/reproduction_r5b_corrected_canddif_results.csv'
R5_METRIC='results/tables/reproduction_r5b_corrected_metrics_by_condition.csv'
G3_METRIC_MAN='results/manifests/g3_candidate_position_metrics_freeze.json'
G3_STAT_MAN='results/manifests/g3_candidate_position_statistical_analysis_manifest.json'
G3_SYN_MAN='results/manifests/g3_candidate_position_scientific_synthesis_manifest.json'
R5_MAN='results/manifests/reproduction_r5b_corrected_freeze.json'
G3_SOURCES=[METRIC,CANDDIF,STAT,SYN,BREAK,RELIABILITY,PAIRS,G3_METRIC_MAN,G3_STAT_MAN,G3_SYN_MAN]
R5_SOURCES=[R5_CAND,R5_METRIC,R5_MAN]
HASHES={
 METRIC:'ce005175b0e6222ca488080506cd8a8e13d1314e136e0421c782b88f14e4e25e',
 CANDDIF:'6349802ba933d5ab0218fd72268e4aa0c71563f3a8c1cc91d8a0273ee9e59f80',
 STAT:'2b2e1fa3d661fa1c4c2e81d56350548eeb33d1fb9594ef5b924b548af7a555a3',
 SYN:'f24af8cc57f3403e26453a6ea9d636b061c4375cbeeb3c9f9e53f6ecd643f19d',
 BREAK:'869daa07c5325bb1108ba5016712db979d5de7d2453f206f3200a92885f6cc07',
 RELIABILITY:'b31de0ea8c3b788091eeb7893af7a795029ddb36059d87a13c88f1257da7d537',
 PAIRS:'9de7febe5fe472b57f383f59c11a83f49353772482223360aa42fe0e0be75255',
 R5_CAND:'65d026ff2f3b225daccdf649bf232c831e42eac437799279dcfd647f545fe3de',
 R5_METRIC:'d91f5e476132cc6e73a5895241eb55cbde7b631347d19d73b57698604c978ca0',
}
FIG1='fig_rq4_g3_first_random_utility_pairs'
FIG2='fig_rq4_g3_first_random_bootstrap_deltas'
FIG3='fig_rq4_targeted_r5b_canddif_recovery'
IDS=[FIG1,FIG2,FIG3]
METRICS=['HR@5','NDCG@5','MRR']; STAT_METRICS=['hr_at_5','ndcg_at_5','mrr']
DATASETS=['beauty','sports','ml1m','lastfm','foodcom']
MODELS=['qwen2:7b-instruct','mistral:7b-instruct','gemma2:9b']; PROMPTS=['BASE','LLMRANK']
DECISION_MAP={'CI_SUPPORTED_POSITIVE':'POSITIVE','CI_SUPPORTED_NEGATIVE':'NEGATIVE','NON_SIGNIFICANT_CI_INCLUDES_ZERO':'INCONCLUSIVE'}
OUT='results/final_figures/rq4'; MOUT='results/manifests/final_figures'; GENERATOR='src/analysis/final_figures/rq4.py'
FIXED_TIME=datetime(2026,9,21,tzinfo=timezone.utc)


def validate_sources(root:Path=ROOT)->dict[str,str]:
    """Validate each frozen scientific input against its recorded hash."""
    for rel,digest in HASHES.items():
        p=repository_path(root,rel)
        if not p.is_file() or sha256_file(p)!=digest: raise FigureValidationError(f'RQ4 source hash mismatch: {rel}')
    metric=json.loads(repository_path(root,G3_METRIC_MAN).read_text(encoding='utf-8'))
    stat=json.loads(repository_path(root,G3_STAT_MAN).read_text(encoding='utf-8'))
    synthesis=json.loads(repository_path(root,G3_SYN_MAN).read_text(encoding='utf-8'))
    r5=json.loads(repository_path(root,R5_MAN).read_text(encoding='utf-8'))
    for rel in [METRIC,CANDDIF,RELIABILITY]:
        if metric['output_hashes'][rel].lower()!=HASHES[rel]: raise FigureValidationError('G3 metric manifest mismatch')
    if stat['output_hashes'][STAT].lower()!=HASHES[STAT] or synthesis['output_hashes'][SYN].lower()!=HASHES[SYN] or synthesis['output_hashes'][BREAK].lower()!=HASHES[BREAK]:
        raise FigureValidationError('G3 inferential provenance mismatch')
    if metric['input_hashes_observed'][PAIRS].lower()!=HASHES[PAIRS]: raise FigureValidationError('G3 pair provenance mismatch')
    for rel in [R5_CAND,R5_METRIC]:
        if r5['result_table_hashes'][rel].lower()!=HASHES[rel]: raise FigureValidationError('R5B corrected freeze mismatch')
    if r5['corrected_status_counts']!={'backend_incomplete':0,'invalid':0,'parse_failure':0,'requests_accounted':1200,'requests_expected':1200,'valid':1200,'wrong_length':0}:
        raise FigureValidationError('R5B corrected accounting changed')
    for rel in [G3_METRIC_MAN,G3_STAT_MAN,G3_SYN_MAN,R5_MAN]:
        HASHES[rel]=sha256_file(repository_path(root,rel))
    return dict(HASHES)


def _g3(root:Path)->tuple[pd.DataFrame,pd.DataFrame]:
    validate_sources(root)
    p=pd.read_csv(repository_path(root,PAIRS),dtype=str,keep_default_na=False)
    if len(p)!=6000 or any(not p[k].eq('true').all() for k in ['has_exactly_first_and_random','same_candidate_membership','same_history','same_target','candidate_order_differs','first_positive_position_ok','only_candidate_order_differs']):
        raise FigureValidationError('G3 6000-pair membership/ordering contract failed')
    if not p.first_positive_position.eq('1').all(): raise FigureValidationError('FIRST is not target-informed position 1')
    rel=pd.read_csv(repository_path(root,RELIABILITY),dtype=str,keep_default_na=False)
    overall=rel.loc[rel.scope.eq('overall')]
    if len(overall)!=1 or overall.iloc[0][['planned_requests','strict_valid','invalid','backend_errors']].tolist()!=['12000','11834','166','0']:
        raise FigureValidationError('G3 request accounting changed')
    m=pd.read_csv(repository_path(root,METRIC),dtype=str,keep_default_na=False)
    if len(m)!=180 or set(m.metric_id)!=set(METRICS) or set(m.candidate_order_condition)!={'FIRST','RANDOM'}: raise FigureValidationError('G3 utility universe changed')
    if not m.planned_applicable_users.eq('200').all() or not m.backend_error_users.eq('0').all(): raise FigureValidationError('G3 utility denominator changed')
    keys=['dataset','model','prompt_strategy','metric_id']
    if m.duplicated(keys+['candidate_order_condition']).any(): raise FigureValidationError('G3 duplicate utility cell')
    if set(m.dataset)!=set(DATASETS) or set(m.model)!=set(MODELS) or set(m.prompt_strategy)!=set(PROMPTS) or not m.groupby(keys).size().eq(2).all():
        raise FigureValidationError('G3 configuration universe incomplete')
    s=pd.read_csv(repository_path(root,STAT),dtype=str,keep_default_na=False)
    if len(s)!=90 or set(s.metric)!=set(STAT_METRICS) or s.duplicated(['dataset','model','prompt_strategy','metric']).any(): raise FigureValidationError('G3 statistical universe changed')
    if not s.condition_a.eq('RANDOM').all() or not s.condition_b.eq('FIRST').all() or not s.contrast.eq('FIRST_minus_RANDOM').all() or not s.paired_user_count.eq('200').all(): raise FigureValidationError('G3 contrast orientation/denominator changed')
    if not s.bootstrap_resamples.eq('10000').all() or not s.p_value.eq('').all() or not s.multiple_comparison_adjusted_p_value.eq('').all(): raise FigureValidationError('G3 inferential method changed')
    if not set(s.frozen_decision)==set(DECISION_MAP): raise FigureValidationError('G3 decision categories changed')
    low=s.ci_low.astype(float); high=s.ci_high.astype(float)
    derived=np.where(low>0,'CI_SUPPORTED_POSITIVE',np.where(high<0,'CI_SUPPORTED_NEGATIVE','NON_SIGNIFICANT_CI_INCLUDES_ZERO'))
    if not (s.frozen_decision.to_numpy()==derived).all(): raise FigureValidationError('G3 decisions differ from CIs')
    counts=s.frozen_decision.value_counts()
    if tuple(int(counts.get(k,0)) for k in DECISION_MAP)!=(85,1,4): raise FigureValidationError('G3 85/1/4 decision contract changed')
    expected_breakdowns=[
        ('prompt_strategy','BASE',(45,0,0)),
        ('prompt_strategy','LLMRANK',(40,1,4)),
        ('model','qwen2:7b-instruct',(30,0,0)),
        ('model','mistral:7b-instruct',(25,1,4)),
        ('model','gemma2:9b',(30,0,0)),
    ]
    for column, value, expected_counts in expected_breakdowns:
        subset=s.loc[s[column].eq(value)]
        observed=tuple(int(subset.frozen_decision.eq(decision).sum()) for decision in DECISION_MAP)
        if observed!=expected_counts:
            raise FigureValidationError(f'G3 frozen breakdown changed: {column}={value}')
    neg=s.loc[s.frozen_decision.eq('CI_SUPPORTED_NEGATIVE')]
    if len(neg)!=1 or neg.iloc[0][['dataset','model','prompt_strategy','metric']].tolist()!=['ml1m','mistral:7b-instruct','LLMRANK','hr_at_5']:
        raise FigureValidationError('G3 negative configuration identity changed')
    if not np.allclose(neg[['random_mean','first_mean','first_minus_random_delta','ci_low','ci_high']].astype(float).iloc[0], [.410,.250,-.160,-.250,-.075],atol=1e-12):
        raise FigureValidationError('G3 negative configuration values changed')
    syn=pd.read_csv(repository_path(root,SYN),dtype=str,keep_default_na=False)
    if len(syn)!=90 or not syn.comparison_id.isin(s.comparison_id).all(): raise FigureValidationError('G3 synthesis coverage changed')
    br=pd.read_csv(repository_path(root,BREAK),dtype=str,keep_default_na=False)
    overall=br.loc[br.scope.eq('overall')]
    if len(overall)!=1 or overall.iloc[0][['ci_supported_positive','ci_supported_negative','non_significant_ci_includes_zero']].tolist()!=['85','1','4']:
        raise FigureValidationError('G3 synthesis counts changed')
    return m,s


def utility_values(root:Path=ROOT)->pd.DataFrame:
    """Preserve each of 90 paired RANDOM/FIRST utility comparisons."""
    m,s=_g3(root); keys=['dataset','model','prompt_strategy','metric_id']
    a=m.loc[m.candidate_order_condition.eq('RANDOM'),keys+['value','planned_applicable_users','strict_valid_users','invalid_terminal_users']].rename(columns={'value':'random_value','strict_valid_users':'random_strict_valid','invalid_terminal_users':'random_invalid'})
    b=m.loc[m.candidate_order_condition.eq('FIRST'),keys+['value','strict_valid_users','invalid_terminal_users']].rename(columns={'value':'first_value','strict_valid_users':'first_strict_valid','invalid_terminal_users':'first_invalid'})
    out=a.merge(b,on=keys,validate='one_to_one')
    t=s[['dataset','model','prompt_strategy','metric','first_minus_random_delta']].rename(columns={'metric':'stat_metric'})
    mapping=dict(zip(METRICS,STAT_METRICS)); out['stat_metric']=out.metric_id.map(mapping)
    out=out.merge(t,on=['dataset','model','prompt_strategy','stat_metric'],validate='one_to_one')
    if not np.allclose(out.first_value.astype(float)-out.random_value.astype(float),out.first_minus_random_delta.astype(float),atol=1e-12): raise FigureValidationError('G3 paired utility delta mismatch')
    out['condition_contract']='SAME_CANDIDATE_SET_FIRST_TARGET_INFORMED_ORACLE'
    out['_d']=out.dataset.map({x:i for i,x in enumerate(DATASETS)}); out['_m']=out.model.map({x:i for i,x in enumerate(MODELS)}); out['_p']=out.prompt_strategy.map({x:i for i,x in enumerate(PROMPTS)}); out['_metric']=out.metric_id.map({x:i for i,x in enumerate(METRICS)})
    return out.sort_values(['_metric','_d','_m','_p']).drop(columns=['_metric','_d','_m','_p']).reset_index(drop=True)


def bootstrap_values(root:Path=ROOT)->pd.DataFrame:
    """Copy every frozen paired CI and classify it without new inference."""
    _,s=_g3(root)
    cols=['result_id','dataset','model','prompt_strategy','metric','condition_a','condition_b','paired_user_count','random_mean','first_mean','first_minus_random_delta','ci_low','ci_high','frozen_decision','bootstrap_resamples','confidence_interval','p_value_status']
    out=s[cols].copy().rename(columns={'metric':'metric_id','paired_user_count':'n_pairs','frozen_decision':'canonical_decision'})
    out['decision']=out.canonical_decision.map(DECISION_MAP)
    out['orientation']='FIRST_MINUS_RANDOM'
    out['_d']=out.dataset.map({x:i for i,x in enumerate(DATASETS)}); out['_m']=out.model.map({x:i for i,x in enumerate(MODELS)}); out['_p']=out.prompt_strategy.map({x:i for i,x in enumerate(PROMPTS)}); out['_metric']=out.metric_id.map({x:i for i,x in enumerate(STAT_METRICS)})
    return out.sort_values(['_metric','_d','_m','_p']).drop(columns=['_metric','_d','_m','_p']).reset_index(drop=True)


def r5_values(root:Path=ROOT)->pd.DataFrame:
    """Keep corrected R5B CandDif and both condition values as supporting evidence."""
    validate_sources(root)
    c=pd.read_csv(repository_path(root,R5_CAND),dtype=str,keep_default_na=False)
    m=pd.read_csv(repository_path(root,R5_METRIC),dtype=str,keep_default_na=False)
    if len(c)!=6 or len(m)!=12 or set(c.dataset)!={'Beauty','Sports','LastFM'} or set(c.model_name)!={'qwen2:7b-instruct','mistral:7b-instruct'}:
        raise FigureValidationError('R5B corrected configuration universe changed')
    if not c.n_users.eq('100').all() or not c.evidence_class.eq('PAPER_ALIGNED_ADAPTED_CORRECTED_RECOVERY').all(): raise FigureValidationError('R5B corrected evidence class changed')
    if not m.n_valid_outputs.eq('100').all() or not m.n_invalid_outputs.eq('0').all() or not m.n_backend_incomplete.eq('0').all(): raise FigureValidationError('R5B corrected accounting changed')
    rows=[]
    for item in c.to_dict('records'):
        subset=m.loc[m.dataset.eq(item['dataset']) & m.model_name.eq(item['model_name'])]
        if len(subset)!=2 or set(subset.experimental_condition)!={'CANDDIF_FIRST','CANDDIF_RANDOM'}: raise FigureValidationError('R5B condition pair missing')
        first=subset.loc[subset.experimental_condition.eq('CANDDIF_FIRST')].iloc[0]
        random=subset.loc[subset.experimental_condition.eq('CANDDIF_RANDOM')].iloc[0]
        for suffix,metric in [('HR','hr_at_5'),('NDCG','ndcg_at_5')]:
            row={'dataset':item['dataset'],'model_name':item['model_name'],'metric_id':metric,'canddif_value':item[f'CandDif_{suffix}'],'first_value':item[f'{suffix}_first'],'random_value':item[f'{suffix}_random'],'n_users':item['n_users'],'evidence_class':item['evidence_class'],'role':'SUPPORTING_R5B_CORRECTED_NO_G3_POOLING'}
            if not np.isclose(float(row['first_value']),float(first[metric])) or not np.isclose(float(row['random_value']),float(random[metric])): raise FigureValidationError('R5B corrected metric pair mismatch')
            rows.append(row)
    out=pd.DataFrame(rows)
    for metric,positive in [('hr_at_5',5),('ndcg_at_5',6)]:
        if int((out.loc[out.metric_id.eq(metric),'canddif_value'].astype(float)>0).sum())!=positive: raise FigureValidationError('R5B CandDif sign count changed')
    exception=out.loc[out.dataset.eq('Beauty') & out.model_name.eq('mistral:7b-instruct') & out.metric_id.eq('hr_at_5')]
    if len(exception)!=1 or not np.isclose(float(exception.canddif_value.iloc[0]),-.11441035117774434): raise FigureValidationError('R5B HR exception changed')
    return out.sort_values(['metric_id','dataset','model_name']).reset_index(drop=True)


def _style()->None: plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.hashsalt':'irp-final-figures-v1','pdf.compression':9})

def _labels()->list[str]:
    short={MODELS[0]:'Qwen2',MODELS[1]:'Mistral',MODELS[2]:'Gemma2'}
    return [f'{d.title()} | {short[m]} | {p}' for d in DATASETS for m in MODELS for p in PROMPTS]


def _utility_plot(v:pd.DataFrame)->plt.Figure:
    fig,axes=plt.subplots(1,3,figsize=(17,11),sharey=False)
    labels=_labels(); model_colors={MODELS[0]:'#0072B2',MODELS[1]:'#D55E00',MODELS[2]:'#009E73'}
    for ax,metric in zip(axes,METRICS,strict=True):
        rows=v.loc[v.metric_id.eq(metric)].copy(); rows['_d']=rows.dataset.map({x:i for i,x in enumerate(DATASETS)}); rows['_m']=rows.model.map({x:i for i,x in enumerate(MODELS)}); rows['_p']=rows.prompt_strategy.map({x:i for i,x in enumerate(PROMPTS)}); rows=rows.sort_values(['_d','_m','_p'])
        for i,row in enumerate(rows.itertuples(index=False)):
            color=model_colors[row.model]; ax.plot([float(row.random_value),float(row.first_value)],[i,i],color=color,linewidth=1.1,alpha=.85)
            ax.plot(float(row.random_value),i,marker='o',color=color,markersize=3.5); ax.plot(float(row.first_value),i,marker='s',color=color,markersize=3.5)
        ax.set_title(metric); ax.set_xlim(-.025,1.025); ax.set_ylim(29.6,-.6); ax.set_yticks(range(30)); ax.set_yticklabels(labels if ax is axes[0] else []); ax.tick_params(axis='y',labelsize=7,labelleft=(ax is axes[0])); ax.set_xlabel('Utility (RANDOM circle -> FIRST square)'); ax.grid(axis='x',alpha=.25)
        for y in [5.5,11.5,17.5,23.5]: ax.axhline(y,color='.7',linewidth=.6)
    fig.suptitle('RQ4 G3 paired utility under identical candidate membership',y=.98,fontsize=15)
    fig.text(.5,.018,'FIRST places the known relevant target at position 1: target-informed oracle stress condition, not a deployable ordering strategy.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.18,right=.985,bottom=.085,top=.92,wspace=.12)
    return fig


def _forest_plot(v:pd.DataFrame)->plt.Figure:
    fig=plt.figure(figsize=(17,13)); axes=[fig.add_subplot(2,3,i+1) for i in range(3)]; bar=fig.add_subplot(2,1,2)
    colors={'POSITIVE':'#0072B2','NEGATIVE':'#D55E00','INCONCLUSIVE':'#777777'}; labels=_labels()
    for ax,metric in zip(axes,STAT_METRICS,strict=True):
        rows=v.loc[v.metric_id.eq(metric)].copy(); rows['_d']=rows.dataset.map({x:i for i,x in enumerate(DATASETS)}); rows['_m']=rows.model.map({x:i for i,x in enumerate(MODELS)}); rows['_p']=rows.prompt_variant.map({x:i for i,x in enumerate(PROMPTS)}) if 'prompt_variant' in rows else rows.prompt_strategy.map({x:i for i,x in enumerate(PROMPTS)}); rows=rows.sort_values(['_d','_m','_p'])
        for i,row in enumerate(rows.itertuples(index=False)):
            color=colors[row.decision]; lo,delta,hi=map(float,[row.ci_low,row.first_minus_random_delta,row.ci_high]); ax.plot([lo,hi],[i,i],color=color,lw=1); ax.plot(delta,i,'o',color=color,markersize=3)
        counts=rows.decision.value_counts(); ax.axvline(0,color='black',lw=.8); ax.set_title(f"{metric.upper()} (+{counts.get('POSITIVE',0)} / -{counts.get('NEGATIVE',0)} / ?{counts.get('INCONCLUSIVE',0)})"); ax.set_ylim(29.6,-.6); ax.set_yticks(range(30),labels if ax is axes[0] else []); ax.tick_params(axis='y',labelsize=6); ax.set_xlabel('FIRST - RANDOM, paired 95% CI'); ax.grid(axis='x',alpha=.2)
        for y in [5.5,11.5,17.5,23.5]: ax.axhline(y,color='.7',lw=.6)
    counts=v.decision.value_counts(); left=0
    for key in ['POSITIVE','NEGATIVE','INCONCLUSIVE']:
        n=int(counts.get(key,0)); bar.barh(0,n,left=left,color=colors[key],label=f'{key}: {n}',height=.45); left+=n
    bar.set_xlim(0,90); bar.set_ylim(-.8,.8); bar.set_yticks([]); bar.set_xlabel('Configuration x metric comparisons (n=90)'); bar.legend(loc='upper center',bbox_to_anchor=(.5,-.28),ncol=3,frameon=False)
    fig.suptitle('RQ4 G3 paired bootstrap: configuration-level FIRST - RANDOM effects',y=.992,fontsize=15)
    fig.text(.5,.01,'Blue=positive; orange=negative; gray=inconclusive. FIRST is a target-informed oracle stress condition, not a deployable policy.',ha='center',fontsize=8)
    axes[0].set_position([.17,.41,.24,.49]); axes[1].set_position([.45,.41,.24,.49]); axes[2].set_position([.73,.41,.24,.49]); bar.set_position([.17,.18,.80,.09])
    return fig


def _r5_plot(v:pd.DataFrame)->plt.Figure:
    fig,axes=plt.subplots(1,2,figsize=(10,5.5)); datasets=['Beauty','Sports','LastFM']; models=['qwen2:7b-instruct','mistral:7b-instruct']; allvals=v.canddif_value.astype(float); limit=max(abs(allvals.min()),abs(allvals.max())); norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit)
    for ax,metric,title in zip(axes,['hr_at_5','ndcg_at_5'],['CandDif HR@5','CandDif NDCG@5'],strict=True):
        subset=v.loc[v.metric_id.eq(metric)]; matrix=np.array([[float(subset.loc[subset.dataset.eq(d)&subset.model_name.eq(m),'canddif_value'].iloc[0]) for m in models] for d in datasets]); image=ax.imshow(matrix,cmap='coolwarm',norm=norm,aspect='auto')
        for i in range(3):
            for j in range(2): ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',color='black',fontsize=11)
        ax.set_title(title); ax.set_xticks([0,1],['Qwen2','Mistral']); ax.set_yticks(range(3),datasets); ax.set_xlabel('Model'); ax.set_ylabel('Dataset'); ax.set_xticks(np.arange(-.5,2,1),minor=True); ax.set_yticks(np.arange(-.5,3,1),minor=True); ax.grid(which='minor',color='white',linewidth=1); ax.tick_params(which='minor',bottom=False,left=False)
    cb=fig.colorbar(image,ax=axes,fraction=.04,pad=.03); cb.set_label('Frozen CandDif value')
    fig.suptitle('RQ4 corrected R5B targeted recovery (supporting evidence)',y=.98,fontsize=14)
    fig.text(.5,.035,'Targeted paper-aligned adapted recovery; separate from systematic G3. Negative Beauty/Mistral HR is retained.',ha='center',fontsize=8)
    fig.subplots_adjust(left=.10,right=.85,bottom=.19,top=.82,wspace=.35)
    return fig


def _save(fig:plt.Figure,target:Path,figure_id:str)->dict[str,str]:
    out={}
    for ext in ['png','pdf','svg']:
        rel=f'{OUT}/figures/{figure_id}.{ext}'; p=repository_path(target,rel); p.parent.mkdir(parents=True,exist_ok=True)
        meta={'Software':'IRP deterministic final-figure pipeline'} if ext=='png' else {'Creator':'IRP deterministic final-figure pipeline','Producer':'Matplotlib','CreationDate':FIXED_TIME,'ModDate':FIXED_TIME} if ext=='pdf' else {'Creator':'IRP deterministic final-figure pipeline','Date':'2026-09-21'}
        fig.savefig(p,dpi=170,metadata=meta,facecolor='white')
        if ext=='svg': p.write_text('\n'.join(line.rstrip() for line in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
        out[ext]=rel; out[f'{ext}_sha256']=sha256_file(p)
    plt.close(fig); return out


def _values(figure_id:str,root:Path)->pd.DataFrame:
    return utility_values(root) if figure_id==FIG1 else bootstrap_values(root) if figure_id==FIG2 else r5_values(root)


def generate_all(root:Path=ROOT,*,artifact_root:Path|None=None,created_from_commit:str|None=None)->list[dict]:
    """Write only the three authorized RQ4 figure chains."""
    root=root.resolve(); target=(artifact_root or root).resolve(); validate_sources(root); _style(); commit=created_from_commit or git_head(root)
    if len(commit)!=40: raise FigureValidationError('full commit SHA required')
    out=[]
    for figure_id in IDS:
        values=_values(figure_id,root); rel=f'{OUT}/values/{figure_id}_values.csv'; p=repository_path(target,rel); write_dataframe(p,values)
        pd.testing.assert_frame_equal(pd.read_csv(p,dtype=str,keep_default_na=False).astype(str),values.astype(str),check_dtype=False,check_exact=True)
        images=_save(_utility_plot(values) if figure_id==FIG1 else _forest_plot(values) if figure_id==FIG2 else _r5_plot(values),target,figure_id)
        sources=G3_SOURCES if figure_id!=FIG3 else R5_SOURCES
        supporting=figure_id==FIG3
        manifest={'schema_version':'1.0','figure_id':figure_id,'rq':'RQ4','scientific_purpose':'Targeted corrected R5B CandDif recovery, supporting only' if supporting else 'Systematic G3 paired candidate-order utility' if figure_id==FIG1 else 'Systematic G3 paired bootstrap FIRST-minus-RANDOM effects',
         'canonical_source_artifacts':sources,'canonical_source_sha256':{x:HASHES[x] for x in sources},'figure_values_csv':rel,'figure_values_sha256':sha256_file(p),'generator':GENERATOR,'generator_sha256':sha256_file(repository_path(root,GENERATOR)),
         'output_png':images['png'],'output_png_sha256':images['png_sha256'],'output_pdf':images['pdf'],'output_pdf_sha256':images['pdf_sha256'],'output_svg':images['svg'],'output_svg_sha256':images['svg_sha256'],
         'scientific_interpretation':'Corrected targeted recovery is supporting evidence and is not pooled with systematic G3.' if supporting else 'Initial candidate ordering affects utility under paired identical-membership conditions; FIRST-minus-RANDOM effects are predominantly positive with exceptions.',
         'limitations':['R5B targeted and G3 systematic results are never pooled.','Paper-aligned adapted recovery, not paper-exact reproduction.'] if supporting else ['FIRST explicitly places the known relevant target at candidate position 1: target-informed oracle stress condition, not a deployable ordering strategy.','Paired comparisons keep candidate membership fixed and change only initial order.','No p-values, multiplicity-adjusted p-values, or synthetic significance scores.'],
         'created_from_commit':commit,'evidence_role':'SUPPORTING_R5B_TARGETED' if supporting else 'PRIMARY_G3_SYSTEMATIC','display_contract':{'no_r5b_g3_pooling':True,'first_target_informed_oracle':not supporting,'no_deployability_claim':True,'no_statistical_recalculation':True},'row_count':len(values)}
        if figure_id==FIG2: manifest['decision_counts']={k:int(values.decision.eq(k).sum()) for k in ['POSITIVE','NEGATIVE','INCONCLUSIVE']}
        mp=repository_path(target,f'{MOUT}/{figure_id}_manifest.json'); write_json(mp,manifest); validate_figure_manifest(root,mp,artifact_root=target); out.append(manifest)
    return out


def validate_all(root:Path=ROOT,*,artifact_root:Path|None=None)->list[dict]:
    """Verify source, values, output hashes, and R5B/G3 evidence separation."""
    root=root.resolve(); target=(artifact_root or root).resolve(); validate_sources(root); out=[]
    for figure_id in IDS:
        mp=repository_path(target,f'{MOUT}/{figure_id}_manifest.json'); manifest=validate_figure_manifest(root,mp,artifact_root=target)
        expected=_values(figure_id,root).astype(str).reset_index(drop=True)
        actual=pd.read_csv(repository_path(target,manifest['figure_values_csv']),dtype=str,keep_default_na=False).astype(str).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=True)
        if manifest['evidence_role']!=('SUPPORTING_R5B_TARGETED' if figure_id==FIG3 else 'PRIMARY_G3_SYSTEMATIC'): raise FigureValidationError('RQ4 evidence role changed')
        out.append(manifest)
    return out


def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--mode',choices=['generate','validate'],required=True); p.add_argument('--artifact-root',type=Path); p.add_argument('--created-from-commit'); a=p.parse_args(argv)
    result=generate_all(ROOT,artifact_root=a.artifact_root,created_from_commit=a.created_from_commit) if a.mode=='generate' else validate_all(ROOT,artifact_root=a.artifact_root)
    print(json.dumps({'status':'PASS','figures':[m['figure_id'] for m in result]})); return 0
if __name__=='__main__': raise SystemExit(main())
