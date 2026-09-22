"""Reproduce the final RQ5 package offline from bundled immutable evidence.

No network clients or model execution functions are imported. Archival producers
are documentary provenance, not the operational reproduction entry points.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import logging
import platform
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.evaluation.rq5_freeze_metrics import parse_ranking, mostpop, beyond_utility
NS = "results/rq5_final_scientific_freeze"
POLICY = "FAIL_STRICT_ALL_MANIFEST_REQUESTS_INVALID_ZERO"
Q = "qwen2:7b-instruct"
METRICS = ["hr_at_5", "ndcg_at_5"]
COMPARISONS = [("profile_only_vs_history_only", "profile_only"), ("profile_plus_history_vs_history_only", "profile_plus_history")]

def stable(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

def digest(data):
    return hashlib.sha256(data).hexdigest()

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True)+"\n", encoding="utf-8", newline="\n")

def write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")

class Evidence:
    """Resolve only inventory-declared ZIP members; no fallback to other worktrees."""
    def __init__(self, root=ROOT):
        self.root = root
        self.inventory = pd.read_csv(root/NS/"package_inventory.csv", dtype=str, keep_default_na=False)
        self.members = {}
        self.archives = {}
        for row in self.inventory.to_dict("records"):
            archive, member = row["path"].split("!", 1)
            require(not Path(archive).is_absolute() and ".." not in Path(archive).parts, "unsafe archive path")
            self.members[member] = (archive, row)
            if archive not in self.archives:
                self.archives[archive] = zipfile.ZipFile(root/archive)
    def read(self, member):
        archive, row = self.members[member]
        return self.archives[archive].read(member)
    def obj(self, member):
        return json.loads(self.read(member))
    def table(self, member, strings=False):
        return pd.read_csv(io.BytesIO(self.read(member)), dtype=str if strings else None, keep_default_na=False, float_precision="round_trip")
    def resolve(self, path):
        path = str(path).replace("\\", "/")
        if path in self.members: return path
        for alias in ["submission", "research"]:
            candidate = alias+"/"+path
            if candidate in self.members: return candidate
        raise FileNotFoundError(path)
    def verify(self):
        for member, (_, row) in self.members.items():
            data = self.read(member)
            require(len(data) == int(row["size"]) and digest(data) == row["sha256"], "evidence hash: "+member)
        return len(self.members)
    def close(self):
        for archive in self.archives.values(): archive.close()

def source_table(evidence, prefix, suffix, strings=False):
    return evidence.table(f"submission/results/tables/{prefix}_{suffix}.csv", strings)

def profile_text(payload):
    for key in ["profile_text", "raw_response", "response_text", "user_profile"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip(): return value.strip()
    for key in ["ollama_response", "ollama_raw_response"]:
        value = payload.get(key, {}).get("response", "")
        if isinstance(value, str) and value.strip(): return value.strip()
    return ""

def profiles(evidence, family, spec):
    suffix = "profile_reuse_audit" if family == "corrected" else "profile_manifest"
    frame = source_table(evidence, spec["prefix"], suffix, True)
    records, audit = {}, []
    for row in frame.to_dict("records"):
        path = row["raw_source_file"] if family == "corrected" else row["output_destination"]
        member = evidence.resolve(path)
        payload = evidence.obj(member)
        text = profile_text(payload)
        require(bool(text), "empty profile")
        require(digest(text.encode()).lower() == row["profile_hash"].lower(), "profile text hash")
        require(str(payload.get("user_id", payload.get("manifest_row", {}).get("user_id", row["user_id"]))) == row["user_id"], "profile user")
        records[row["profile_request_id"]] = (row, text)
        source = row.get("source_profile_file", row.get("raw_source_file", ""))
        source_member = evidence.resolve(source) if source else ""
        if source_member:
            require(profile_text(evidence.obj(source_member)) == text, "profile reuse text differs")
        audit.append({"profile_request_id": row["profile_request_id"], "dataset": row["dataset"], "user_id": row["user_id"], "profile_generator": row["profile_generator"], "evidence_member": member, "source_member": source_member, "profile_sha256": digest(text.encode()), "source_sha256": digest(evidence.read(source_member)) if source_member else "", "reused": bool(source_member), "status": "PASS"})
    require(len(records) == spec["profiles"], "profile count")
    return records, pd.DataFrame(audit)

def popularity(evidence, config):
    cfg = config["popularity"]
    data = evidence.table(cfg["interactions"], True)
    train = data.loc[data["split"].eq(cfg["split"])]
    counts = {str(k): int(v) for k,v in train[cfg["item_column"]].value_counts(sort=False).items()}
    threshold = float(np.quantile(list(counts.values()), cfg["quantile"], method="linear"))
    catalog = evidence.table(cfg["catalog"], True)
    return counts, {k for k,v in counts.items() if v < threshold}, set(catalog["internal_item_id"]), {"source": cfg["interactions"], "source_sha256": digest(evidence.read(cfg["interactions"])), "catalog_sha256": digest(evidence.read(cfg["catalog"])), "frequency_sha256": digest(stable(counts).encode()), "train_rows": len(train), "positive_count_items": len(counts), "long_tail_threshold": threshold, "tie_break": cfg["tie_break"], "classification": cfg["classification"]}

def validate_request_pair(base, treatment):
    for key in ["dataset", "user_id", "model_name", "candidate_item_ids_json", "candidate_item_titles_json", "candidate_list_hash", "positive_item_id", "positive_candidate_position", "history_context_hash", "parser_contract", "prompt_template_id", "temperature", "seed", "max_new_tokens", "expected_output_count"]:
        require(base[key] == treatment[key], "matched request differs: "+key)
    marker = "Use only the train-history, generated profile when provided, and candidate context below."
    require(marker in base["prompt_render"] and marker in treatment["prompt_render"], "ranking instruction missing")
    require(base["prompt_render"].split(marker,1)[1] == treatment["prompt_render"].split(marker,1)[1], "instruction/candidates differ")
    if treatment["condition"] == "profile_plus_history":
        require(base["prompt_render"].split(marker,1)[0].strip() == treatment["prompt_render"].split("Generated user profile:\n",1)[0].strip(), "visible history differs")
    else:
        require(treatment["prompt_render"].startswith("Generated user profile:\n"), "profile-only visibility")

def rankings(evidence, family, spec, profile_records, pop=None):
    universe = source_table(evidence, spec["prefix"], spec["request_suffix"], True)
    require(len(universe) == spec["rankings"] and universe.request_id.is_unique, "request accounting")
    require(set(universe.dataset) == set(spec["datasets"]), "dataset scope")
    require(set(universe.model_name) == set(spec["recommenders"]), "model scope")
    require(set(universe.profile_generator_name)-{"none"} == set(spec["generators"]), "generator scope")
    bases = {}
    for row in universe.loc[universe.condition.eq("history_only")].to_dict("records"):
        bases[(row["dataset"],row["model_name"],row["user_id"],row.get("history_length",""))] = row
    rows, pair_checks, reuse = [], [], []
    for row in universe.to_dict("records"):
        member = evidence.resolve(row["output_destination"])
        raw = evidence.obj(member)
        for key in ["request_id", "dataset", "user_id", "condition", "model_name", "prompt_hash", "candidate_list_hash", "positive_item_id"]:
            require(str(raw[key]) == row[key], "raw identity: "+key)
        require(raw["execution_status"] == "SUCCEEDED", "nonterminal/backend failed raw")
        require(raw["runtime_model_id"] == row["model_name"], "runtime model mismatch")
        require(raw["prompt_render"] == row["prompt_render"] and digest(row["prompt_render"].encode()).upper() == row["prompt_hash"].upper(), "prompt hash")
        candidates = list(map(str,json.loads(row["candidate_item_ids_json"])))
        require(len(candidates)==20 and len(set(candidates))==20, "candidate membership")
        require(candidates[int(row["positive_candidate_position"])-1] == row["positive_item_id"], "target position")
        parsed = parse_ranking(raw["response_text"], candidates, row["positive_item_id"])
        require(bool(parsed["strict_valid"]) == bool(raw["parse"]["parse_valid"]), "parse validity changed")
        for key in METRICS:
            require(parsed[key] == float(raw["parse"][key]), "raw utility mismatch: "+key)
        if row["condition"] != "history_only":
            pr, text = profile_records[row["profile_request_id"]]
            require(pr["dataset"] == row["dataset"] and pr["user_id"] == row["user_id"], "profile dependency user")
            require(pr["profile_generator"] == row["profile_generator_name"], "profile generator mismatch")
            require(digest(text.encode()).upper()==row["profile_hash"].upper(), "ranking profile hash")
            require("Generated user profile:\n"+text+"\n\n" in row["prompt_render"], "profile treatment differs from evidence")
            base=bases[(row["dataset"],row["model_name"],row["user_id"],row.get("history_length",""))]
            validate_request_pair(base,row)
            pair_checks.append({"baseline_request_id":base["request_id"],"treatment_request_id":row["request_id"],"status":"PASS"})
        source = raw.get("reuse_source_file", "")
        if source:
            src = evidence.resolve(source); old=evidence.obj(src)
            require(raw["response_text"]==old.get("response_text",old.get("raw_response_text")), "reused response differs")
            require(raw["prompt_hash"]==old["prompt_hash"], "reused prompt differs")
            reuse.append({"request_id": row["request_id"], "source_member":src, "source_sha256":digest(evidence.read(src)), "status":"PASS"})
        rec = {"request_id":row["request_id"],"dataset":row["dataset"],"user_id":row["user_id"],"recommender":row["model_name"],"profile_generator":row["profile_generator_name"],"condition":row["condition"],"strict_valid":parsed["strict_valid"],"hr_at_5":parsed["hr_at_5"],"ndcg_at_5":parsed["ndcg_at_5"],"raw_member":member,"raw_sha256":digest(evidence.read(member)),"denominator_policy":POLICY}
        if family=="history_v2":
            counts,tail,catalog,_ = pop
            require(set(candidates)<=catalog, "candidate canonical ID absent from catalog")
            rec["history_length"]=int(row["history_length"])
            rec.update(beyond_utility(parsed,candidates,row["positive_item_id"],counts,tail))
        rows.append(rec)
    result=pd.DataFrame(rows)
    require(int(result.strict_valid.sum()) == spec["valid"], "valid count")
    require(len(reuse)=={"corrected":200,"systematic":2400,"history_v2":600}[family], "exact ranking reuse count")
    return result,pd.DataFrame(pair_checks),pd.DataFrame(reuse),universe

def aggregate(per, family):
    metrics=METRICS+(["aplt_at_5","serendipity_at_5"] if family=="history_v2" else [])
    keys=["dataset","recommender","profile_generator","condition"]
    if family=="history_v2": keys=["history_length","condition"]
    rows=[]
    for key,g in per.groupby(keys,sort=True):
        rec=dict(zip(keys,key));rec.update({"requests":len(g),"strict_valid":int(g.strict_valid.sum()),"denominator_policy":POLICY})
        require(len(g)==200,"cell denominator")
        for metric in metrics:rec[metric]=float(g[metric].sum()/len(g))
        rows.append(rec)
    return pd.DataFrame(rows)

def contributions(evidence,per,family,spec):
    index=per.set_index("request_id")
    if family!="history_v2":
        old=source_table(evidence,spec["prefix"],"paired_contributions",True)
        rows=[]
        for row in old.to_dict("records"):
            h=index.loc[row["history_request_id"]];t=index.loc[row["treatment_request_id"]];metric=row["metric"]
            require(h.user_id==t.user_id==row["user_id"],"paired user mismatch")
            a,b=float(h[metric]),float(t[metric]); delta=b-a
            require(a==float(row["history_value"]) and b==float(row["treatment_value"]) and delta==float(row["delta"]),"frozen paired contributions differ")
            row.update(history_value=a,treatment_value=b,delta=delta)
            rows.append(row)
        return pd.DataFrame(rows)
    lookup=per.set_index(["history_length","condition","user_id"]);rows=[]
    for base in per.loc[per.condition.eq("history_only")].to_dict("records"):
        for comparison,cond in COMPARISONS:
            t=lookup.loc[(base["history_length"],cond,base["user_id"])]
            for metric in METRICS+["aplt_at_5","serendipity_at_5"]:
                a,b=float(base[metric]),float(t[metric])
                rows.append({"dataset":"beauty","recommender":Q,"profile_generator":Q,"history_length":base["history_length"],"user_id":base["user_id"],"comparison":comparison,"metric":metric,"history_value":a,"treatment_value":b,"delta":b-a,"denominator_policy":POLICY})
    return pd.DataFrame(rows)

def bootstrap(contrib,family,config):
    stats=config["statistics"];rng=np.random.default_rng(stats["seed"]);rows=[]
    keys=["dataset","recommender","profile_generator","comparison","metric"]
    if family=="systematic":keys=["dataset","recommender","comparison","metric"]
    if family=="history_v2":keys=["history_length","comparison","metric"]
    for key,g in contrib.groupby(keys,sort=True):
        delta=g.delta.astype(float).to_numpy();require(len(delta)==200,"bootstrap pair count")
        idx=rng.integers(0,len(delta),size=(stats["resamples"],len(delta)))
        lo,hi=np.quantile(delta[idx].mean(axis=1),stats["quantiles"],method=stats["quantile_method"])
        rec=dict(zip(keys,key))
        if family in {"systematic","history_v2"}:rec["profile_generator"]=Q
        if family=="history_v2":rec.update(dataset="beauty",recommender=Q)
        rec.update(paired_n=len(delta),mean_delta=float(delta.mean()),ci_lower=float(lo),ci_upper=float(hi),bootstrap_resamples=stats["resamples"],bootstrap_seed=stats["seed"],decision="CI_SUPPORTED_POSITIVE" if lo>0 else "CI_SUPPORTED_NEGATIVE" if hi<0 else "NON_SIGNIFICANT_CI_INCLUDES_ZERO",p_value="",multiplicity_correction="not_computed",inference_scope="local_exploratory_paired_user_percentile_95")
        rows.append(rec)
    return pd.DataFrame(rows)

def compare_numeric(new,old,keys,cols,tolerance=0.0):
    a=new.set_index(keys).sort_index();b=old.set_index(keys).sort_index()
    require(a.index.equals(b.index),"different row identities")
    for col in cols:
        diff=float(np.max(np.abs(a[col].astype(float).to_numpy()-b[col].astype(float).to_numpy())))
        require(diff<=tolerance,f"{col} discrepancy: {diff}")


def r5d(evidence,out):
    manifest_member="research/results/manifests/reproduction_r5d_generation_involved_v2_freeze.json"
    manifest=evidence.obj(manifest_member)
    for section in ["raw_file_hashes","table_hashes","source_hashes","report_hashes","diagnostic_hashes"]:
        for path,expected in manifest[section].items():
            require(digest(evidence.read("research/"+path))==expected.lower(),"R5D frozen hash "+path)
    records=[];profiles_by_id={};all_raw=[]
    for path in manifest["raw_file_hashes"]:
        raw=evidence.obj("research/"+path);all_raw.append(raw)
        if raw["task"]=="profile_generation":
            obj=json.loads(raw["raw_response_text"])
            require(set(obj)=={"user_profile"} and isinstance(obj["user_profile"],str) and bool(obj["user_profile"].strip()),"R5D structural profile")
            require(digest(obj["user_profile"].strip().encode())==raw["profile_hash"].lower(),"R5D profile hash")
            profiles_by_id[raw["request_id"]]=raw
    for raw in all_raw:
        if raw["task"]!="ranking":continue
        valid=bool(raw["parse_valid"])
        rank=int(raw["positive_rank"]) if valid else 0
        hr=float(0<rank<=5);nd=1/np.log2(rank+1) if hr else 0.
        require(hr==raw["hr_at_5"] and abs(nd-raw["ndcg_at_5"])<1e-15,"R5D utility")
        if raw["r5d_profile_dependency_request_id"]:
            dep=profiles_by_id[raw["r5d_profile_dependency_request_id"]]
            require(dep["profile_hash"]==raw["profile_hash"] and all(dep[k]==raw[k] for k in ["dataset","user_id","model_name"]),"R5D dependency")
        records.append({"dataset":raw["dataset"],"recommender":raw["model_name"],"user_id":raw["user_id"],"request_id":raw["request_id"],"strict_valid":int(valid),"hr_at_5":hr,"ndcg_at_5":float(raw["ndcg_at_5"]),"denominator_policy":POLICY})
    per=pd.DataFrame(records);require(len(profiles_by_id)==400 and len(per)==800 and per.strict_valid.sum()==795,"R5D accounting")
    rows=[]
    for (dataset,model),g in per.groupby(["dataset","recommender"],sort=True):
        require(len(g)==100,"R5D cell denominator")
        rows.append({"dataset":dataset,"recommender":model,"requests":100,"strict_valid":int(g.strict_valid.sum()),"hr_at_5":float(g.hr_at_5.mean()),"ndcg_at_5":float(g.ndcg_at_5.mean()),"denominator_policy":POLICY})
    write_csv(out/"r5d/per_user.csv",per);write_csv(out/"r5d/metrics.csv",pd.DataFrame(rows))
    return {"status":"PASS","requests":1200,"profiles":400,"rankings":800,"valid":1195,"invalid":5,"original_freeze_commit":"ba1dbe8367216f7ea1044b2ff8db69318f8678ee","manifest_member":manifest_member,"original_manifest_sha256":digest(evidence.read(manifest_member))}

def recompute(root,config,out):
    """Derive all authoritative analytics; never mutate any archived input."""
    evidence=Evidence(root)
    try:
        verified=evidence.verify();logging.info("Verified %d archived evidence members",verified)
        summary={"r5d":r5d(evidence,out),"raw_inference_unchanged":True,"evidence_members_verified":verified}
        pop=popularity(evidence,config)
        for family,spec in config["families"].items():
            profile_records,profile_audit=profiles(evidence,family,spec)
            per,matching,reuse,universe=rankings(evidence,family,spec,profile_records,pop if family=="history_v2" else None)
            means=aggregate(per,family);con=contributions(evidence,per,family,spec);bs=bootstrap(con,family,config)
            require(len(bs)==spec["bootstrap_rows"],"bootstrap row accounting")
            old_means=source_table(evidence,spec["prefix"],"metrics")
            keys=["dataset","recommender","condition"]+(["profile_generator"] if family=="corrected" else [])
            if family=="history_v2": keys=["history_length","condition"]
            compare_numeric(means,old_means,keys,METRICS+["strict_valid"])
            if family!="history_v2":
                old_bs=source_table(evidence,spec["prefix"],"bootstrap_results")
                bkeys=["dataset","recommender","comparison","metric"]+(["profile_generator"] if family=="corrected" else [])
                compare_numeric(bs,old_bs,bkeys,["mean_delta","ci_lower","ci_upper"])
                require(bs.set_index(bkeys).decision.sort_index().equals(old_bs.set_index(bkeys).decision.sort_index()),"bootstrap decisions")
            # Operational tables never depend on source-worktree absolute paths.
            for col in con.columns:
                if con[col].dtype==object:
                    require(not con[col].astype(str).str.contains(r"[A-Za-z]:[\\/]",regex=True).any(),"absolute contribution path")
            directory=out/family
            for name,frame in [("per_user",per),("metrics",means),("paired_contributions",con),("bootstrap_results",bs),("matching_validation",matching),("ranking_reuse_audit",reuse),("profile_reuse_audit",profile_audit)]:
                write_csv(directory/(name+".csv"),frame)
            # Normalize only path metadata in a new request manifest; original bytes stay archived.
            normalized=universe.copy()
            for col in normalized.columns:
                if col.endswith(("_file","_path","_root")) or col in {"source_file","source_scope"}:
                    normalized[col]=normalized[col].map(lambda value: value.replace("\\","/").split("IRP_BeyondUtility_Project/",1)[-1].split("IRP_BeyondUtility_Submission_Worktree/",1)[-1] if isinstance(value,str) else value)
            write_csv(directory/"request_universe.csv",normalized)
            summary[family]={"status":"PASS","profiles":len(profile_audit),"rankings":len(per),"strict_valid":int(per.strict_valid.sum()),"matched_pairs":len(matching),"ranking_reused":len(reuse),"bootstrap_contrasts":len(bs),"decisions":bs.decision.value_counts().to_dict(),"raw_inference_unchanged":True,"authority":spec["authority"]}
            if family=="systematic":
                require(int(profile_audit.reused.sum())==400,"systematic profile reuse count")
                write_csv(out/"rq5_primary_systematic_profile_reuse_audit.csv",profile_audit)
            if family=="history_v2":
                old=source_table(evidence,spec["prefix"],"per_user_metrics")
                old=old.set_index("request_id");new=per.set_index("request_id")
                invalid_changes=[]
                for ident,row in new.loc[new.strict_valid.eq(0)].iterrows():
                    require(all(row[m]==0.0 for m in METRICS+["aplt_at_5","serendipity_at_5"]),"terminal-invalid contribution not zero")
                    if float(old.loc[ident,"aplt_at_5"])!=row.aplt_at_5:
                        invalid_changes.append({"request_id":ident,"history_length":int(row.history_length),"condition":row.condition,"old_aplt":float(old.loc[ident,"aplt_at_5"]),"new_aplt":float(row.aplt_at_5)})
                require(len(invalid_changes)==3,"unexpected APLT affected request count")
                base=per.loc[per.history_length.eq(10)&per.condition.eq("history_only")].copy()
                base=base.sort_values("user_id",key=lambda s:s.astype(int));oldbase=old.loc[base.request_id]
                vector=[int(v) for v in base.mostpop_hit_at_5]
                oldvector=[int(v) for v in oldbase.mostpop_hit_at_5]
                reference=[{"user_id":row.user_id,"top5":json.loads(row.mostpop_top5_item_ids_json)} for row in base.itertuples()]
                oldreference=[{"user_id":str(row.user_id),"top5":json.loads(row.mostpop_top5_item_ids_json)} for row in oldbase.itertuples()]
                migration={**pop[3],"version":"V2","raw_inference_unchanged":True,"reason":["strict-invalid APLT must be zero","replace NumPy-dependent tie ordering with canonical item ID ordering"],"aplt_affected_requests":invalid_changes,"aplt_affected_cells":[{"history_length":x["history_length"],"condition":x["condition"]} for x in invalid_changes],"old_mostpop_sha256":digest(stable(oldreference).encode()),"new_mostpop_sha256":digest(stable(reference).encode()),"old_hit_vector_sha256":digest(stable(oldvector).encode()),"new_hit_vector_sha256":digest(stable(vector).encode()),"old_hit_count":sum(oldvector),"audit_observed_historical_hit_count":56,"new_hit_count":sum(vector),"audit_observed_is_not_target":True,"old_per_user_member":f"submission/results/tables/{spec['prefix']}_per_user_metrics.csv","old_per_user_sha256":digest(evidence.read(f"submission/results/tables/{spec['prefix']}_per_user_metrics.csv")),"new_per_user_sha256":digest((directory/"per_user.csv").read_bytes()),"affected_artifacts":["per_user.csv","metrics.csv","paired_contributions.csv","bootstrap_results.csv"],"old_artifacts_preserved_in":"evidence_history_v1.zip"}
                for metric_file in ["metrics","bootstrap_results"]:
                    migration["old_"+metric_file+"_sha256"]=digest(evidence.read(f"submission/results/tables/{spec['prefix']}_{metric_file}.csv"))
                    migration["new_"+metric_file+"_sha256"]=digest((directory/(metric_file+".csv")).read_bytes())
                write_json(directory/"migration_manifest.json",migration)
                write_csv(directory/"mostpop_reference.csv",base[["user_id","mostpop_top5_item_ids_json","mostpop_hit_at_5"]])
                summary[family].update(utility="PASS",aplt="PASS",serendipity="PASS",mostpop_deterministic="PASS",mostpop_hit_count=sum(vector))
            write_json(directory/"validation.json",summary[family])
            logging.info("%s: %s",family,summary[family])
        write_json(out/"validation.json",summary)
        synthesis={"PROFILE_GENERATION_FEASIBILITY":"SUPPORTED_UNDER_VERIFIED_STRUCTURAL_CONTRACT","DOWNSTREAM_PROFILE_UTILITY":"MEASURED","PROFILE_INFORMATION_PRESERVATION":"OPERATIONAL_ONLY","CORRECTED_PROFILE_ONLY_EFFECT":"LOCAL_MATCHED_EFFECT","CORRECTED_PROFILE_PLUS_HISTORY_EFFECT":"NO_GENERAL_POSITIVE_EFFECT_SUPPORTED","SYSTEMATIC_PROFILE_PLUS_HISTORY_SIGNAL":"LOCAL_EXPLORATORY_SIGNAL","CAUSAL_PROFILE_BENEFIT":"PARTIAL_LOCAL_TREATMENT_EFFECT_ONLY","semantic_preference_preservation_validated":False,"multiplicity_correction":False,"independent_replications":False,"systematic_reused_corrected_rankings":2400,"authority":{"corrected":"FOREGROUND","r5d":"SUPPORTING","systematic":"SUPPORTING","history_v2":"EXPERIMENTALLY_DISTINCT","f6l_statistics":"FORENSIC","pilot_v1":"SUPERSEDED","history_v1":"FORENSIC"},"signals":[],"limitations":["Local models, cohorts and delivered profile interventions only","No semantic preference-preservation validation","No p-values or multiplicity correction; exploratory confidence intervals","Reused observations are not independent replications","History-slice popularity is local adapted, not paper-exact","Lexicographic canonical item ID tie-break is a new V2 analysis contract"]}
        for family in ["corrected","systematic"]:
            bs=pd.read_csv(out/family/"bootstrap_results.csv",keep_default_na=False,float_precision="round_trip")
            for row in bs.loc[~bs.decision.eq("NON_SIGNIFICANT_CI_INCLUDES_ZERO")].to_dict("records"):
                row["family"]=family;synthesis["signals"].append(row)
        write_json(out/"scientific_synthesis.json",synthesis)
        return summary
    finally:evidence.close()


def figures(root,config,out,plot_dir):
    """Render only authoritative CSVs, one axes per figure, default matplotlib colors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import textwrap
    from datetime import datetime,timezone
    plt.rcParams["svg.hashsalt"]="rq5-final-scientific-freeze-v2"
    plot_dir.mkdir(parents=True,exist_ok=True)
    products=[]
    def finish(identifier,fig,values,sources,family,interpretation,limitations):
        csv_path=out/"figures"/(identifier+"_values.csv")
        relative_sources=[str(p.relative_to(root)).replace("\\","/") for p in sources]
        hashes={str(p.relative_to(root)).replace("\\","/"):digest(p.read_bytes()) for p in sources}
        values=values.copy();values["source_artifacts"]=stable(hashes)
        write_csv(csv_path,values)
        outputs={}
        for ext in ["png","pdf","svg"]:
            path=plot_dir/(identifier+"."+ext)
            metadata={"Software":"RQ5 offline freeze"} if ext=="png" else {"Date":"2026-09-20"} if ext=="svg" else {"CreationDate":datetime(2026,9,20,tzinfo=timezone.utc),"ModDate":datetime(2026,9,20,tzinfo=timezone.utc)}
            fig.savefig(path,bbox_inches="tight",dpi=150,metadata=metadata)
            outputs[str(path.relative_to(root)).replace("\\","/")]=digest(path.read_bytes())
        plt.close(fig)
        manifest={"figure_id":identifier,"rq_identifier":"RQ5","evidence_family":family,"source_csv":str(csv_path.relative_to(root)).replace("\\","/"),"source_csv_sha256":digest(csv_path.read_bytes()),"source_artifacts":hashes,"figure_sha256":outputs,"generation_script":"src/analysis/rq5_final_freeze.py","config":"configs/rq5_final_freeze.yaml","scientific_interpretation":interpretation,"limitations":limitations}
        write_json(out/"figures"/(identifier+"_manifest.json"),manifest);products.append(manifest)
    metrics_path=out/"corrected/metrics.csv";m=pd.read_csv(metrics_path,keep_default_na=False,float_precision="round_trip")
    values=m.melt(id_vars=["dataset","recommender","profile_generator","condition"],value_vars=METRICS,var_name="metric",value_name="value")
    fig,ax=plt.subplots(figsize=(15,6));labels=[f"{x.dataset}\n{x.recommender.split(':')[0]}/{x.profile_generator.split(':')[0]}\n{x.condition.replace('profile_plus_history','H+P').replace('history_only','H').replace('profile_only','P')}" for x in m.itertuples()]
    for metric in METRICS:ax.plot(range(len(m)),m[metric],marker="o",label=metric)
    ax.set_xticks(range(len(m)),labels,rotation=65,ha="right",fontsize=7);ax.set_ylabel("Utility @5");ax.set_title("RQ5 corrected matched conditions - invalid rankings contribute zero");ax.legend()
    finish("fig_rq5_profile_conditions_main",fig,values,[metrics_path],"corrected","Absolute matched-condition utility","200-user full denominators; local protocol; no semantic-fidelity inference")
    bs_path=out/"corrected/bootstrap_results.csv";bs=pd.read_csv(bs_path,keep_default_na=False,float_precision="round_trip")
    bs["trajectory"]=bs.dataset+" | "+bs.recommender+" | "+bs.profile_generator;bs["contrast"]=bs.comparison+" / "+bs.metric
    pivot=bs.pivot(index="trajectory",columns="contrast",values="mean_delta")
    fig,ax=plt.subplots(figsize=(13,6));im=ax.imshow(pivot.to_numpy(),aspect="auto");fig.colorbar(im,ax=ax,label="Treatment minus History")
    ax.set_yticks(range(len(pivot)),pivot.index,fontsize=8);ax.set_xticks(range(len(pivot.columns)),[x.replace("_vs_history_only","").replace("profile_plus_history","H+P").replace("profile_only","P") for x in pivot.columns],rotation=20,ha="right",fontsize=8)
    for i,key in enumerate(pivot.index):
        for j,col in enumerate(pivot.columns):
            row=bs.loc[bs.trajectory.eq(key)&bs.contrast.eq(col)].iloc[0];mark=" +CI" if row.decision=="CI_SUPPORTED_POSITIVE" else " -CI" if row.decision=="CI_SUPPORTED_NEGATIVE" else ""
            ax.text(j,i,f"{row.mean_delta:+.3f}"+mark,ha="center",va="center",fontsize=8)
    ax.set_title("RQ5 corrected matched deltas - exploratory 95% paired bootstrap")
    finish("fig_rq5_profile_deltas_heatmap",fig,bs,[bs_path],"corrected","Matched substitution/addition effects relative to History","CI markers unadjusted for multiplicity; no general positive addition effect")
    r5path=out/"r5d/metrics.csv";m=pd.read_csv(r5path,float_precision="round_trip");fig,ax=plt.subplots(figsize=(11,5))
    for metric in METRICS:ax.plot(range(len(m)),m[metric],marker="o",label=metric)
    ax.set_xticks(range(len(m)),[x.dataset+"\n"+x.recommender.split(':')[0] for x in m.itertuples()]);ax.set_ylabel("Utility @5");ax.set_title("RQ5 R5D V2 - absolute Profile+History utility");ax.legend()
    finish("fig_rq5_profile_plus_history_extension",fig,m,[r5path],"r5d","Pipeline feasibility and absolute downstream utility","No aligned History-only counterfactual; not a causal profile benefit")
    evidence=Evidence(root)
    try:q=evidence.table("submission/results/tables/fig_rq5_profile_example_qualitative_values.csv",True)
    finally:evidence.close()
    # Keep only illustrative content and canonical member identity; discard historical absolute paths.
    q=q[["block_label","block_text","source_sha256"]].copy();q["evidence_member"]="research/results/llm/phase_f6l_profile_execution_20260801_010000/F6L-PROFILE-REQ-0415.json"
    fig,ax=plt.subplots(figsize=(12,5.5));ax.axis("off");blocks=[]
    for row in q.itertuples():blocks.append(row.block_label+"\n"+textwrap.fill(row.block_text[:1400],100))
    ax.text(0,1,"\n\n".join(blocks),va="top",fontsize=9);ax.set_title("RQ5 qualitative illustration - not semantic-fidelity validation")
    finish("fig_rq5_profile_example_qualitative",fig,q,[root/NS/"evidence_f6l.zip",root/NS/"evidence_figure_inputs.zip"],"illustrative","A real generated profile and interpretive cues","Qualitative illustration only; text shortened for display; no profile-quality validation")
    hp=out/"history_v2/metrics.csv";h=pd.read_csv(hp,float_precision="round_trip");fig,ax=plt.subplots(figsize=(13,6))
    for cond,g in h.groupby("condition",sort=True):
        ax.set_prop_cycle(None)
        style={"history_only":"-","profile_only":"--","profile_plus_history":":"}[cond]
        for metric in METRICS+["aplt_at_5","serendipity_at_5"]:ax.plot(g.history_length,g[metric],marker="o",linestyle=style,label=cond+" / "+metric)
    ax.set_xlabel("History length");ax.set_ylabel("Metric @5");ax.set_xticks(config["families"]["history_v2"]["history_lengths"]);ax.legend(bbox_to_anchor=(1.02,1),loc="upper left",fontsize=8);ax.set_title("RQ5 Beauty/Qwen - V2 deterministic analysis; local adapted popularity")
    finish("fig_rq5_history_length_profile_conditions",fig,h,[hp,out/"history_v2/migration_manifest.json"],"history_v2","Experimentally distinct history-length slice; PASS metrics only","Beauty/Qwen only; local adapted APLT/MostPop; V2 canonical ID tie-break; exploratory")
    write_json(out/"figures/figure_index.json",products)
    return products


def compare_tree(expected,observed):
    for path in observed.rglob("*"):
        if path.is_file():
            rel=path.relative_to(observed)
            require((expected/rel).is_file(),"missing authoritative artifact "+str(rel))
            require(path.read_bytes()==(expected/rel).read_bytes(),"reproduction bytes differ: "+str(rel))


def verify_hashes(root):
    manifest=json.loads((root/NS/"freeze_manifest.json").read_text())
    for rel,record in manifest["artifacts"].items():
        require(not Path(rel).is_absolute() and ".." not in Path(rel).parts,"unsafe manifest path")
        require(digest((root/rel).read_bytes())==record["sha256"],"package hash "+rel)
    evidence=Evidence(root)
    try:count=evidence.verify()
    finally:evidence.close()
    for m in json.loads((root/NS/"figures/figure_index.json").read_text()):
        for rel,h in {**m["source_artifacts"],**m["figure_sha256"],m["source_csv"]:m["source_csv_sha256"]}.items():
            require(digest((root/rel).read_bytes())==h,"figure provenance "+rel)
    return {"status":"PASS","artifacts":len(manifest["artifacts"]),"evidence_members":count}


def seal(root,config):
    """Seal the reviewed package; actual containing commit is attested after commit creation."""
    namespace=root/NS
    validation=json.loads((namespace/"validation.json").read_text())
    require(all(validation[k]["status"]=="PASS" for k in ["r5d","corrected","systematic","history_v2"]),"cannot seal failed analysis")
    artifacts={}
    paths=list(namespace.rglob("*"))+list((root/"results/plots/rq5_final_scientific_freeze").glob("*"))
    paths += [root/x for x in ["src/analysis/rq5_final_freeze.py","src/analysis/rq5_package_import.py","src/analysis/rq5_clean_room_check.py","src/evaluation/rq5_freeze_metrics.py","tests/test_rq5_final_freeze.py","configs/rq5_final_freeze.yaml","configs/rq5_freeze_requirements.txt","report/rq5_final_scientific_freeze.md","results/tables/rq5_primary_systematic_profile_reuse_audit.csv"]]
    excluded={"freeze_manifest.json","scientific_freeze_manifest.json","commit_provenance.json","clean_room_validation.json"}
    for path in sorted(set(paths)):
        if not path.is_file() or path.name in excluded:continue
        rel=path.relative_to(root).as_posix()
        artifacts[rel]={"sha256":digest(path.read_bytes()),"size":path.stat().st_size}
    import matplotlib
    environment={"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"pyyaml":yaml.__version__,"matplotlib":matplotlib.__version__,"platform":platform.platform(),"dependency_lock":"configs/rq5_freeze_requirements.txt"}
    manifest={"schema_version":2,"rq":"RQ5","status":"FINAL_SCIENTIFIC_FREEZE","base_commit":config["base_commit"],"source_commits":config["source_commits"],"freeze_commit":{"resolution":"commit_provenance.json","rule":"actual commit whose tree contains this exact freeze_manifest.json blob; never the pre-freeze checkout","self_reference_explanation":"Git commit cannot contain its own hash; a subsequent provenance-only commit records the containing commit"},"experiment_families":{"r5d":"SUPPORTING","corrected":"FOREGROUND","systematic":"SUPPORTING","history_v2":"EXPERIMENTALLY_DISTINCT","history_v1":"FORENSIC","f6l_statistics":"FORENSIC","pilot_v1":"SUPERSEDED","pilot_v2":"SUPPORTING_REUSE_DEPENDENCY"},"artifacts":artifacts,"evidence_inventory":NS+"/package_inventory.csv","external_bundles_required":[],"raw_evidence_storage":"Versioned deterministic ZIP archives in the package; each member has SHA256 and size in package_inventory.csv","historical_absolute_paths":"Archival original bytes may retain historical paths as documentary text; operational reproduction resolves only inventory members and repository-relative paths","environment":environment,"validation":validation,"limitations":json.loads((namespace/"scientific_synthesis.json").read_text())["limitations"],"external_reproduction_commands":["python -B src/analysis/rq5_final_freeze.py --config configs/rq5_final_freeze.yaml --mode verify","python -B src/analysis/rq5_final_freeze.py --config configs/rq5_final_freeze.yaml --mode reproduce --output-dir results/rq5_reproduction_check","python -B src/analysis/rq5_final_freeze.py --config configs/rq5_final_freeze.yaml --mode figures","python -B -m pytest tests/test_rq5_final_freeze.py -p no:cacheprovider"],"reproduction_environment":"Install exact versions in configs/rq5_freeze_requirements.txt using Python 3.12; no inference service required","test_results":"tests_pass documented in validation_tests.json; clean-room attestation recorded separately","raw_inference_unchanged":True,"no_new_inference":True}
    for family in ["corrected","systematic"]:
        fm={"family":family,"authority":manifest["experiment_families"][family],"base_commit":config["base_commit"],"source_commits":config["source_commits"],"freeze_commit":manifest["freeze_commit"],"raw_unchanged":True,"environment":environment,"validation":validation[family],"package_inventory":NS+"/package_inventory.csv","dependencies":["evidence_"+family+".zip","evidence_f6l.zip","evidence_pilot_v2.zip"]+(["evidence_corrected.zip"] if family=="systematic" else []),"artifact_hashes":{k:v for k,v in artifacts.items() if ("/"+family+"/") in k or Path(k).suffix in {".py",".yaml",".txt",".md"} or Path(k).name in {"package_inventory.csv","scientific_synthesis.json","validation.json"} or Path(k).name.startswith("evidence_")}}
        family_path=namespace/family/"scientific_freeze_manifest.json"
        write_json(family_path,fm)
        manifest["artifacts"][family_path.relative_to(root).as_posix()]={"sha256":digest(family_path.read_bytes()),"size":family_path.stat().st_size}
    write_json(namespace/"freeze_manifest.json",manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("configs/rq5_final_freeze.yaml"))
    parser.add_argument("--mode",choices=["recompute","figures","verify","reproduce","seal"],required=True)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format="%(message)s")
    config=yaml.safe_load((ROOT/args.config).read_text(encoding="utf-8"));require(config["allow_api_calls"] is False,"offline config required")
    out=ROOT/args.output_dir if args.output_dir else ROOT/NS
    if args.mode=="reproduce":
        require(args.output_dir is not None and out.resolve()!=(ROOT/NS).resolve(),"reproduction requires a separate --output-dir")
    if args.mode=="verify":print(json.dumps(verify_hashes(ROOT)));return
    if args.mode=="seal":seal(ROOT,config);print("RQ5 package manifest sealed");return
    if args.mode in {"recompute","reproduce"}:
        recompute(ROOT,config,out)
        if out.resolve()==(ROOT/NS).resolve():
            alias=ROOT/"results/tables/rq5_primary_systematic_profile_reuse_audit.csv"
            alias.write_bytes((out/"rq5_primary_systematic_profile_reuse_audit.csv").read_bytes())
    if args.mode=="figures":figures(ROOT,config,ROOT/NS,ROOT/"results/plots/rq5_final_scientific_freeze")
    if args.mode=="reproduce":
        require(args.output_dir is not None and out.resolve()!=(ROOT/NS).resolve(),"reproduction requires a separate --output-dir")
        compare_tree(ROOT/NS,out)
        print("RQ5 offline analytic reproduction PASS")

if __name__=="__main__":main()
