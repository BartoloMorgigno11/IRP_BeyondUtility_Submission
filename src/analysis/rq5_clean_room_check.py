"""Validate a tracked RQ5 export with an isolated copied runtime, no scientific fallbacks."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parents[2]
NS="results/rq5_final_scientific_freeze"
SEEDS=["numpy","pandas","PyYAML","matplotlib","pytest"]

def distributions():
    found={};pending=list(SEEDS)
    while pending:
        name=pending.pop();key=canonicalize_name(name)
        if key in found:continue
        dist=metadata.distribution(name);found[key]=dist
        for text in dist.requires or []:
            req=Requirement(text)
            if req.marker is None or req.marker.evaluate({"extra":""}):pending.append(req.name)
    return found

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--commit",default="HEAD")
    args=parser.parse_args();dest=args.destination.resolve()
    if dest.exists():raise FileExistsError("Clean-room destination must not exist")
    dest.mkdir(parents=True);checkout=dest/"checkout";checkout.mkdir()
    git=["git","-c","safe.directory="+ROOT.as_posix(),"-C",str(ROOT)]
    commit=subprocess.check_output(git+["rev-parse",args.commit],text=True).strip()
    # The path list comes from the committed manifest, never from untracked worktree contents.
    manifest=json.loads(subprocess.check_output(git+["show",commit+":"+NS+"/freeze_manifest.json"]))
    paths=list(manifest["artifacts"])+[NS+"/freeze_manifest.json","src/analysis/rq5_clean_room_check.py"]
    paths=sorted(set(paths))
    archive=dest/"tracked_package.tar"
    subprocess.run(git+["archive","--format=tar","--output="+str(archive),commit,"--",*paths],check=True)
    with tarfile.open(archive) as tar:tar.extractall(checkout,filter="data")
    # Copy only the Python runtime and installed distributions in the pinned dependency closure.
    # No original repository data, ignored outputs, bytecode caches or source trees are copied.
    runtime=dest/"runtime"
    shutil.copytree(Path(sys.base_prefix),runtime,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    site_source=Path(sys.prefix)/("Lib/site-packages" if os.name=="nt" else "lib/python"+str(sys.version_info.major)+"."+str(sys.version_info.minor)+"/site-packages")
    site_target=runtime/("Lib/site-packages" if os.name=="nt" else "lib/python"+str(sys.version_info.major)+"."+str(sys.version_info.minor)+"/site-packages")
    versions={}
    for name,dist in distributions().items():
        versions[name]=dist.version
        for entry in dist.files or []:
            original=Path(dist.locate_file(entry)).resolve()
            if not original.is_relative_to(site_source.resolve()) or not original.is_file() or original.suffix==".pyc":continue
            target=site_target/original.relative_to(site_source.resolve());target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
    interpreter=runtime/("python.exe" if os.name=="nt" else "bin/python3")
    # Deny reads under all project worktrees. The cloned runtime and checkout are outside them.
    forbidden=[str(ROOT.parent/name) for name in ["IRP_BeyondUtility_Project","IRP_BeyondUtility_Submission_Worktree","IRP_BeyondUtility_FinalSubmission_Worktree","IRP_BeyondUtility_RQ5_Finalization_Worktree"]]
    guard=dest/"guard.py"
    guard.write_text("""import builtins,io,os,pathlib,runpy,socket,sys
forbidden=[pathlib.Path(x).resolve() for x in %r]
original_open=builtins.open
original_io_open=io.open
def check(path):
    if isinstance(path,(str,bytes,os.PathLike)):
        resolved=pathlib.Path(os.fsdecode(path)).resolve()
        if any(resolved.is_relative_to(root) for root in forbidden):raise PermissionError('Forbidden original-worktree access: '+str(resolved))
def guarded_open(path,*args,**kwargs):check(path);return original_open(path,*args,**kwargs)
def guarded_io_open(path,*args,**kwargs):check(path);return original_io_open(path,*args,**kwargs)
def deny_network(*args,**kwargs):raise RuntimeError('Network/inference prohibited in clean-room validation')
builtins.open=guarded_open
io.open=guarded_io_open
socket.socket.connect=deny_network
socket.create_connection=deny_network
sys.path.insert(0,str(pathlib.Path.cwd()))
mode=sys.argv[1]
sys.argv=sys.argv[2:]
if mode=='pytest':
    import pytest
    raise SystemExit(pytest.main(sys.argv))
runpy.run_path(sys.argv[0],run_name='__main__')
""" % forbidden,encoding="utf-8")
    env=os.environ.copy();env.pop("PYTHONPATH",None);env.pop("PYTHONHOME",None)
    env.update(PYTHONDONTWRITEBYTECODE="1",PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",PYTHONHASHSEED="0",MPLCONFIGDIR=str(dest/"mplconfig"))
    commands=[
        ["script","src/analysis/rq5_final_freeze.py","--config","configs/rq5_final_freeze.yaml","--mode","verify"],
        ["script","src/analysis/rq5_final_freeze.py","--config","configs/rq5_final_freeze.yaml","--mode","reproduce","--output-dir","results/rq5_reproduction_check"],
        ["script","src/analysis/rq5_final_freeze.py","--config","configs/rq5_final_freeze.yaml","--mode","figures"],
        ["script","src/analysis/rq5_final_freeze.py","--config","configs/rq5_final_freeze.yaml","--mode","verify"],
        ["pytest","tests/test_rq5_final_freeze.py","-q","-p","no:cacheprovider","-o","addopts=","--junitxml=results/clean_room_tests.xml"]]
    results=[]
    for i,command in enumerate(commands):
        result=subprocess.run([str(interpreter),"-B",str(guard),*command],cwd=checkout,env=env,capture_output=True,text=True,encoding="utf-8",errors="replace")
        log_bytes=(result.stdout+result.stderr).encode("utf-8")
        (dest/f"step_{i+1}.log").write_bytes(log_bytes)
        print(f"Clean-room step {i+1}: exit={result.returncode}",flush=True)
        if result.returncode:print(result.stdout+result.stderr,flush=True)
        results.append({"arguments":command,"exit_code":result.returncode,"log_sha256":hashlib.sha256(log_bytes).hexdigest()})
        if result.returncode:break
    testfile=checkout/"results/clean_room_tests.xml";counts={}
    if testfile.exists():
        suites=ET.parse(testfile).getroot().findall("testsuite")
        counts={k:sum(int(s.get(k,0)) for s in suites) for k in ["tests","failures","errors","skipped"]}
    output={"status":"PASS" if len(results)==len(commands) and all(r["exit_code"]==0 for r in results) else "FAIL","validated_commit":commit,"exported_tracked_paths":len(paths),"export_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"original_worktree_access_denied":True,"network_denied":True,"isolated_runtime":True,"preexisting_caches_used":False,"external_scientific_dependencies":[],"dependency_versions":versions,"steps":results,"test_counts":counts}
    (dest/"clean_room_validation.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(output),flush=True)
    raise SystemExit(0 if output["status"]=="PASS" else 1)

if __name__=="__main__":main()
