# Plan C Author Contact Request Draft

Subject: Artifact request for reproducing "Beyond Utility: Evaluating LLM as Recommender"

Dear authors,

I am conducting a university reproducibility project for your paper,
*Beyond Utility: Evaluating LLM as Recommender*. I am using the public
repository as the primary reference and keeping the original code read-only.

I would like to ask whether the processed artifacts linked from the repository
README are still available, or whether they have moved to a new location. In
particular, I am looking for the Beauty and Sports processed data files needed
by the evaluation scripts:

- `sequential_data.txt`
- `negative_samples.txt`
- `datamaps.json`
- `meta.json`

The Dropbox links currently referenced in the README appear difficult to
recover from my local checks. If there are updated links, checksums, or
recommended mirrors, I would be grateful for them.

I would also appreciate any clarification or artifacts for the full paper
replication:

- exact preprocessing scripts and random seeds;
- sampled user lists and any K-S test procedure details;
- candidate-set construction details and candidate ordering seeds;
- MovieLens-1M and LastFM processed artifacts, if released;
- missing SOTA/baseline output files, especially Sports and any ML-1M/LastFM
  outputs used for ranking or re-ranking;
- reranking candidate pools derived from BPRMF, GRU4Rec, SASRec, and LightGCN;
- any file-size or SHA-256 checksums for the released artifacts.

My goal is to report exact reproduction only when the evidence supports it. If
some artifacts are no longer available, I will label later work as a
reconstruction from public raw data rather than as an exact reproduction.

Thank you for your time and for releasing the repository.

Best regards,

[Your name]
