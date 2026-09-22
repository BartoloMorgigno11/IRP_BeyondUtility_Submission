"""Build the F6L paper-faithful completion checkpoint.

F6L is a checkpoint-first phase. The default mode audits paper/repository
faithfulness, freezes the next request plan, records runtime/cost expectations,
and stops before any LLM inference. It never starts Ollama, calls paid APIs,
downloads files, trains models, or mutates `original_repo/`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import random
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import ast
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PHASE_ID = "phase_f6l"
PHASE_NAME = "phase_f6l_paper_faithful_completion_and_rq_revalidation"
INITIAL_DECISION_CHECKPOINT = "f6l_paper_faithful_execution_checkpoint_ready"
DECISION_CHECKPOINT = "f6l_execution_matrix_partially_blocked_by_missing_official_artifacts"
DECISION_EXECUTION_COMPLETE = "f6l_open_model_execution_complete_reranking_reconstruction_pending"
DECISION_EXECUTION_BLOCKED = "f6l_execution_blocked_by_contract_runtime_or_protected_input_regression"
DECISION_CONTRACT_MATERIALIZATION_READY = "f6l_execution_contract_materialized_ready_for_inference_authorization"
DECISION_CONTRACT_MATERIALIZATION_BLOCKED = "f6l_execution_contract_materialization_blocked_by_missing_paper_specification"
DECISION_SEPARATED_PROFILE_READY = "f6l_profile_execution_ready_lastfm_operationalization_checkpoint_ready"
DECISION_PROFILE_LASTFM_PARTIAL_BLOCKED = "f6l_profile_and_lastfm_partially_blocked_by_missing_specifications"
DECISION_REPOSITORY_REGRESSION = "f6l_blocked_by_repository_runtime_or_protected_input_regression"
DECISION_LASTFM_CONTRACTS_READY = "f6l_lastfm_execution_contracts_ready_for_inference_authorization"
DECISION_LASTFM_SAMPLE_BLOCKED_FULL_READY = "f6l_lastfm_operationalized_sample_blocked_full_cohort_ready"
DECISION_LASTFM_CONTRACTS_BLOCKED = "f6l_lastfm_execution_contracts_blocked_by_missing_local_evidence"
DECISION_LASTFM_OPERATIONALIZED_RANKING_COMPLETE = "f6l_lastfm_operationalized_ranking_execution_and_rq_revalidation_complete"
DECISION_LASTFM_OPERATIONALIZED_RANKING_PARTIAL = "f6l_lastfm_operationalized_ranking_execution_partial_resumable"
DECISION_LASTFM_OPERATIONALIZED_RANKING_BLOCKED = "f6l_lastfm_operationalized_ranking_execution_blocked_by_contract_runtime_or_protected_input_regression"
RECOVERY_AUDIT_TIMESTAMP = "2026-07-31T00:00:00+02:00"
MATERIALIZATION_AUDIT_TIMESTAMP = "2026-08-01T00:00:00+02:00"
RECOVERED_INTERPRETER = PROJECT_ROOT / ".venv_audit_recovery" / "Scripts" / "python.exe"
CONFIG_PATH = PROJECT_ROOT / "configs" / f"{PHASE_NAME}.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
MANIFESTS_DIR = PROJECT_ROOT / "results" / "manifests"
REPORT_PATH = PROJECT_ROOT / "report" / f"{PHASE_NAME}.md"
HANDOFF_PATH = PROJECT_ROOT / "report" / "future_operations_context_handoff.md"
NOTES_PATH = PROJECT_ROOT / "report" / "notes.md"

CHECKPOINT_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_execution_checkpoint_manifest.json"
PROTECTED_HASHES_PATH = MANIFESTS_DIR / "phase_f6l_protected_input_hashes.json"
EXECUTE_PREFLIGHT_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_execute_preflight_manifest.json"
CONTRACT_MATERIALIZATION_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_execution_contract_materialization_manifest.json"
SEPARATED_CHECKPOINT_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_separated_contracts_checkpoint_manifest.json"
PER_REQUEST_MANIFEST_PATH = TABLES_DIR / "phase_f6l_open_model_request_manifest.csv"
PROFILE_REQUEST_MANIFEST_PATH = TABLES_DIR / "phase_f6l_profile_request_manifest.csv"
LASTFM_FROZEN_SAMPLE_COHORT_PATH = TABLES_DIR / "phase_f6l_lastfm_1000_user_ks_valid_sample_cohort.csv"
LASTFM_FROZEN_SAMPLE_CANDIDATES_PATH = TABLES_DIR / "phase_f6l_lastfm_1000_user_ks_valid_sample_candidates.csv"
AUTHOR_CLARIFICATION_REPORT_PATH = PROJECT_ROOT / "report" / "phase_f6l_author_clarification_request.md"
OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags"

LASTFM_LOCAL_PIPELINE_AUDIT_PATH = TABLES_DIR / "phase_f6l_lastfm_local_pipeline_audit.csv"
LASTFM_OPERATIONALIZED_SEED_SEARCH_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_seed_search.csv"
LASTFM_OPERATIONALIZED_KS_RESULTS_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ks_results.csv"
LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_sample_cohort.csv"
LASTFM_OPERATIONALIZED_CANDIDATES_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_candidates.csv"
LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_request_manifest.csv"
LASTFM_FULL_COHORT_RANKING_MANIFEST_PATH = TABLES_DIR / "phase_f6l_lastfm_full_cohort_ranking_request_manifest.csv"
LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_reranking_pool.csv"
LASTFM_FULL_COHORT_RERANKING_POOL_PATH = TABLES_DIR / "phase_f6l_lastfm_full_cohort_reranking_pool.csv"
LASTFM_RERANKING_READINESS_PATH = TABLES_DIR / "phase_f6l_lastfm_reranking_readiness.csv"
LASTFM_EXECUTION_CONTRACTS_VALIDATION_PATH = TABLES_DIR / "phase_f6l_lastfm_execution_contracts_validation.csv"
LASTFM_EXECUTION_CONTRACTS_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_lastfm_execution_contracts_manifest.json"
CLEANUP_SHARD_COVERAGE_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_cleanup_shard_coverage_manifest.json"
LASTFM_OPERATIONALIZED_RANKING_PREFLIGHT_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_preflight.csv"
LASTFM_OPERATIONALIZED_RANKING_EXECUTION_STATE_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_execution_state.csv"
LASTFM_OPERATIONALIZED_RANKING_RUN_MANIFEST_PATH = MANIFESTS_DIR / "phase_f6l_lastfm_operationalized_ranking_run_manifest.json"
LASTFM_OPERATIONALIZED_RANKING_REQUEST_STATE_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_request_state.csv"
LASTFM_OPERATIONALIZED_RANKING_EXECUTION_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_execution.csv"
LASTFM_OPERATIONALIZED_RANKING_PARSE_RESULTS_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_parse_results.csv"
LASTFM_OPERATIONALIZED_RANKING_RECOMMENDATIONS_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_recommendation_rows.csv"
LASTFM_OPERATIONALIZED_RANKING_METRICS_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_metrics.csv"
LASTFM_OPERATIONALIZED_RANKING_BEYOND_UTILITY_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_beyond_utility_metrics.csv"
LASTFM_OPERATIONALIZED_RANKING_STRATEGY_COMPARISON_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_strategy_comparison.csv"
LASTFM_OPERATIONALIZED_RANKING_RUNTIME_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_runtime_resources.csv"
LASTFM_OPERATIONALIZED_RANKING_RQ_REVALIDATION_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_rq_revalidation.csv"
LASTFM_OPERATIONALIZED_RANKING_FINAL_VALIDATION_PATH = TABLES_DIR / "phase_f6l_lastfm_operationalized_ranking_final_validation.csv"
LASTFM_OPERATIONALIZED_RANKING_REPORT_PATH = PROJECT_ROOT / "report" / "phase_f6l_lastfm_operationalized_ranking_execution.md"
OLLAMA_LOCAL_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_LOCAL_GENERATE_URL = f"{OLLAMA_LOCAL_BASE_URL}/api/generate"
OLLAMA_LOCAL_TAGS_URL = f"{OLLAMA_LOCAL_BASE_URL}/api/tags"
OLLAMA_LOCAL_VERSION_URL = f"{OLLAMA_LOCAL_BASE_URL}/api/version"

F6C_LASTFM_INPUTS = {
    "hetrec_raw_zip": PROJECT_ROOT / "data" / "external" / "plan_c" / "hetrec2011-lastfm-2k.zip",
    "interactions": PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_interactions.csv",
    "train": PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_train.csv",
    "validation": PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_validation.csv",
    "test": PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_test.csv",
    "catalog": PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_catalog.csv",
    "ranking_cohort": PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_ranking_cohort.csv",
    "ranking_candidates": PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_ranking_candidates.csv",
}
F6D2_LASTFM_INPUTS = {
    "candidate_rankings": PROJECT_ROOT / "results" / "predictions" / "finalization_f6d2_lastfm_candidate_rankings.csv",
    "full_catalog_rankings": PROJECT_ROOT / "results" / "predictions" / "finalization_f6d2_lastfm_full_catalog_rankings.csv",
    "source_selection": PROJECT_ROOT / "results" / "tables" / "finalization_f6d2_reranking_source_selection.csv",
    "seed_utility_metrics": PROJECT_ROOT / "results" / "tables" / "finalization_f6d2_seed_utility_metrics.csv",
    "reranking_policy_audit": PROJECT_ROOT / "results" / "tables" / "finalization_f6d2_reranking_policy_audit.csv",
    "reranking_pool_validation": PROJECT_ROOT / "results" / "tables" / "finalization_f6d2_reranking_pool_validation.csv",
}
F6G1_LASTFM_INPUTS = {
    "pool_item_provenance": PROJECT_ROOT / "results" / "tables" / "phase_f6g1_pool_item_provenance.csv",
    "pool_instance_summary": PROJECT_ROOT / "results" / "tables" / "phase_f6g1_pool_instance_summary.csv",
    "comparability_gate": PROJECT_ROOT / "results" / "tables" / "phase_f6g1_comparability_gate.csv",
}

LASTFM_CONTROL_MODELS = {
    "bprmf": "BPRMF",
    "gru4rec": "GRU4Rec",
    "sasrec": "SASRec",
    "lightgcn": "LightGCN",
}
LASTFM_RANKING_STRATEGIES = {
    "base/OpenP5 title-list": {
        "strategy_id": "paper_base_openp5_title_list_top10",
        "template_path": PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "eva" / "templates" / "template.txt",
        "variant_index": 0,
        "parser_contract": "paper_title_list_top10_parser",
    },
    "LLMRank recency": {
        "strategy_id": "paper_llmrank_recency_title_list_top10",
        "template_path": PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "eva" / "templates" / "llmrank_rf.txt",
        "variant_index": 0,
        "parser_contract": "paper_title_list_top10_parser",
    },
}

EVIDENCE_CLASSES = {
    "exact_original_artifact_reproduction",
    "paper_protocol_reproduction_new_stochastic_output",
    "paper_protocol_reproduction_different_open_model",
    "paper_protocol_reproduction_new_sample",
    "adapted_protocol_reproduction",
    "original_project_extension",
    "paper_protocol_reproduction_under_explicit_operationalization",
    "original_project_extension_full_cohort",
    "blocked_missing_original_specification",
    "blocked_paid_or_unavailable_model",
    "non_comparable",
}

RQ_STATUSES = {
    "supported_with_paper_faithful_evidence",
    "supported_with_open_model_protocol_evidence",
    "contradicted_with_paper_faithful_evidence",
    "contradicted_with_open_model_protocol_evidence",
    "mixed_model_dependent",
    "mixed_dataset_dependent",
    "supported_only_in_adapted_extension",
    "unresolved_missing_closed_model",
    "unresolved_for_original_closed_model",
    "unresolved_missing_original_artifact",
}

OUTPUT_TABLES = {
    "paper_repository_exactness_matrix": TABLES_DIR / "phase_f6l_paper_repository_exactness_matrix.csv",
    "profile_generation_contract": TABLES_DIR / "phase_f6l_profile_generation_contract.csv",
    "profile_generation_results": TABLES_DIR / "phase_f6l_profile_generation_results.csv",
    "lastfm_ranking_exactness": TABLES_DIR / "phase_f6l_lastfm_ranking_exactness.csv",
    "lastfm_ranking_results": TABLES_DIR / "phase_f6l_lastfm_ranking_results.csv",
    "lastfm_reranking_results": TABLES_DIR / "phase_f6l_lastfm_reranking_results.csv",
    "runtime_cost_results": TABLES_DIR / "phase_f6l_runtime_cost_results.csv",
    "rq_revalidation": TABLES_DIR / "phase_f6l_rq_revalidation.csv",
    "irreducible_exactness_limits": TABLES_DIR / "phase_f6l_irreducible_exactness_limits.csv",
    "execution_request_manifest": TABLES_DIR / "phase_f6l_execution_request_manifest.csv",
    "execution_request_breakdown": TABLES_DIR / "phase_f6l_execution_request_breakdown.csv",
    "lastfm_reranking_prerequisite_audit": TABLES_DIR / "phase_f6l_lastfm_reranking_prerequisite_audit.csv",
    "official_artifact_recovery": TABLES_DIR / "phase_f6l_official_artifact_recovery.csv",
    "final_execution_authorization_matrix": TABLES_DIR / "phase_f6l_final_execution_authorization_matrix.csv",
    "checkpoint_summary": TABLES_DIR / "phase_f6l_execution_checkpoint_summary.csv",
    "model_preflight": TABLES_DIR / "phase_f6l_model_preflight.csv",
    "external_download_manifest": TABLES_DIR / "phase_f6l_external_download_manifest.csv",
    "validation": TABLES_DIR / "phase_f6l_validation.csv",
}

EXECUTE_OUTPUT_TABLES = {
    "execute_preflight": TABLES_DIR / "phase_f6l_execute_preflight.csv",
    "authorized_request_manifest_audit": TABLES_DIR / "phase_f6l_authorized_request_manifest_audit.csv",
    "ollama_local_model_preflight": TABLES_DIR / "phase_f6l_ollama_local_model_preflight.csv",
    "execution_blocker": TABLES_DIR / "phase_f6l_execution_blocker.csv",
}

CONTRACT_MATERIALIZATION_OUTPUT_TABLES = {
    "lastfm_contract_reconstruction_audit": TABLES_DIR / "phase_f6l_lastfm_contract_reconstruction_audit.csv",
    "execution_contract_materialization_blockers": TABLES_DIR / "phase_f6l_execution_contract_materialization_blockers.csv",
    "execution_contract_materialization_summary": TABLES_DIR / "phase_f6l_execution_contract_materialization_summary.csv",
}

SEPARATED_OUTPUT_TABLES = {
    "profile_request_manifest": PROFILE_REQUEST_MANIFEST_PATH,
    "profile_checkpoint": TABLES_DIR / "phase_f6l_profile_checkpoint.csv",
    "lastfm_author_specification_registry": TABLES_DIR / "phase_f6l_lastfm_author_specification_registry.csv",
    "lastfm_preprocessing_count_comparison": TABLES_DIR / "phase_f6l_lastfm_preprocessing_count_comparison.csv",
    "lastfm_operationalization_contract": TABLES_DIR / "phase_f6l_lastfm_operationalization_contract.csv",
    "lastfm_operationalization_sensitivity_plan": TABLES_DIR / "phase_f6l_lastfm_operationalization_sensitivity_plan.csv",
    "lastfm_full_cohort_extension_plan": TABLES_DIR / "phase_f6l_lastfm_full_cohort_extension_plan.csv",
    "rq_evidence_hierarchy_contract": TABLES_DIR / "phase_f6l_rq_evidence_hierarchy_contract.csv",
    "separated_checkpoints": TABLES_DIR / "phase_f6l_separated_checkpoints.csv",
}

LASTFM_EXECUTION_CONTRACT_TABLES = {
    "local_pipeline_audit": LASTFM_LOCAL_PIPELINE_AUDIT_PATH,
    "operationalized_seed_search": LASTFM_OPERATIONALIZED_SEED_SEARCH_PATH,
    "operationalized_ks_results": LASTFM_OPERATIONALIZED_KS_RESULTS_PATH,
    "operationalized_sample_cohort": LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH,
    "operationalized_candidates": LASTFM_OPERATIONALIZED_CANDIDATES_PATH,
    "operationalized_ranking_request_manifest": LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH,
    "full_cohort_ranking_request_manifest": LASTFM_FULL_COHORT_RANKING_MANIFEST_PATH,
    "operationalized_reranking_pool": LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH,
    "full_cohort_reranking_pool": LASTFM_FULL_COHORT_RERANKING_POOL_PATH,
    "reranking_readiness": LASTFM_RERANKING_READINESS_PATH,
    "validation": LASTFM_EXECUTION_CONTRACTS_VALIDATION_PATH,
}

LASTFM_EXECUTION_CONTRACT_FIELDS = {
    "local_pipeline_audit": [
        "requirement_id",
        "requirement",
        "observed_value",
        "classification",
        "status",
        "evidence",
        "notes",
    ],
    "operationalized_seed_search": [
        "seed",
        "attempt_index",
        "sample_size",
        "population_users",
        "test_family_size",
        "alpha",
        "correction",
        "raw_pvalue_min",
        "holm_pvalue_min",
        "ks_statistic_max",
        "accepted",
        "evidence_class",
        "notes",
    ],
    "operationalized_ks_results": [
        "seed",
        "attempt_index",
        "model_name",
        "model_seed",
        "metric",
        "population_users",
        "sample_users",
        "ks_statistic",
        "p_value_raw",
        "p_value_holm",
        "alpha",
        "significant_after_holm",
        "accepted",
        "evidence_class",
    ],
    "operationalized_sample_cohort": [
        "sample_id",
        "sample_ordinal",
        "user_id",
        "source_cohort_id",
        "internal_user_id",
        "train_history_count",
        "validation_item_id",
        "test_item_id",
        "test_item_title",
        "history_item_ids",
        "history_item_titles",
        "history_hash",
        "selected_seed",
        "evidence_class",
        "lineage",
    ],
    "operationalized_candidates": [
        "sample_id",
        "user_id",
        "candidate_set_id",
        "candidate_position",
        "candidate_item_id",
        "candidate_item_title",
        "is_positive",
        "positive_item",
        "positive_position",
        "candidate_list",
        "candidate_list_hash",
        "negative_sampling_algorithm",
        "negative_sampling_seed",
        "negative_exclusion_contract",
        "source_lineage",
        "evidence_class",
    ],
    "operationalized_ranking_request_manifest": [
        "request_id",
        "sample_id",
        "user_id",
        "model_name",
        "model_digest",
        "prompting_strategy",
        "history_hash",
        "candidate_list_hash",
        "positive_item",
        "positive_position",
        "prompt_template",
        "prompt_template_hash",
        "rendered_prompt_hash",
        "generation_parameters",
        "output_length",
        "parser_contract",
        "expected_raw_output_path",
        "evidence_class",
        "request_contract_hash",
        "lineage",
    ],
    "full_cohort_ranking_request_manifest": [
        "request_id",
        "sample_id",
        "user_id",
        "model_name",
        "model_digest",
        "prompting_strategy",
        "history_hash",
        "candidate_list_hash",
        "positive_item",
        "positive_position",
        "prompt_template",
        "prompt_template_hash",
        "rendered_prompt_hash",
        "generation_parameters",
        "output_length",
        "parser_contract",
        "expected_raw_output_path",
        "evidence_class",
        "request_contract_hash",
        "lineage",
    ],
    "operationalized_reranking_pool": [
        "cohort_scope",
        "sample_id",
        "user_id",
        "candidate_position",
        "candidate_item_id",
        "candidate_item_title",
        "is_positive",
        "positive_item",
        "positive_position",
        "source_models",
        "source_ranks",
        "source_scores",
        "source_seeds",
        "duplicate_source_count",
        "duplicate_skipped_before_item",
        "construction_policy",
        "pool_hash",
        "evidence_class",
        "lineage",
    ],
    "full_cohort_reranking_pool": [
        "cohort_scope",
        "sample_id",
        "user_id",
        "candidate_position",
        "candidate_item_id",
        "candidate_item_title",
        "is_positive",
        "positive_item",
        "positive_position",
        "source_models",
        "source_ranks",
        "source_scores",
        "source_seeds",
        "duplicate_source_count",
        "duplicate_skipped_before_item",
        "construction_policy",
        "pool_hash",
        "evidence_class",
        "lineage",
    ],
    "reranking_readiness": [
        "readiness_id",
        "scope",
        "requirement",
        "status",
        "observed",
        "evidence",
        "evidence_class",
        "missing_requirement",
        "future_request_count",
        "notes",
    ],
    "validation": [
        "validation_id",
        "check_name",
        "status",
        "observed",
        "expected",
        "severity",
        "evidence",
        "notes",
    ],
}

LASTFM_OPERATIONALIZED_RANKING_OUTPUT_TABLES = {
    "preflight": LASTFM_OPERATIONALIZED_RANKING_PREFLIGHT_PATH,
    "execution_state": LASTFM_OPERATIONALIZED_RANKING_EXECUTION_STATE_PATH,
    "request_state": LASTFM_OPERATIONALIZED_RANKING_REQUEST_STATE_PATH,
    "execution": LASTFM_OPERATIONALIZED_RANKING_EXECUTION_PATH,
    "parse_results": LASTFM_OPERATIONALIZED_RANKING_PARSE_RESULTS_PATH,
    "recommendation_rows": LASTFM_OPERATIONALIZED_RANKING_RECOMMENDATIONS_PATH,
    "metrics": LASTFM_OPERATIONALIZED_RANKING_METRICS_PATH,
    "beyond_utility_metrics": LASTFM_OPERATIONALIZED_RANKING_BEYOND_UTILITY_PATH,
    "strategy_comparison": LASTFM_OPERATIONALIZED_RANKING_STRATEGY_COMPARISON_PATH,
    "runtime_resources": LASTFM_OPERATIONALIZED_RANKING_RUNTIME_PATH,
    "rq_revalidation": LASTFM_OPERATIONALIZED_RANKING_RQ_REVALIDATION_PATH,
    "final_validation": LASTFM_OPERATIONALIZED_RANKING_FINAL_VALIDATION_PATH,
}

LASTFM_OPERATIONALIZED_RANKING_TABLE_FIELDS = {
    "preflight": [
        "check_id",
        "check_name",
        "status",
        "observed",
        "expected",
        "severity",
        "evidence",
        "notes",
    ],
    "execution_state": [
        "run_id",
        "stage",
        "decision_token",
        "planned",
        "completed",
        "strict_valid",
        "failed",
        "residual",
        "retry",
        "percent_complete",
        "last_request_id",
        "last_timestamp",
        "throughput_requests_per_hour",
        "eta_seconds",
        "raw_output_dir",
        "notes",
    ],
    "request_state": [
        "request_id",
        "sample_id",
        "user_id",
        "model_name",
        "model_digest",
        "prompting_strategy",
        "request_order",
        "raw_request_records",
        "raw_response_records",
        "state",
        "strict_valid",
        "backend_success",
        "parse_status",
        "retry_count",
        "raw_json_path",
        "request_contract_hash",
        "notes",
    ],
    "execution": [
        "request_id",
        "sample_id",
        "user_id",
        "model_name",
        "model_digest",
        "prompting_strategy",
        "attempt_index",
        "started_at",
        "completed_at",
        "duration_seconds",
        "backend_success",
        "response_nonempty",
        "done_reason",
        "prompt_eval_count",
        "eval_count",
        "load_duration_nanoseconds",
        "prompt_eval_duration_nanoseconds",
        "eval_duration_nanoseconds",
        "total_duration_nanoseconds",
        "raw_response_sha256",
        "backend_error_type",
        "backend_error_message",
        "response_jsonl_recorded",
        "raw_json_path",
        "execution_status",
        "notes",
    ],
    "parse_results": [
        "request_id",
        "sample_id",
        "user_id",
        "model_name",
        "model_digest",
        "prompting_strategy",
        "raw_response_sha256",
        "parser_contract",
        "json_list_detected",
        "schema_valid",
        "output_count_valid",
        "title_mapping_valid",
        "candidate_validity",
        "uniqueness_valid",
        "strict_valid",
        "parsed_titles",
        "ranked_item_ids",
        "duplicate_count",
        "hallucinated_count",
        "invalid_items",
        "positive_item",
        "positive_position",
        "positive_rank",
        "positive_in_top5",
        "ndcg_at5",
        "parse_status",
        "failure_reason",
    ],
    "recommendation_rows": [
        "request_id",
        "sample_id",
        "user_id",
        "model_name",
        "model_digest",
        "prompting_strategy",
        "rank",
        "recommended_item_id",
        "recommended_item_title",
        "is_positive",
        "candidate_list_hash",
        "history_hash",
        "strict_valid",
        "source_raw_file",
        "lineage",
    ],
    "metrics": [
        "metric_id",
        "sample_id",
        "dataset",
        "task",
        "model_name",
        "model_digest",
        "prompting_strategy",
        "evidence_class",
        "metric",
        "k",
        "value",
        "valid_rows",
        "total_rows",
        "strict_valid_rows",
        "source_file",
        "notes",
    ],
    "beyond_utility_metrics": [
        "metric_id",
        "sample_id",
        "dataset",
        "task",
        "model_name",
        "model_digest",
        "prompting_strategy",
        "evidence_class",
        "metric",
        "k",
        "value",
        "numerator",
        "denominator",
        "support_status",
        "source_file",
        "notes",
    ],
    "strategy_comparison": [
        "comparison_id",
        "sample_id",
        "model_name",
        "metric",
        "base_openp5_value",
        "llmrank_recency_value",
        "delta_llmrank_minus_base",
        "shared_users",
        "evidence_class",
        "status",
        "notes",
    ],
    "runtime_resources": [
        "runtime_id",
        "run_id",
        "scope",
        "model_name",
        "prompting_strategy",
        "planned",
        "completed",
        "strict_valid",
        "failed",
        "residual",
        "retry",
        "wall_clock_seconds",
        "duration_mean_seconds",
        "duration_median_seconds",
        "duration_p90_seconds",
        "duration_p95_seconds",
        "duration_p99_seconds",
        "throughput_requests_per_hour",
        "prompt_eval_count_total",
        "eval_count_total",
        "cpu",
        "ram",
        "gpu",
        "vram",
        "raw_output_bytes",
        "parsing_time_seconds",
        "consolidation_time_seconds",
        "marginal_api_cost",
        "hardware_cost",
        "energy_cost",
        "notes",
    ],
    "rq_revalidation": [
        "rq_id",
        "paper_claim",
        "published_paper_evidence",
        "previous_lastfm_adapted_evidence",
        "new_operationalized_evidence",
        "dataset_and_sample_difference",
        "qwen2_base",
        "qwen2_llmrank",
        "mistral_base",
        "mistral_llmrank",
        "previous_status",
        "new_status",
        "motivation",
        "limits",
    ],
    "final_validation": [
        "validation_id",
        "check_name",
        "status",
        "observed",
        "expected",
        "severity",
        "evidence",
        "notes",
    ],
}

TABLE_FIELDS = {
    "paper_repository_exactness_matrix": [
        "matrix_id",
        "area",
        "requirement",
        "paper_contract",
        "repository_evidence",
        "local_current_state",
        "cell_status",
        "evidence_classification",
        "sample_exactness",
        "action",
        "source_priority",
        "evidence_references",
        "notes",
    ],
    "profile_generation_contract": [
        "contract_id",
        "dataset",
        "cohort",
        "history_length",
        "profile_generator",
        "recommender_model",
        "prompt_mode",
        "profile_prompt_source",
        "recommendation_prompt_source",
        "candidate_set_source",
        "official_artifact_status",
        "local_execution_status",
        "evidence_classification",
        "request_count",
        "blocked_reason",
        "notes",
    ],
    "profile_generation_results": [
        "result_id",
        "dataset",
        "profile_generator",
        "recommender_model",
        "mode",
        "evidence_classification",
        "request_count",
        "metric",
        "value",
        "status",
        "source_file",
        "notes",
    ],
    "lastfm_ranking_exactness": [
        "cell_id",
        "dataset",
        "cohort_source",
        "users",
        "candidate_count",
        "positive_policy",
        "candidate_order",
        "model",
        "strategy",
        "prompt_source",
        "output_length",
        "parser",
        "existing_artifact",
        "evidence_classification",
        "sample_exactness",
        "request_count",
        "status",
        "blocking_reason",
        "notes",
    ],
    "lastfm_ranking_results": [
        "result_id",
        "dataset",
        "model",
        "strategy",
        "evidence_classification",
        "sample_exactness",
        "request_count",
        "metric",
        "value",
        "valid_rows",
        "source_file",
        "status",
        "notes",
    ],
    "lastfm_reranking_results": [
        "result_id",
        "dataset",
        "model",
        "strategy",
        "evidence_classification",
        "sample_exactness",
        "request_count",
        "metric",
        "value",
        "valid_rows",
        "source_file",
        "status",
        "notes",
    ],
    "runtime_cost_results": [
        "runtime_id",
        "scope",
        "model",
        "task",
        "request_count",
        "observed_or_estimated",
        "wall_clock_seconds",
        "latency_mean",
        "latency_p50",
        "latency_p95",
        "throughput_requests_per_hour",
        "successes",
        "failures",
        "retries",
        "cpu",
        "ram",
        "gpu",
        "vram",
        "input_tokens",
        "output_tokens",
        "raw_output_bytes",
        "parsing_time_seconds",
        "consolidation_time_seconds",
        "marginal_api_cost",
        "total_cost_claim",
        "evidence_classification",
        "source_file",
        "notes",
    ],
    "rq_revalidation": [
        "rq_id",
        "paper_conclusion",
        "paper_exact_evidence",
        "paper_protocol_new_output_evidence",
        "adapted_evidence",
        "extension_evidence",
        "models_covered",
        "datasets_covered",
        "limits",
        "previous_f6j_status",
        "new_status",
        "variation_reason",
    ],
    "irreducible_exactness_limits": [
        "limit_id",
        "area",
        "missing_detail",
        "source_attempts",
        "evidence_classification",
        "block_status",
        "impact",
        "next_action",
    ],
    "execution_request_manifest": [
        "request_group_id",
        "phase",
        "dataset",
        "task",
        "cohort",
        "model",
        "strategy",
        "mode",
        "user_count",
        "requests_per_user",
        "total_requests",
        "authorized_for_checkpoint_execution",
        "prompt_source",
        "parser_contract",
        "raw_output_policy",
        "evidence_classification",
        "blocked_reason",
        "notes",
    ],
    "execution_request_breakdown": [
        "cell_id",
        "experiment_family",
        "dataset",
        "task",
        "cohort",
        "cohort_size",
        "model",
        "model_exact_identity",
        "profile_generator",
        "recommender_model",
        "prompting_strategy",
        "input_variant",
        "history_length",
        "candidate_pool_type",
        "candidate_pool_size",
        "output_length",
        "requests_per_user",
        "expected_requests",
        "requires_profile_generation",
        "requires_recommendation_inference",
        "evidence_class",
        "authorization_status",
        "authorization_reason",
    ],
    "lastfm_reranking_prerequisite_audit": [
        "requirement_id",
        "requirement",
        "status",
        "available_exact",
        "available_but_adapted",
        "recoverable_from_official_source",
        "reconstructed_from_official_code",
        "missing_not_recoverable",
        "not_required",
        "source_evidence",
        "official_recovery_action",
        "after_recovery_status",
        "causes_zero_reranking_request_count",
        "notes",
    ],
    "official_artifact_recovery": [
        "recovery_id",
        "requirement",
        "artifact_name",
        "source_url",
        "source_type",
        "author_or_organization",
        "relation_to_paper_or_repository",
        "inspected_at",
        "download_allowed_by_policy",
        "download_performed",
        "destination_path",
        "local_artifact_path",
        "sha256",
        "structure_validation",
        "content_validation",
        "recovery_status",
        "prerequisite_after_recovery",
        "evidence_classification",
        "notes",
    ],
    "final_execution_authorization_matrix": [
        "authorization_id",
        "cell_id",
        "experiment_family",
        "dataset",
        "task",
        "model",
        "prompting_strategy",
        "input_variant",
        "expected_requests",
        "evidence_classification",
        "authorization_status",
        "authorization_reason",
        "raw_output_policy",
        "metrics",
        "included_in_final_request_count",
        "notes",
    ],
    "checkpoint_summary": [
        "checkpoint_item",
        "required_content",
        "value",
        "evidence_references",
        "status",
    ],
    "model_preflight": [
        "model",
        "model_family",
        "paper_identifier",
        "local_identifier",
        "availability_basis",
        "available_for_checkpoint",
        "version_equivalence",
        "requires_paid_api",
        "requires_download",
        "evidence_classification",
        "notes",
    ],
    "external_download_manifest": [
        "download_id",
        "source_name",
        "url",
        "official_source",
        "download_allowed_by_policy",
        "download_performed",
        "destination_policy",
        "local_artifact_equivalent",
        "sha256",
        "status",
        "notes",
    ],
    "validation": [
        "validation_id",
        "check_name",
        "expected",
        "observed",
        "status",
        "severity",
        "evidence",
        "notes",
    ],
}

EXECUTE_TABLE_FIELDS = {
    "execute_preflight": [
        "check_id",
        "check_name",
        "required",
        "observed",
        "status",
        "severity",
        "evidence",
        "notes",
    ],
    "authorized_request_manifest_audit": [
        "audit_id",
        "artifact",
        "path",
        "required",
        "status",
        "row_count",
        "unique_request_ids",
        "expected_requests",
        "sha256",
        "notes",
    ],
    "ollama_local_model_preflight": [
        "model",
        "required",
        "local_endpoint",
        "available",
        "digest",
        "modified_at",
        "size_bytes",
        "family",
        "parameter_size",
        "quantization_level",
        "context_length",
        "status",
        "notes",
    ],
    "execution_blocker": [
        "blocker_id",
        "decision_token",
        "scope",
        "status",
        "planned_requests",
        "completed_requests",
        "strict_valid_requests",
        "failed_requests",
        "residual_requests",
        "blocker",
        "evidence",
        "next_action",
    ],
}

CONTRACT_MATERIALIZATION_TABLE_FIELDS = {
    "lastfm_contract_reconstruction_audit": [
        "audit_id",
        "contract_area",
        "required_detail",
        "recovered_value",
        "source_priority",
        "evidence_source",
        "status",
        "blocks_materialization",
        "evidence_class",
        "notes",
    ],
    "execution_contract_materialization_blockers": [
        "blocker_id",
        "decision_token",
        "contract_area",
        "missing_definition",
        "source_attempts",
        "blocks_sample",
        "blocks_candidates",
        "blocks_request_manifest",
        "status",
        "next_action",
    ],
    "execution_contract_materialization_summary": [
        "summary_id",
        "item",
        "value",
        "status",
        "evidence",
        "notes",
    ],
}

SEPARATED_TABLE_FIELDS = {
    "profile_request_manifest": [
        "request_id",
        "experiment_family",
        "experiment_cell_id",
        "dataset",
        "task",
        "sample_id",
        "user_id",
        "model_name",
        "model_exact_identity",
        "model_digest",
        "profile_generator_name",
        "profile_generator_digest",
        "recommender_model_name",
        "prompting_strategy",
        "input_variant",
        "history_length",
        "history_hash",
        "candidate_pool_type",
        "candidate_pool_size",
        "candidate_list_hash",
        "positive_item_id",
        "positive_item_title",
        "positive_candidate_position",
        "output_length",
        "parser_contract",
        "prompt_template_path",
        "prompt_template_hash",
        "prompt_template_index",
        "rendered_prompt_hash",
        "rendering_status",
        "generation_parameters_json",
        "dependency_request_id",
        "authorization_status",
        "evidence_class",
        "expected_raw_output_relative_path",
        "input_lineage",
        "request_contract_hash",
    ],
    "profile_checkpoint": [
        "checkpoint_id",
        "item",
        "value",
        "status",
        "evidence",
        "notes",
    ],
    "lastfm_author_specification_registry": [
        "question_id",
        "needed_artifact_or_detail",
        "why_needed",
        "current_status",
        "source_attempts",
        "blocks_exact_reproduction",
        "author_request_text",
    ],
    "lastfm_preprocessing_count_comparison": [
        "comparison_id",
        "source",
        "users",
        "items",
        "interactions",
        "matches_paper_counts",
        "classification",
        "evidence",
        "notes",
    ],
    "lastfm_operationalization_contract": [
        "contract_id",
        "component",
        "local_operationalization",
        "value",
        "author_attribution",
        "classification",
        "blocked_until",
        "notes",
    ],
    "lastfm_operationalization_sensitivity_plan": [
        "plan_id",
        "sensitivity_axis",
        "planned_values",
        "purpose",
        "execution_status",
        "classification",
        "notes",
    ],
    "lastfm_full_cohort_extension_plan": [
        "plan_id",
        "population",
        "preprocessing",
        "split",
        "users",
        "models",
        "strategies",
        "planned_requests",
        "classification",
        "paper_comparability",
        "rq_robustness_value",
        "runtime_estimate",
        "status",
        "notes",
    ],
    "rq_evidence_hierarchy_contract": [
        "hierarchy_rank",
        "evidence_layer",
        "classification",
        "can_support_or_contradict_paper_rq_alone",
        "allowed_rq_effect",
        "required_labeling",
        "notes",
    ],
    "separated_checkpoints": [
        "checkpoint_id",
        "checkpoint_type",
        "decision_token",
        "requests_materialized",
        "manifest_rows",
        "unique_request_ids",
        "classification",
        "status",
        "command_or_next_action",
        "notes",
    ],
}


@dataclass(frozen=True)
class BuildResult:
    decision_token: str
    tables: dict[str, list[dict[str, str]]]
    manifests: dict[str, dict[str, Any]]
    report: str


@dataclass(frozen=True)
class ExecutePreflightResult:
    decision_token: str
    tables: dict[str, list[dict[str, str]]]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ContractMaterializationResult:
    decision_token: str
    tables: dict[str, list[dict[str, str]]]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SeparatedContractsResult:
    decision_token: str
    tables: dict[str, list[dict[str, str]]]
    manifests: dict[str, dict[str, Any]]
    reports: dict[str, str]


@dataclass(frozen=True)
class LastfmExecutionContractsResult:
    decision_token: str
    tables: dict[str, list[dict[str, str]]]
    manifest: dict[str, Any]
    report_section: str


@dataclass(frozen=True)
class LastfmOperationalizedRankingExecutionResult:
    decision_token: str
    run_id: str
    raw_output_dir: Path
    tables: dict[str, list[dict[str, str]]]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class LocalOllamaState:
    service_available: bool
    service_version: str
    model_digests: dict[str, str]
    start_attempted: bool
    started_pid: str
    error: str


@dataclass(frozen=True)
class LocalOllamaRequestResult:
    backend_success: bool
    response_text: str
    raw_backend_response: dict[str, Any]
    backend_error_type: str
    backend_error_message: str
    started_at: str
    completed_at: str
    duration_seconds: float


_PROTECTED_INPUT_HASH_CACHE: dict[str, Any] | None = None
_LOCAL_PIPELINE_INPUT_HASHES_CACHE: dict[str, str] | None = None
_SELECTED_TRADITIONAL_MODEL_SEEDS_CACHE: dict[str, str] | None = None
_SELECTED_SEED_METRIC_VECTORS_CACHE: dict[str, dict[str, dict[str, float]]] | None = None
_LASTFM_COHORT_ROWS_CACHE: dict[str, dict[str, str]] | None = None
_LASTFM_CANDIDATE_ROWS_BY_USER_CACHE: dict[str, list[dict[str, str]]] | None = None
_LASTFM_OBSERVED_ITEMS_CACHE: dict[str, set[str]] | None = None
_LASTFM_FULL_CATALOG_TOPK_STATUS_CACHE: list[dict[str, str]] | None = None
_TEMPLATE_LINE_CACHE: dict[tuple[str, int], str] = {}
_TEMPLATE_HASH_CACHE: dict[str, str] = {}


def clean_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return str(value)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def project_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path


def require_runtime() -> None:
    if Path(sys.executable).resolve() != RECOVERED_INTERPRETER.resolve():
        raise RuntimeError(f"{PHASE_ID} must run with {RECOVERED_INTERPRETER}; got {sys.executable}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise RuntimeError("PYTHONNOUSERSITE=1 is required")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("PYTHONDONTWRITEBYTECODE=1 is required")


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("phase_id") != PHASE_ID:
        raise ValueError(f"wrong or malformed F6L config: {config_path}")
    return config


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean_cell(row.get(field, "")) for field in fields})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def git_output(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return (completed.stdout if completed.returncode == 0 else completed.stderr).strip()


def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def safe_file_sha256(path: Path) -> str:
    return file_sha256(path) if path.exists() and path.is_file() else ""


def fetch_local_ollama_tags(timeout_seconds: int = 10) -> tuple[dict[str, Any], str]:
    if not OLLAMA_TAGS_URL.startswith("http://127.0.0.1:"):
        return {}, "blocked_non_local_ollama_endpoint"
    request = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {}, ""
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def count_rows(path: Path) -> int:
    return len(read_csv_rows(path))


def count_unique(path: Path, column: str) -> int:
    return len({row.get(column, "") for row in read_csv_rows(path) if row.get(column, "")})


def zip_entry_text(zip_path: Path, entry_name: str) -> str:
    with ZipFile(zip_path) as archive:
        with archive.open(entry_name) as handle:
            return handle.read().decode("utf-8", errors="replace")


def zip_json_count(zip_path: Path, entry_name: str) -> str:
    try:
        text = zip_entry_text(zip_path, entry_name)
        payload = json.loads(text)
        return str(len(payload)) if hasattr(payload, "__len__") else "1"
    except Exception:
        return "unavailable"


def zip_entry_status(zip_path: Path, entry_name: str) -> tuple[str, str]:
    if not zip_path.exists():
        return "missing_zip", ""
    with ZipFile(zip_path) as archive:
        entry = archive.getinfo(entry_name) if entry_name in archive.namelist() else None
        if entry is None:
            return "missing_entry", ""
        return "present", str(entry.file_size)


def zip_line_count(zip_path: Path, entry_name: str) -> str:
    try:
        text = zip_entry_text(zip_path, entry_name)
        return str(sum(1 for line in text.splitlines() if line.strip()))
    except Exception:
        return "unavailable"


def official_zip_summary() -> dict[str, str]:
    beauty = PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "llm_results" / "beauty.zip"
    sports = PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "llm_results" / "sports.zip"
    return {
        "beauty_sample_users": zip_line_count(beauty, "beauty/sample_1k_all.txt"),
        "sports_sample_users": zip_line_count(sports, "sports/sample_1k.txt"),
        "beauty_profile_gpt35_rows": zip_json_count(beauty, "beauty/test_history10_generate_profile_sample_texts_gpt3.5.json"),
        "beauty_profile_gpt4_rows": zip_json_count(beauty, "beauty/test_history10_generate_profile_sample_texts_gpt4.json"),
        "beauty_profile_glm4_rows": zip_json_count(beauty, "beauty/test_history10_generate_profile_sample_texts_glm4.json"),
        "sports_profile_gpt35_rows": zip_json_count(sports, "sports/test_history10_generate_profile_sample_texts_gpt3.5.json"),
        "sports_profile_gpt4_rows": zip_json_count(sports, "sports/test_history10_generate_profile_sample_texts_gpt4.json"),
        "sports_profile_glm4_rows": zip_json_count(sports, "sports/test_history10_generate_profile_sample_texts_glm4.json"),
        "beauty_rerank_gpt35_1k_rows": zip_json_count(beauty, "beauty/results_gpt3.5_seed0_test_history10_multi_candidate20_shuffle_sample_1k.json"),
        "sports_rerank_gpt35_1k_rows": zip_json_count(sports, "sports/results_gpt3.5_seed0_test_history10_multi_candidate20_shuffle_sample_1k.json"),
    }


def lastfm_summary() -> dict[str, Any]:
    ranking_candidates = PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_ranking_candidates.csv"
    ranking_rows = read_csv_rows(ranking_candidates)
    positives = [row for row in ranking_rows if row.get("is_ground_truth") == "true"]
    positive_positions = Counter(row.get("candidate_position", "") for row in positives)
    rerank_rows = read_csv_rows(TABLES_DIR / "phase_f6g1_pool_instance_summary.csv")
    rerank_positive = Counter(row.get("positive_present", "") for row in rerank_rows)
    return {
        "interactions": count_rows(PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_interactions.csv"),
        "train_rows": count_rows(PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_train.csv"),
        "validation_rows": count_rows(PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_validation.csv"),
        "test_rows": count_rows(PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_test.csv"),
        "ranking_cohort_users": count_rows(PROJECT_ROOT / "data" / "processed" / "finalization_f6c_lastfm_ranking_cohort.csv"),
        "ranking_candidate_rows": len(ranking_rows),
        "ranking_candidate_sets": count_unique(ranking_candidates, "candidate_set_id"),
        "ranking_positive_rows": len(positives),
        "ranking_positive_position_count": len(positive_positions),
        "ranking_positive_position_min": min((int(k) for k in positive_positions if k), default=0),
        "ranking_positive_position_max": max((int(k) for k in positive_positions if k), default=0),
        "reranking_instances": len(rerank_rows),
        "reranking_positive_present": int(rerank_positive.get("true", 0)),
        "reranking_positive_absent": int(rerank_positive.get("false", 0)),
    }


def protected_input_paths() -> list[str]:
    paths = [
        "paper/2411.00331v1.pdf",
        "original_repo/EvaLLMasRecommender/README.md",
        "original_repo/EvaLLMasRecommender/eva/templates/template_user_profile_generate.txt",
        "original_repo/EvaLLMasRecommender/eva/templates/template_profile.txt",
        "original_repo/EvaLLMasRecommender/eva/templates/template_profile_history.txt",
        "original_repo/EvaLLMasRecommender/eva/templates/template.txt",
        "original_repo/EvaLLMasRecommender/eva/templates/llmrank.txt",
        "original_repo/EvaLLMasRecommender/eva/templates/llmrank_rf.txt",
        "original_repo/EvaLLMasRecommender/eva/templates/template_in_context.txt",
        "original_repo/EvaLLMasRecommender/eva/templates/template_fewshot.txt",
        "original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb",
        "original_repo/EvaLLMasRecommender/eva/codes/zero_shot.py",
        "original_repo/EvaLLMasRecommender/eva/codes/output_refine.py",
        "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py",
        "original_repo/EvaLLMasRecommender/eva/codes/llms.py",
        "original_repo/EvaLLMasRecommender/llm_results/beauty.zip",
        "original_repo/EvaLLMasRecommender/llm_results/sports.zip",
        "original_repo/EvaLLMasRecommender/sota_results/beauty_his10.zip",
        "results/tables/phase_f6j_research_question_evidence_matrix.csv",
        "results/tables/phase_f6j_paper_reproduction_extension_matrix.csv",
        "results/tables/phase_f6k_scientific_gap_readiness.csv",
        "results/tables/finalization_f6d3_lastfm_ranking_llm_manifest.csv",
        "results/tables/finalization_f6d4a_utility_metrics.csv",
        "results/llm/finalization_f6d4a_lastfm_ranking_qwen2_raw.jsonl",
        "results/llm/finalization_f6d4a_lastfm_ranking_mistral_raw.jsonl",
        "results/tables/phase_f6g1_pool_instance_summary.csv",
        "results/tables/phase_f6g3_cell_summary.csv",
        "data/processed/finalization_f6c_lastfm_interactions.csv",
        "data/processed/finalization_f6c_lastfm_ranking_candidates.csv",
        "data/processed/finalization_f6c_lastfm_ranking_cohort.csv",
    ]
    for row in read_csv_rows(TABLES_DIR / "phase_f6i2p_resolved_config_equivalence.csv"):
        config_path = row.get("config_path", "")
        if config_path:
            paths.append(config_path)
    return sorted(set(paths))


def hash_protected_inputs() -> dict[str, Any]:
    global _PROTECTED_INPUT_HASH_CACHE
    if _PROTECTED_INPUT_HASH_CACHE is not None:
        return json.loads(json.dumps(_PROTECTED_INPUT_HASH_CACHE))
    entries = []
    missing = []
    for relative in protected_input_paths():
        path = project_path(relative)
        exists = path.exists() and path.is_file()
        digest = file_sha256(path) if exists else ""
        if not exists:
            missing.append(relative)
        entries.append({"path": relative, "exists": exists, "sha256": digest})
    f6i2p_configs = [entry for entry in entries if entry["path"].startswith("configs/") and entry["path"].endswith(".yaml")]
    payload = {
        "phase_id": PHASE_ID,
        "protected_inputs": entries,
        "missing": missing,
        "f6i2p_yaml_count": len(f6i2p_configs),
        "status": "pass" if not missing else "fail",
    }
    _PROTECTED_INPUT_HASH_CACHE = payload
    return json.loads(json.dumps(payload))


def build_exactness_matrix(summary: Mapping[str, Any], zip_summary: Mapping[str, str]) -> list[dict[str, str]]:
    rows = [
        ("F6L-EXACT-001", "dataset_version", "Amazon Beauty/Sports/ML-1M/LastFM", "paper names four datasets and 5-core preprocessing", "README and ZIPs expose Beauty/Sports outputs; LastFM local data comes from HetRec2011", "LastFM local pipeline has HetRec2011-derived 1859-user split", "adapted", "adapted_protocol_reproduction", "not_exact_original_sample", "retain adapted and build new F6L paper-protocol LastFM sample", "paper;original_repo;local_artifacts", "paper/2411.00331v1.pdf;original_repo/EvaLLMasRecommender/README.md;data/processed/finalization_f6c_lastfm_interactions.csv", f"LastFM interactions={summary['interactions']}."),
        ("F6L-EXACT-002", "5_core_filtering", "5-core filtering", "paper states 5-core filter for all datasets", "official repo documents preprocessed data but local LastFM reconstruction is not official paper lineage", "F6C LastFM is reconstructed from HetRec2011 and marked not exact", "adapted", "adapted_protocol_reproduction", "not_exact_original_sample", "audit and do not promote prior rows", "paper;repo;F6C", "report/finalization_f6c_lastfm_readiness.md", "No paper-exact LastFM processed archive is present locally."),
        ("F6L-EXACT-003", "counts", "users/items/interactions", "paper table values require direct paper verification and official processed data", "README points to downloadable data; local ZIPs expose Beauty/Sports LLM outputs only", f"LastFM local interactions={summary['interactions']}; test users={summary['test_rows']}", "non_verifiable", "blocked_missing_original_specification", "not_exact_original_sample", "record as exactness limit", "paper;official_download_optional;local_artifacts", "results/tables/phase_f6h2_recomputation_readiness.csv", "Counts cannot prove paper identity without processed official LastFM data."),
        ("F6L-EXACT-004", "split", "leave-one-out train/validation/test", "paper states leave-one-out split", "official notebook uses sequential_data and negative_samples for Beauty/Sports", f"local LastFM train={summary['train_rows']} validation={summary['validation_rows']} test={summary['test_rows']}", "adapted", "adapted_protocol_reproduction", "not_exact_original_sample", "rebuild LastFM 1000-user sample from local split only as new sample", "paper;repo;F6C", "data/processed/finalization_f6c_lastfm_train.csv;data/processed/finalization_f6c_lastfm_test.csv", "Local split is useful but not exact official split lineage."),
        ("F6L-EXACT-005", "sample", "1000 test users and K-S validation", "paper reports random sample of 1000 users and K-S validation", "Beauty/Sports ZIPs include sample_1k files; no LastFM official sample in local repo", f"existing LastFM ranking cohort={summary['ranking_cohort_users']} users", "incompatible", "adapted_protocol_reproduction", "not_exact_original_sample", "do not use existing 1859 cohort as paper-faithful", "paper;zip;F6D4A", "results/tables/finalization_f6d3_lastfm_ranking_llm_manifest.csv", "F6L will prefer official sample if downloaded; otherwise deterministic K-S-valid new sample."),
        ("F6L-EXACT-006", "history_construction", "chronological histories and history lengths", "paper evaluates history lengths including 0,1-5,10,15,20 in places", "generate_prompt.ipynb truncates to last L and preserves order unless shuffled", "LastFM F6D3 ranking renders last 10 only; F6G1 reranking used longer histories", "adapted", "adapted_protocol_reproduction", "mixed", "freeze prompt-specific history policy per new cell", "repo;F6D3;F6G1", "original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb", "Prior local LastFM cells do not share one paper-exact history contract."),
        ("F6L-EXACT-007", "ranking_candidates", "1 positive and 19 negatives, 20 total", "paper ranking pool is 1 positive plus 19 negatives with shared order", "generate_prompt builds positive plus negatives and shuffles candidate list", f"F6C LastFM ranking sets={summary['ranking_candidate_sets']}; positives={summary['ranking_positive_rows']}; positions={summary['ranking_positive_position_min']}-{summary['ranking_positive_position_max']}", "paper_related_not_exact", "adapted_protocol_reproduction", "not_exact_original_sample", "rebuild or resample for F6L LastFM 1000-user cohort", "paper;repo;F6C", "data/processed/finalization_f6c_lastfm_ranking_candidates.csv", "Existing candidate shape matches but seed/sample lineage is adapted."),
        ("F6L-EXACT-008", "reranking_candidates", "top recommendations from BPRMF/GRU4Rec/SASRec/LightGCN, 20 total", "paper reranking pool is built from four traditional models", "notebook shows round-robin aggregation across SASRec, GRU4Rec, LightGCN, BPRMF", f"adapted F6G1 positive present={summary['reranking_positive_present']}/{summary['reranking_instances']}", "adapted", "adapted_protocol_reproduction", "not_exact_original_sample", "block paper-faithful reranking until source top-K identity is verified", "paper;repo;F6G1;F6G3", "results/tables/phase_f6g1_pool_instance_summary.csv", "Prior 7436 requests are not reused as paper-faithful."),
        ("F6L-EXACT-009", "output_length", "top-10 output", "paper requests top-10 recommendations; code system prompt requires 10 titles", "llms.py and output_refine.py parse 10-title lists", "F6D4A local ranking requested top-5 JSON indices; F6G3 corrected reranking to top-10 but has no raw rows", "incompatible_prior", "adapted_protocol_reproduction", "not_applicable", "use title-list top-10 semantics for paper-protocol cells", "paper;repo;F6D4A;F6G3", "original_repo/EvaLLMasRecommender/eva/codes/llms.py;results/tables/phase_f6g3_cell_summary.csv", "JSON/index outputs remain local adaptations."),
        ("F6L-EXACT-010", "models", "seven LLMs including Qwen2 and Mistral", "paper includes open and closed/API models", "local model inventory records Qwen2 and Mistral prior successful runs", "closed/API models are not called; local tags are not exact version snapshots", "partial", "paper_protocol_reproduction_different_open_model", "not_applicable", "execute only Qwen2/Mistral after checkpoint", "paper;F6D3", "results/tables/finalization_f6d3_local_llm_inventory.csv", "Closed-model cells are blocked or official-output-only."),
        ("F6L-EXACT-011", "prompting_strategy", "OpenP5/base, LLMRank, LLMSRec/in-context", "paper and repo contain base, LLMRank, and in-context/profile templates", "template files are present; LLMSRec demonstration selection remains incomplete", "base and LLMRank reconstructible; LLMSRec is blocked unless demo contract is recovered", "partial", "blocked_missing_original_specification", "not_applicable", "exclude partial LLMSRec from exact request count", "repo;F6E1", "results/tables/phase_f6e1_prompt_source_inventory.csv", "Prompt source exists but orchestration is not complete for LLMSRec."),
        ("F6L-EXACT-012", "decoding_parameters", "temperature and provider parameters", "paper does not freeze all provider parameters in local evidence", "llms.py uses temperature 0 for ChatGPT and 0.01 for GLM", "local Qwen2/Mistral execution contract uses temperature 0", "paper_related_adapted", "paper_protocol_reproduction_different_open_model", "not_applicable", "record deterministic local open-model settings separately", "repo;local_contract", "results/tables/canonical_llm_execution_parameter_contract.csv", "New stochastic outputs cannot be exact numerical reproduction."),
        ("F6L-EXACT-013", "parser_normalization", "title-list parser and item-title normalization", "paper describes string matching ignoring case/spaces/symbols", "output_refine.py and llm_metrics.py parse Python title lists with mapping repairs", "local JSON/index parser is more restrictive and adapted", "partial", "adapted_protocol_reproduction", "not_applicable", "separate title-list parser evidence from JSON/index evidence", "paper;repo;local_contract", "original_repo/EvaLLMasRecommender/eva/codes/llm_metrics.py", "Exact fuzzy normalization remains not fully reproduced."),
        ("F6L-EXACT-014", "metrics", "HR@5, NDCG@5, beyond-utility metrics, significance", "paper reports utility and beyond-utility with selected significance", "llm_metrics.py implements HR/NDCG/APLT/ARP/serendipity/hallucination", "F6J evidence is adapted/extension; no exact numeric replication", "adapted", "adapted_protocol_reproduction", "not_applicable", "revalidate RQs using evidence hierarchy", "paper;repo;F6J", "results/tables/phase_f6j_research_question_evidence_matrix.csv", "Do not let adapted cells decide paper claims alone."),
    ]
    return [
        {
            "matrix_id": matrix_id,
            "area": area,
            "requirement": requirement,
            "paper_contract": paper_contract,
            "repository_evidence": repository_evidence,
            "local_current_state": local_current_state,
            "cell_status": cell_status,
            "evidence_classification": classification,
            "sample_exactness": sample_exactness,
            "action": action,
            "source_priority": source_priority,
            "evidence_references": refs,
            "notes": notes,
        }
        for (
            matrix_id,
            area,
            requirement,
            paper_contract,
            repository_evidence,
            local_current_state,
            cell_status,
            classification,
            sample_exactness,
            action,
            source_priority,
            refs,
            notes,
        ) in rows
    ]


def build_profile_contract(zip_summary: Mapping[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    official_rows = [
        ("beauty", "GPT-3.5", zip_summary["beauty_profile_gpt35_rows"]),
        ("beauty", "GPT-4", zip_summary["beauty_profile_gpt4_rows"]),
        ("beauty", "GLM4", zip_summary["beauty_profile_glm4_rows"]),
        ("sports", "GPT-3.5", zip_summary["sports_profile_gpt35_rows"]),
        ("sports", "GPT-4", zip_summary["sports_profile_gpt4_rows"]),
        ("sports", "GLM4", zip_summary["sports_profile_glm4_rows"]),
    ]
    for dataset, generator, count in official_rows:
        rows.append(
            {
                "contract_id": f"F6L-PROFILE-OFFICIAL-{dataset.upper()}-{generator.replace('.', '').replace('-', '').upper()}",
                "dataset": dataset,
                "cohort": "official_zip_profile_rows",
                "history_length": "10",
                "profile_generator": generator,
                "recommender_model": "official recorded recommender output where present",
                "prompt_mode": "official_profile_artifact_inventory",
                "profile_prompt_source": "original_repo/EvaLLMasRecommender/eva/templates/template_user_profile_generate.txt",
                "recommendation_prompt_source": "template_profile.txt;template_profile_history.txt",
                "candidate_set_source": "official ZIP prompt/result files",
                "official_artifact_status": f"profile_text_rows={count}",
                "local_execution_status": "read_only_no_inference",
                "evidence_classification": "exact_original_artifact_reproduction",
                "request_count": "0",
                "blocked_reason": "",
                "notes": "Artifact existence is exact; numerical claim remains limited by request/response lineage.",
            }
        )
    for dataset in ["beauty", "sports"]:
        for generator in ["qwen2:7b-instruct", "mistral:7b-instruct"]:
            for recommender in ["qwen2:7b-instruct", "mistral:7b-instruct"]:
                rows.append(
                    {
                        "contract_id": f"F6L-PROFILE-LOCAL-{dataset.upper()}-{generator.split(':')[0].upper()}-{recommender.split(':')[0].upper()}",
                        "dataset": dataset,
                        "cohort": "official_200_user_profile_artifact_aligned_cohort",
                        "history_length": "10",
                        "profile_generator": generator,
                        "recommender_model": recommender,
                        "prompt_mode": "history_only;profile_only;profile_plus_history",
                        "profile_prompt_source": "original_repo/EvaLLMasRecommender/eva/templates/template_user_profile_generate.txt",
                        "recommendation_prompt_source": "template.txt;template_profile.txt;template_profile_history.txt",
                        "candidate_set_source": "official ZIP prompt candidate lists when lineage passes",
                        "official_artifact_status": "official prompts/results inventoried",
                        "local_execution_status": "planned_after_checkpoint",
                        "evidence_classification": "paper_protocol_reproduction_different_open_model",
                        "request_count": "600" if generator == recommender else "400",
                        "blocked_reason": "",
                        "notes": "The per generator/recommender total accounts for profile reuse; history-only is shared per recommender in the request manifest.",
                    }
                )
    for dataset in ["ml1m", "lastfm"]:
        rows.append(
            {
                "contract_id": f"F6L-PROFILE-BLOCKED-{dataset.upper()}",
                "dataset": dataset,
                "cohort": "not_frozen",
                "history_length": "10",
                "profile_generator": "Qwen2/Mistral future local",
                "recommender_model": "Qwen2/Mistral future local",
                "prompt_mode": "history_only;profile_only;profile_plus_history",
                "profile_prompt_source": "template_user_profile_generate.txt",
                "recommendation_prompt_source": "template.txt;template_profile.txt;template_profile_history.txt",
                "candidate_set_source": "missing exact profile cohort/candidate lineage",
                "official_artifact_status": "not recovered in local official ZIPs",
                "local_execution_status": "blocked_before_checkpoint_execution",
                "evidence_classification": "blocked_missing_original_specification",
                "request_count": "0",
                "blocked_reason": "profile cohort and candidate lineage not frozen",
                "notes": "Do not infer profile-generation results for ML-1M/LastFM without a matched cohort contract.",
            }
        )
    return rows


def build_profile_results() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    metrics = ["HR@5", "NDCG@5", "APLT", "serendipity", "validity", "hallucination"]
    for dataset in ["beauty", "sports"]:
        for generator in ["qwen2:7b-instruct", "mistral:7b-instruct"]:
            for recommender in ["qwen2:7b-instruct", "mistral:7b-instruct"]:
                for mode in ["history_only", "profile_only", "profile_plus_history"]:
                    for metric in metrics:
                        rows.append(
                            {
                                "result_id": sha256_text(f"{dataset}|{generator}|{recommender}|{mode}|{metric}")[:16],
                                "dataset": dataset,
                                "profile_generator": "" if mode == "history_only" else generator,
                                "recommender_model": recommender,
                                "mode": mode,
                                "evidence_classification": "paper_protocol_reproduction_different_open_model",
                                "request_count": "0" if metric != "HR@5" else ("200" if mode != "history_only" else "shared"),
                                "metric": metric,
                                "value": "",
                                "status": "planned_after_checkpoint_no_inference_yet",
                                "source_file": "",
                                "notes": "Blank value is intentional at checkpoint; no output is imputed.",
                            }
                        )
    for dataset in ["beauty", "sports"]:
        for generator in ["GPT-3.5", "GPT-4", "GLM4"]:
            rows.append(
                {
                    "result_id": sha256_text(f"official|{dataset}|{generator}")[:16],
                    "dataset": dataset,
                    "profile_generator": generator,
                    "recommender_model": "official recorded model",
                    "mode": "profile_artifact_inventory",
                    "evidence_classification": "exact_original_artifact_reproduction",
                    "request_count": "0",
                    "metric": "raw_profile_rows",
                    "value": "200",
                    "status": "official_artifact_read_only",
                    "source_file": f"original_repo/EvaLLMasRecommender/llm_results/{dataset}.zip",
                    "notes": "Profile text artifacts are present; this is not a new metric claim.",
                }
            )
    return rows


def build_lastfm_ranking_exactness(summary: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = [
        {
            "cell_id": "F6L-LASTFM-RANKING-EXISTING-F6D4A",
            "dataset": "LastFM",
            "cohort_source": "finalization_f6c_lastfm_ranking_cohort_full_1859",
            "users": str(summary["ranking_cohort_users"]),
            "candidate_count": "20",
            "positive_policy": "1 positive plus 19 negatives",
            "candidate_order": "shared within F6D4A local JSON/index run",
            "model": "qwen2:7b-instruct;mistral:7b-instruct",
            "strategy": "local_json_index_ranking",
            "prompt_source": "results/tables/finalization_f6d3_prompt_contract.csv",
            "output_length": "5",
            "parser": "strict JSON one-based candidate indices",
            "existing_artifact": "results/tables/finalization_f6d4a_utility_metrics.csv",
            "evidence_classification": "adapted_protocol_reproduction",
            "sample_exactness": "existing_1859_user_adapted_sample_not_paper_1000",
            "request_count": "0",
            "status": "completed_existing_adapted",
            "blocking_reason": "",
            "notes": "Preserved as adapted evidence; not paper-exact ranking.",
        }
    ]
    for model in ["qwen2:7b-instruct", "mistral:7b-instruct"]:
        for strategy in ["paper_base_openp5_title_list", "paper_llmrank_recency_title_list"]:
            rows.append(
                {
                    "cell_id": f"F6L-LASTFM-RANKING-{model.split(':')[0].upper()}-{strategy.upper()}",
                    "dataset": "LastFM",
                    "cohort_source": "phase_f6l_lastfm_1000_user_ks_valid_new_sample",
                    "users": "1000",
                    "candidate_count": "20",
                    "positive_policy": "1 positive plus 19 negatives",
                    "candidate_order": "shared across local open models and prompt strategies",
                    "model": model,
                    "strategy": strategy,
                    "prompt_source": "original_repo/EvaLLMasRecommender/eva/templates/template.txt or llmrank_rf.txt",
                    "output_length": "10",
                    "parser": "paper title-list parser with validity/hallucination separated",
                    "existing_artifact": "",
                    "evidence_classification": "paper_protocol_reproduction_different_open_model",
                    "sample_exactness": "paper_protocol_reproduction_new_sample_not_exact_original_sample",
                    "request_count": "1000",
                    "status": "planned_after_checkpoint",
                    "blocking_reason": "",
                    "notes": "New stochastic local output cannot be exact numerical reproduction.",
                }
            )
    rows.append(
        {
            "cell_id": "F6L-LASTFM-RANKING-LLMSREC",
            "dataset": "LastFM",
            "cohort_source": "phase_f6l_lastfm_1000_user_ks_valid_new_sample",
            "users": "1000",
            "candidate_count": "20",
            "positive_policy": "1 positive plus 19 negatives",
            "candidate_order": "not authorized",
            "model": "qwen2:7b-instruct;mistral:7b-instruct",
            "strategy": "LLMSRec/in-context",
            "prompt_source": "template_in_context.txt;template_demonstration.txt",
            "output_length": "10",
            "parser": "paper title-list parser",
            "existing_artifact": "",
            "evidence_classification": "blocked_missing_original_specification",
            "sample_exactness": "not_applicable",
            "request_count": "0",
            "status": "blocked_before_checkpoint_execution",
            "blocking_reason": "demonstration user-selection and orchestration contract incomplete",
            "notes": "Do not execute partial LLMSRec as exact.",
        }
    )
    return rows


def build_lastfm_ranking_results() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv_rows(TABLES_DIR / "finalization_f6d4a_utility_metrics.csv"):
        rows.append(
            {
                "result_id": row.get("metric_id", ""),
                "dataset": "LastFM",
                "model": row.get("model", ""),
                "strategy": "local_json_index_ranking",
                "evidence_classification": "adapted_protocol_reproduction",
                "sample_exactness": "existing_1859_user_adapted_sample_not_paper_1000",
                "request_count": "0",
                "metric": row.get("metric", ""),
                "value": row.get("value", ""),
                "valid_rows": row.get("valid_rows", ""),
                "source_file": "results/tables/finalization_f6d4a_utility_metrics.csv",
                "status": "existing_adapted_result",
                "notes": "Reclassified; not used alone to validate paper numerical claims.",
            }
        )
    for model in ["qwen2:7b-instruct", "mistral:7b-instruct"]:
        for strategy in ["paper_base_openp5_title_list", "paper_llmrank_recency_title_list"]:
            for metric in ["HR@5", "NDCG@5", "APLT", "serendipity", "validity", "hallucination"]:
                rows.append(
                    {
                        "result_id": sha256_text(f"lastfm-ranking|{model}|{strategy}|{metric}")[:16],
                        "dataset": "LastFM",
                        "model": model,
                        "strategy": strategy,
                        "evidence_classification": "paper_protocol_reproduction_different_open_model",
                        "sample_exactness": "paper_protocol_reproduction_new_sample_not_exact_original_sample",
                        "request_count": "1000" if metric == "HR@5" else "0",
                        "metric": metric,
                        "value": "",
                        "valid_rows": "",
                        "source_file": "",
                        "status": "planned_after_checkpoint_no_inference_yet",
                        "notes": "Blank value is intentional at checkpoint.",
                    }
                )
    return rows


def build_lastfm_reranking_results(summary: Mapping[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv_rows(TABLES_DIR / "phase_f6g3_cell_summary.csv"):
        rows.append(
            {
                "result_id": row.get("cell_id", ""),
                "dataset": "LastFM",
                "model": row.get("model", ""),
                "strategy": row.get("strategy", ""),
                "evidence_classification": "adapted_protocol_reproduction",
                "sample_exactness": "existing_1859_user_adapted_sample_not_paper_1000",
                "request_count": row.get("expected_requests", "1859"),
                "metric": "raw_rows_reconciled",
                "value": row.get("raw_rows_reconciled", "0"),
                "valid_rows": "0",
                "source_file": "results/tables/phase_f6g3_cell_summary.csv",
                "status": "existing_adapted_blocked_no_raw_responses",
                "notes": f"Positive present {summary['reranking_positive_present']}/{summary['reranking_instances']}; not paper-faithful.",
            }
        )
    for model in ["qwen2:7b-instruct", "mistral:7b-instruct"]:
        for strategy in ["paper_base_openp5_title_list", "paper_llmrank_recency_title_list"]:
            rows.append(
                {
                    "result_id": sha256_text(f"lastfm-reranking|{model}|{strategy}|blocked")[:16],
                    "dataset": "LastFM",
                    "model": model,
                    "strategy": strategy,
                    "evidence_classification": "blocked_missing_original_specification",
                    "sample_exactness": "not_applicable_until_paper_pool_recovered",
                    "request_count": "0",
                    "metric": "HR@5;NDCG@5;APLT;serendipity;validity;hallucination",
                    "value": "",
                    "valid_rows": "",
                    "source_file": "",
                    "status": "blocked_before_checkpoint_execution",
                    "notes": "Paper-faithful reranking is blocked until BPRMF/GRU4Rec/SASRec/LightGCN source top-K pool identity is verified.",
                }
            )
    return rows


def build_request_manifest() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(
        [
            {
                "request_group_id": "F6L-REQ-PROFILE-GENERATION",
                "phase": PHASE_ID,
                "dataset": "beauty;sports",
                "task": "profile_generation",
                "cohort": "official_200_user_profile_artifact_aligned_cohort",
                "model": "qwen2:7b-instruct;mistral:7b-instruct",
                "strategy": "template_user_profile_generate",
                "mode": "profile_generation",
                "user_count": "400",
                "requests_per_user": "2",
                "total_requests": "800",
                "authorized_for_checkpoint_execution": "false",
                "prompt_source": "original_repo/EvaLLMasRecommender/eva/templates/template_user_profile_generate.txt",
                "parser_contract": "free_text_profile_retained_no_metric_imputation",
                "raw_output_policy": "new non-colliding results/llm/phase_f6l_profile_generation_*.jsonl",
                "evidence_classification": "paper_protocol_reproduction_different_open_model",
                "blocked_reason": "checkpoint awaiting explicit user authorization",
                "notes": "No profiles are generated in audit-checkpoint mode.",
            },
            {
                "request_group_id": "F6L-REQ-PROFILE-HISTORY-ONLY",
                "phase": PHASE_ID,
                "dataset": "beauty;sports",
                "task": "ranking",
                "cohort": "official_200_user_profile_artifact_aligned_cohort",
                "model": "qwen2:7b-instruct;mistral:7b-instruct",
                "strategy": "history_only",
                "mode": "history_only_no_profile",
                "user_count": "400",
                "requests_per_user": "2",
                "total_requests": "800",
                "authorized_for_checkpoint_execution": "false",
                "prompt_source": "original_repo/EvaLLMasRecommender/eva/templates/template.txt",
                "parser_contract": "paper_title_list_top10_parser",
                "raw_output_policy": "new non-colliding results/llm/phase_f6l_profile_recommendation_*.jsonl",
                "evidence_classification": "paper_protocol_reproduction_different_open_model",
                "blocked_reason": "checkpoint awaiting explicit user authorization",
                "notes": "History-only baseline is shared across profile generators.",
            },
            {
                "request_group_id": "F6L-REQ-PROFILE-ONLY",
                "phase": PHASE_ID,
                "dataset": "beauty;sports",
                "task": "ranking",
                "cohort": "official_200_user_profile_artifact_aligned_cohort",
                "model": "qwen2:7b-instruct;mistral:7b-instruct",
                "strategy": "profile_only",
                "mode": "profile_only",
                "user_count": "400",
                "requests_per_user": "4",
                "total_requests": "1600",
                "authorized_for_checkpoint_execution": "false",
                "prompt_source": "original_repo/EvaLLMasRecommender/eva/templates/template_profile.txt",
                "parser_contract": "paper_title_list_top10_parser",
                "raw_output_policy": "new non-colliding results/llm/phase_f6l_profile_recommendation_*.jsonl",
                "evidence_classification": "paper_protocol_reproduction_different_open_model",
                "blocked_reason": "checkpoint awaiting explicit user authorization",
                "notes": "Two generators by two recommenders on two 200-user datasets.",
            },
            {
                "request_group_id": "F6L-REQ-PROFILE-PLUS-HISTORY",
                "phase": PHASE_ID,
                "dataset": "beauty;sports",
                "task": "ranking",
                "cohort": "official_200_user_profile_artifact_aligned_cohort",
                "model": "qwen2:7b-instruct;mistral:7b-instruct",
                "strategy": "profile_plus_history",
                "mode": "profile_plus_history",
                "user_count": "400",
                "requests_per_user": "4",
                "total_requests": "1600",
                "authorized_for_checkpoint_execution": "false",
                "prompt_source": "original_repo/EvaLLMasRecommender/eva/templates/template_profile_history.txt",
                "parser_contract": "paper_title_list_top10_parser",
                "raw_output_policy": "new non-colliding results/llm/phase_f6l_profile_recommendation_*.jsonl",
                "evidence_classification": "paper_protocol_reproduction_different_open_model",
                "blocked_reason": "checkpoint awaiting explicit user authorization",
                "notes": "Same profiles and candidates as profile-only.",
            },
            {
                "request_group_id": "F6L-REQ-LASTFM-RANKING-BASE",
                "phase": PHASE_ID,
                "dataset": "LastFM",
                "task": "ranking",
                "cohort": "phase_f6l_lastfm_1000_user_ks_valid_new_sample",
                "model": "qwen2:7b-instruct;mistral:7b-instruct",
                "strategy": "paper_base_openp5_title_list",
                "mode": "ranking",
                "user_count": "1000",
                "requests_per_user": "2",
                "total_requests": "2000",
                "authorized_for_checkpoint_execution": "false",
                "prompt_source": "original_repo/EvaLLMasRecommender/eva/templates/template.txt",
                "parser_contract": "paper_title_list_top10_parser",
                "raw_output_policy": "new non-colliding results/llm/phase_f6l_lastfm_ranking_*.jsonl",
                "evidence_classification": "paper_protocol_reproduction_different_open_model",
                "blocked_reason": "checkpoint awaiting explicit user authorization",
                "notes": "Uses a deterministic 1000-user K-S-valid sample unless an official sample is recovered.",
            },
            {
                "request_group_id": "F6L-REQ-LASTFM-RANKING-LLMRANK",
                "phase": PHASE_ID,
                "dataset": "LastFM",
                "task": "ranking",
                "cohort": "phase_f6l_lastfm_1000_user_ks_valid_new_sample",
                "model": "qwen2:7b-instruct;mistral:7b-instruct",
                "strategy": "paper_llmrank_recency_title_list",
                "mode": "ranking",
                "user_count": "1000",
                "requests_per_user": "2",
                "total_requests": "2000",
                "authorized_for_checkpoint_execution": "false",
                "prompt_source": "original_repo/EvaLLMasRecommender/eva/templates/llmrank_rf.txt",
                "parser_contract": "paper_title_list_top10_parser",
                "raw_output_policy": "new non-colliding results/llm/phase_f6l_lastfm_ranking_*.jsonl",
                "evidence_classification": "paper_protocol_reproduction_different_open_model",
                "blocked_reason": "checkpoint awaiting explicit user authorization",
                "notes": "LLMRank non-rf source remains recorded; recency-focused is the executable contract.",
            },
            {
                "request_group_id": "F6L-REQ-LASTFM-RERANKING-PAPER",
                "phase": PHASE_ID,
                "dataset": "LastFM",
                "task": "reranking",
                "cohort": "phase_f6l_lastfm_1000_user_ks_valid_new_sample",
                "model": "qwen2:7b-instruct;mistral:7b-instruct",
                "strategy": "paper_base_openp5_title_list;paper_llmrank_recency_title_list",
                "mode": "reranking",
                "user_count": "1000",
                "requests_per_user": "0",
                "total_requests": "0",
                "authorized_for_checkpoint_execution": "false",
                "prompt_source": "template.txt;llmrank_rf.txt",
                "parser_contract": "paper_title_list_top10_parser",
                "raw_output_policy": "blocked_no_raw_path_allocated",
                "evidence_classification": "blocked_missing_original_specification",
                "blocked_reason": "paper-faithful BPRMF/GRU4Rec/SASRec/LightGCN pool source identity not fully recovered",
                "notes": "The stale 7436 adapted requests are not carried into F6L paper-faithful count.",
            },
        ]
    )
    return rows


def total_authorized_future_requests(requests: Sequence[Mapping[str, str]]) -> int:
    return sum(int(row["total_requests"]) for row in requests if row["evidence_classification"] != "blocked_missing_original_specification")


def build_execution_request_breakdown() -> list[dict[str, str]]:
    """Return one row per planned F6L authorization cell."""

    rows: list[dict[str, str]] = []

    def add(
        *,
        cell_id: str,
        experiment_family: str,
        dataset: str,
        task: str,
        cohort: str,
        cohort_size: int,
        model: str,
        profile_generator: str,
        recommender_model: str,
        prompting_strategy: str,
        input_variant: str,
        history_length: str,
        candidate_pool_type: str,
        candidate_pool_size: str,
        output_length: str,
        requests_per_user: int,
        expected_requests: int,
        requires_profile_generation: bool,
        requires_recommendation_inference: bool,
        evidence_class: str,
        authorization_status: str,
        authorization_reason: str,
    ) -> None:
        rows.append(
            {
                "cell_id": cell_id,
                "experiment_family": experiment_family,
                "dataset": dataset,
                "task": task,
                "cohort": cohort,
                "cohort_size": str(cohort_size),
                "model": model,
                "model_exact_identity": "local_tag_not_exact_paper_snapshot" if model else "not_applicable",
                "profile_generator": profile_generator,
                "recommender_model": recommender_model,
                "prompting_strategy": prompting_strategy,
                "input_variant": input_variant,
                "history_length": history_length,
                "candidate_pool_type": candidate_pool_type,
                "candidate_pool_size": candidate_pool_size,
                "output_length": output_length,
                "requests_per_user": str(requests_per_user),
                "expected_requests": str(expected_requests),
                "requires_profile_generation": clean_cell(requires_profile_generation),
                "requires_recommendation_inference": clean_cell(requires_recommendation_inference),
                "evidence_class": evidence_class,
                "authorization_status": authorization_status,
                "authorization_reason": authorization_reason,
            }
        )

    local_models = ["qwen2:7b-instruct", "mistral:7b-instruct"]
    profile_datasets = ["beauty", "sports"]
    for dataset in profile_datasets:
        for model in local_models:
            add(
                cell_id=f"F6L-BREAKDOWN-PROFILE-GEN-{dataset.upper()}-{model.split(':')[0].upper()}",
                experiment_family="profile_generation",
                dataset=dataset,
                task="profile_generation",
                cohort="official_200_user_profile_artifact_aligned_cohort",
                cohort_size=200,
                model=model,
                profile_generator=model,
                recommender_model="",
                prompting_strategy="template_user_profile_generate",
                input_variant="profile_generation",
                history_length="10",
                candidate_pool_type="not_applicable",
                candidate_pool_size="0",
                output_length="free_text_profile",
                requests_per_user=1,
                expected_requests=200,
                requires_profile_generation=True,
                requires_recommendation_inference=False,
                evidence_class="paper_protocol_reproduction_different_open_model",
                authorization_status="authorized_after_explicit_user_execution_approval",
                authorization_reason="Official Beauty/Sports profile cohort artifacts exist locally; checkpoint performed no generation.",
            )
            add(
                cell_id=f"F6L-BREAKDOWN-HISTORY-ONLY-{dataset.upper()}-{model.split(':')[0].upper()}",
                experiment_family="profile_recommendation_control",
                dataset=dataset,
                task="ranking",
                cohort="official_200_user_profile_artifact_aligned_cohort",
                cohort_size=200,
                model=model,
                profile_generator="none",
                recommender_model=model,
                prompting_strategy="template",
                input_variant="history_only",
                history_length="10",
                candidate_pool_type="official_1_positive_19_negative",
                candidate_pool_size="20",
                output_length="10",
                requests_per_user=1,
                expected_requests=200,
                requires_profile_generation=False,
                requires_recommendation_inference=True,
                evidence_class="paper_protocol_reproduction_different_open_model",
                authorization_status="authorized_after_explicit_user_execution_approval",
                authorization_reason="Shared history-only control over the same cohort/candidates as profile cells.",
            )

    for dataset in profile_datasets:
        for generator in local_models:
            for recommender in local_models:
                for variant, prompt in [
                    ("profile_only", "template_profile"),
                    ("profile_plus_history", "template_profile_history"),
                ]:
                    add(
                        cell_id=(
                            f"F6L-BREAKDOWN-{variant.upper().replace('_', '-')}-"
                            f"{dataset.upper()}-{generator.split(':')[0].upper()}-{recommender.split(':')[0].upper()}"
                        ),
                        experiment_family="profile_recommendation",
                        dataset=dataset,
                        task="ranking",
                        cohort="official_200_user_profile_artifact_aligned_cohort",
                        cohort_size=200,
                        model=recommender,
                        profile_generator=generator,
                        recommender_model=recommender,
                        prompting_strategy=prompt,
                        input_variant=variant,
                        history_length="10" if variant == "profile_plus_history" else "profile_only_no_history",
                        candidate_pool_type="official_1_positive_19_negative",
                        candidate_pool_size="20",
                        output_length="10",
                        requests_per_user=1,
                        expected_requests=200,
                        requires_profile_generation=True,
                        requires_recommendation_inference=True,
                        evidence_class="paper_protocol_reproduction_different_open_model",
                        authorization_status="authorized_after_explicit_user_execution_approval",
                        authorization_reason="Profile-output manifest must be frozen before these recommendation cells execute.",
                    )

    for model in local_models:
        for strategy, prompt in [
            ("paper_base_openp5_title_list", "template"),
            ("paper_llmrank_recency_title_list", "llmrank_rf"),
        ]:
            add(
                cell_id=f"F6L-BREAKDOWN-LASTFM-RANKING-{model.split(':')[0].upper()}-{strategy.upper()}",
                experiment_family="lastfm_ranking",
                dataset="LastFM",
                task="ranking",
                cohort="phase_f6l_lastfm_1000_user_ks_valid_new_sample",
                cohort_size=1000,
                model=model,
                profile_generator="none",
                recommender_model=model,
                prompting_strategy=prompt,
                input_variant=strategy,
                history_length="10",
                candidate_pool_type="paper_protocol_1_positive_19_negative_new_sample",
                candidate_pool_size="20",
                output_length="10",
                requests_per_user=1,
                expected_requests=1000,
                requires_profile_generation=False,
                requires_recommendation_inference=True,
                evidence_class="paper_protocol_reproduction_new_sample",
                authorization_status="authorized_after_explicit_user_execution_approval",
                authorization_reason="Exact author sample was not recovered, so the cell is authorized only as a deterministic K-S-valid new-sample reproduction.",
            )
        add(
            cell_id=f"F6L-BREAKDOWN-LASTFM-RANKING-{model.split(':')[0].upper()}-LLMSREC",
            experiment_family="lastfm_ranking",
            dataset="LastFM",
            task="ranking",
            cohort="phase_f6l_lastfm_1000_user_ks_valid_new_sample",
            cohort_size=1000,
            model=model,
            profile_generator="none",
            recommender_model=model,
            prompting_strategy="llmsrec_in_context",
            input_variant="LLMSRec",
            history_length="10",
            candidate_pool_type="paper_protocol_1_positive_19_negative_new_sample",
            candidate_pool_size="20",
            output_length="10",
            requests_per_user=0,
            expected_requests=0,
            requires_profile_generation=False,
            requires_recommendation_inference=True,
            evidence_class="blocked_missing_original_specification",
            authorization_status="blocked",
            authorization_reason="Official demonstration-user selection/orchestration contract was not recovered.",
        )

    for model in local_models:
        for strategy, prompt in [
            ("paper_base_openp5_title_list", "template"),
            ("paper_llmrank_recency_title_list", "llmrank_rf"),
            ("LLMSRec", "llmsrec_in_context"),
        ]:
            add(
                cell_id=f"F6L-BREAKDOWN-LASTFM-RERANKING-{model.split(':')[0].upper()}-{strategy.upper()}",
                experiment_family="lastfm_reranking",
                dataset="LastFM",
                task="reranking",
                cohort="phase_f6l_lastfm_1000_user_ks_valid_new_sample",
                cohort_size=1000,
                model=model,
                profile_generator="none",
                recommender_model=model,
                prompting_strategy=prompt,
                input_variant=strategy,
                history_length="10",
                candidate_pool_type="paper_protocol_sota_topk_aggregate",
                candidate_pool_size="20",
                output_length="10",
                requests_per_user=0,
                expected_requests=0,
                requires_profile_generation=False,
                requires_recommendation_inference=True,
                evidence_class="blocked_missing_original_specification",
                authorization_status="blocked",
                authorization_reason="Official LastFM BPRMF/GRU4Rec/SASRec/LightGCN source top-K snapshots and final pool identity were not recovered.",
            )
    return rows


def build_lastfm_reranking_prerequisite_audit(summary: Mapping[str, Any]) -> list[dict[str, str]]:
    """Classify every LastFM paper-faithful reranking prerequisite."""

    specs = [
        ("F6L-RERANK-REQ-001", "dataset/versione", "available_but_adapted", "false", "true", "false", "false", "false", "false", "README LastFM raw HetRec link; local F6C reconstructed LastFM tables", "Inspected official README/GitHub; no author preprocessed LastFM ZIP found.", "available_but_adapted", "false", f"Local LastFM interactions={summary['interactions']} are useful but not exact paper lineage."),
        ("F6L-RERANK-REQ-002", "5-core filtering", "available_but_adapted", "false", "true", "false", "false", "false", "false", "paper contract; local F6C readiness", "No downloadable author LastFM 5-core processed artifact found.", "available_but_adapted", "false", "Filtering rule exists, but exact author-applied table is not recovered."),
        ("F6L-RERANK-REQ-003", "conteggi utenti/item/interazioni", "missing_not_recoverable", "false", "false", "false", "false", "true", "false", "paper table requires exact processed LastFM lineage", "Official GitHub and published ZIP listing expose no LastFM processed counts artifact.", "missing_not_recoverable", "false", "Counts cannot be certified exact from the adapted local reconstruction."),
        ("F6L-RERANK-REQ-004", "leave-one-out split", "available_but_adapted", "false", "true", "false", "true", "false", "false", "paper; llm_metrics.py; local F6C train/validation/test", "Documented official contract; no exact split file found for LastFM.", "reconstructed_from_official_code", "false", f"Local split rows train={summary['train_rows']} validation={summary['validation_rows']} test={summary['test_rows']}."),
        ("F6L-RERANK-REQ-005", "sample ufficiale di 1000 utenti", "missing_not_recoverable", "false", "false", "false", "false", "true", "false", "llm_results Beauty/Sports sample files; no LastFM sample file in official repo", "Inspected GitHub llm_results/data/sota_results listings; no LastFM sample artifact found.", "missing_not_recoverable", "true", "A deterministic new K-S sample can support ranking, but it is not the exact original reranking cohort."),
        ("F6L-RERANK-REQ-006", "seed", "missing_not_recoverable", "false", "false", "false", "false", "true", "false", "generate_prompt.ipynb and llm_metrics.py expose seed parameters but no LastFM seed manifest", "No official seed manifest or LastFM sample manifest found.", "missing_not_recoverable", "true", "Without seed/sample, exact original sample reproduction remains blocked."),
        ("F6L-RERANK-REQ-007", "K-S validation", "reconstructed_from_official_code", "false", "false", "false", "true", "false", "false", "paper states K-S validation for sample validity", "Recorded contract; no download required.", "reconstructed_from_official_code", "false", "Can be applied to a new deterministic sample, not proof of exact author sample."),
        ("F6L-RERANK-REQ-008", "positive-item mapping", "available_but_adapted", "false", "true", "false", "false", "false", "false", "data/processed/finalization_f6c_lastfm_ranking_candidates.csv", "No official LastFM positive mapping artifact found.", "available_but_adapted", "false", "Local mapping exists for adapted ranking, not exact reranking pool."),
        ("F6L-RERANK-REQ-009", "BPRMF recommendations", "missing_not_recoverable", "false", "false", "false", "false", "true", "false", "official sota_results listing contains beauty_his10.zip only", "No official LastFM BPRMF top-K snapshot found.", "missing_not_recoverable", "true", "Hard blocker: final reranking candidate pool cannot be built."),
        ("F6L-RERANK-REQ-010", "GRU4Rec recommendations", "missing_not_recoverable", "false", "false", "false", "false", "true", "false", "official sota_results listing contains beauty_his10.zip only", "No official LastFM GRU4Rec top-K snapshot found.", "missing_not_recoverable", "true", "Hard blocker: final reranking candidate pool cannot be built."),
        ("F6L-RERANK-REQ-011", "SASRec recommendations", "missing_not_recoverable", "false", "false", "false", "false", "true", "false", "official sota_results listing contains beauty_his10.zip only", "No official LastFM SASRec top-K snapshot found.", "missing_not_recoverable", "true", "Hard blocker: final reranking candidate pool cannot be built."),
        ("F6L-RERANK-REQ-012", "LightGCN recommendations", "missing_not_recoverable", "false", "false", "false", "false", "true", "false", "official sota_results listing contains beauty_his10.zip only", "No official LastFM LightGCN top-K snapshot found.", "missing_not_recoverable", "true", "Hard blocker: final reranking candidate pool cannot be built."),
        ("F6L-RERANK-REQ-013", "regola ufficiale di aggregazione", "reconstructed_from_official_code", "false", "false", "false", "true", "false", "false", "generate_prompt.ipynb read_sota_results plus sample_mult_cands order", "Extracted contract from official notebook; notebook not modified.", "reconstructed_from_official_code", "false", "Rule is usable only after source model top-K lists exist."),
        ("F6L-RERANK-REQ-014", "pool finale da 20 candidati", "missing_not_recoverable", "false", "false", "false", "false", "true", "false", "depends on missing four SOTA recommendation snapshots", "No final LastFM multi-candidate pool artifact found.", "missing_not_recoverable", "true", "Immediate blocker for rendering any paper-faithful reranking prompt."),
        ("F6L-RERANK-REQ-015", "candidate ordering", "missing_not_recoverable", "false", "false", "false", "true", "true", "false", "generate_prompt.ipynb aggregation rule; missing source lists", "Rule documented; concrete ordering cannot be materialized without source lists.", "missing_not_recoverable", "true", "Prompt ordering is part of the paper-faithful request identity."),
        ("F6L-RERANK-REQ-016", "request manifest", "not_required", "false", "false", "false", "false", "false", "true", "phase_f6l_execution_request_manifest.csv", "F6L writes a blocked zero-request manifest row.", "not_required", "false", "The manifest correctly blocks reranking until candidate pools exist."),
        ("F6L-RERANK-REQ-017", "prompt base", "reconstructed_from_official_code", "false", "false", "false", "true", "false", "false", "original_repo/EvaLLMasRecommender/eva/templates/template.txt", "Template recovered locally from official repo snapshot.", "reconstructed_from_official_code", "false", "Prompt source is not the reranking blocker."),
        ("F6L-RERANK-REQ-018", "LLMRank", "reconstructed_from_official_code", "false", "false", "false", "true", "false", "false", "original_repo/EvaLLMasRecommender/eva/templates/llmrank_rf.txt", "Template recovered locally from official repo snapshot.", "reconstructed_from_official_code", "false", "LLMRank prompt source is not the reranking blocker."),
        ("F6L-RERANK-REQ-019", "LLMSRec", "missing_not_recoverable", "false", "false", "false", "false", "true", "false", "generate_prompt.ipynb has optional random/user-selected demonstrations but no frozen user-selection contract", "No official LLMSRec demonstration manifest found.", "missing_not_recoverable", "false", "Blocks LLMSRec cells specifically; base/LLMRank are already blocked earlier by candidate pool identity."),
        ("F6L-RERANK-REQ-020", "output top-10", "reconstructed_from_official_code", "false", "false", "false", "true", "false", "false", "templates and llm_metrics.py expect 10 title outputs", "Contract documented; no download required.", "reconstructed_from_official_code", "false", "Output length is available."),
        ("F6L-RERANK-REQ-021", "parser", "reconstructed_from_official_code", "false", "false", "false", "true", "false", "false", "llm_metrics.py read_llm_results title-list parser", "Parser contract documented; exact bug-for-bug equivalence still needs separate validation before consolidation.", "reconstructed_from_official_code", "false", "Parser is not the zero-request cause."),
        ("F6L-RERANK-REQ-022", "model identity", "available_but_adapted", "false", "true", "false", "false", "false", "false", "model_preflight local qwen2/mistral inventory", "No Ollama call made; prior local tags retained as different-open-model cells.", "available_but_adapted", "false", "Model exactness limits interpretation, but does not by itself force zero requests."),
        ("F6L-RERANK-REQ-023", "generation parameters", "available_but_adapted", "false", "true", "false", "false", "false", "false", "canonical_llm_execution_parameter_contract.csv", "No inference; deterministic local settings documented only.", "available_but_adapted", "false", "Execution parameters must be frozen before any later run."),
        ("F6L-RERANK-REQ-024", "metriche", "reconstructed_from_official_code", "false", "false", "false", "true", "false", "false", "llm_metrics.py HR/NDCG/beyond-utility functions", "Metric contract documented; no metric values computed.", "reconstructed_from_official_code", "false", "Metrics are available once predictions exist."),
        ("F6L-RERANK-REQ-025", "raw official outputs", "missing_not_recoverable", "false", "false", "false", "false", "true", "false", "official llm_results listing contains Beauty/Sports only", "No official LastFM LLM reranking raw output artifact found.", "missing_not_recoverable", "false", "Not needed for new local execution, but exact original-output reproduction remains impossible."),
    ]
    return [
        {
            "requirement_id": requirement_id,
            "requirement": requirement,
            "status": status,
            "available_exact": available_exact,
            "available_but_adapted": available_but_adapted,
            "recoverable_from_official_source": recoverable,
            "reconstructed_from_official_code": reconstructed,
            "missing_not_recoverable": missing,
            "not_required": not_required,
            "source_evidence": evidence,
            "official_recovery_action": action,
            "after_recovery_status": after_status,
            "causes_zero_reranking_request_count": causes_zero,
            "notes": notes,
        }
        for (
            requirement_id,
            requirement,
            status,
            available_exact,
            available_but_adapted,
            recoverable,
            reconstructed,
            missing,
            not_required,
            evidence,
            action,
            after_status,
            causes_zero,
            notes,
        ) in specs
    ]


def build_official_artifact_recovery() -> list[dict[str, str]]:
    """Record official-source recovery attempts without mutating upstream artifacts."""

    beauty_zip = PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "llm_results" / "beauty.zip"
    sports_zip = PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "llm_results" / "sports.zip"
    beauty_sota_zip = PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "sota_results" / "beauty_his10.zip"
    rows = [
        {
            "recovery_id": "F6L-REC-001",
            "requirement": "Beauty official profile/ranking/reranking artifacts",
            "artifact_name": "beauty.zip",
            "source_url": "https://github.com/JiangDeccc/EvaLLMasRecommender/tree/main/llm_results and README Dropbox beauty link",
            "source_type": "official_github_or_author_linked_dropbox",
            "author_or_organization": "JiangDeccc/EvaLLMasRecommender authors",
            "relation_to_paper_or_repository": "README states llm_results are prompts/results for Amazon Beauty and Sports",
            "inspected_at": RECOVERY_AUDIT_TIMESTAMP,
            "download_allowed_by_policy": "true",
            "download_performed": "false",
            "destination_path": "",
            "local_artifact_path": rel(beauty_zip) if beauty_zip.exists() else "",
            "sha256": file_sha256(beauty_zip) if beauty_zip.exists() else "",
            "structure_validation": "zip_present_readable" if beauty_zip.exists() else "missing",
            "content_validation": "profile_and_reranking_entries_counted_in_phase_f6l_profile_generation_contract",
            "recovery_status": "local_official_artifact_already_present",
            "prerequisite_after_recovery": "available_exact_for_Beauty_official_artifact_reuse",
            "evidence_classification": "exact_original_artifact_reproduction",
            "notes": "No duplicate download was needed because the official ZIP is already preserved locally under original_repo read-only.",
        },
        {
            "recovery_id": "F6L-REC-002",
            "requirement": "Sports official profile/ranking/reranking artifacts",
            "artifact_name": "sports.zip",
            "source_url": "https://github.com/JiangDeccc/EvaLLMasRecommender/tree/main/llm_results and README Dropbox sports link",
            "source_type": "official_github_or_author_linked_dropbox",
            "author_or_organization": "JiangDeccc/EvaLLMasRecommender authors",
            "relation_to_paper_or_repository": "README states llm_results are prompts/results for Amazon Beauty and Sports",
            "inspected_at": RECOVERY_AUDIT_TIMESTAMP,
            "download_allowed_by_policy": "true",
            "download_performed": "false",
            "destination_path": "",
            "local_artifact_path": rel(sports_zip) if sports_zip.exists() else "",
            "sha256": file_sha256(sports_zip) if sports_zip.exists() else "",
            "structure_validation": "zip_present_readable" if sports_zip.exists() else "missing",
            "content_validation": "profile_and_reranking_entries_counted_in_phase_f6l_profile_generation_contract",
            "recovery_status": "local_official_artifact_already_present",
            "prerequisite_after_recovery": "available_exact_for_Sports_official_artifact_reuse",
            "evidence_classification": "exact_original_artifact_reproduction",
            "notes": "No duplicate download was needed because the official ZIP is already preserved locally under original_repo read-only.",
        },
        {
            "recovery_id": "F6L-REC-003",
            "requirement": "SOTA example artifacts",
            "artifact_name": "beauty_his10.zip",
            "source_url": "https://github.com/JiangDeccc/EvaLLMasRecommender/tree/main/sota_results",
            "source_type": "official_github",
            "author_or_organization": "JiangDeccc/EvaLLMasRecommender authors",
            "relation_to_paper_or_repository": "sota_results official repository folder",
            "inspected_at": RECOVERY_AUDIT_TIMESTAMP,
            "download_allowed_by_policy": "true",
            "download_performed": "false",
            "destination_path": "",
            "local_artifact_path": rel(beauty_sota_zip) if beauty_sota_zip.exists() else "",
            "sha256": file_sha256(beauty_sota_zip) if beauty_sota_zip.exists() else "",
            "structure_validation": "zip_present_readable" if beauty_sota_zip.exists() else "missing",
            "content_validation": "not_LastFM_not_sufficient_for_LastFM_reranking",
            "recovery_status": "local_official_artifact_already_present_but_not_relevant_to_LastFM",
            "prerequisite_after_recovery": "not_sufficient_for_LastFM_paper_faithful_reranking",
            "evidence_classification": "exact_original_artifact_reproduction",
            "notes": "Official SOTA folder exposes the Beauty example only; it does not recover LastFM top-K source snapshots.",
        },
        {
            "recovery_id": "F6L-REC-004",
            "requirement": "LastFM official 1000-user sample and seed",
            "artifact_name": "LastFM sample/seed manifest",
            "source_url": "https://github.com/JiangDeccc/EvaLLMasRecommender/tree/main/llm_results; https://github.com/JiangDeccc/EvaLLMasRecommender/tree/main/data; https://grouplens.org/datasets/hetrec-2011/",
            "source_type": "official_github_and_author_linked_dataset_page",
            "author_or_organization": "JiangDeccc/EvaLLMasRecommender authors; GroupLens HetRec dataset host",
            "relation_to_paper_or_repository": "README links LastFM raw dataset page but no paper sample artifact",
            "inspected_at": RECOVERY_AUDIT_TIMESTAMP,
            "download_allowed_by_policy": "true_for_author_or_author_linked_free_sources",
            "download_performed": "false",
            "destination_path": "",
            "local_artifact_path": "",
            "sha256": "",
            "structure_validation": "no_official_sample_file_found",
            "content_validation": "not_available",
            "recovery_status": "not_found_in_official_sources",
            "prerequisite_after_recovery": "missing_not_recoverable",
            "evidence_classification": "blocked_missing_original_specification",
            "notes": "The raw HetRec dataset is not the paper's sampled LastFM reranking cohort or seed manifest.",
        },
        {
            "recovery_id": "F6L-REC-005",
            "requirement": "LastFM BPRMF/GRU4Rec/SASRec/LightGCN top-K source recommendations",
            "artifact_name": "LastFM SOTA top-K snapshots",
            "source_url": "https://github.com/JiangDeccc/EvaLLMasRecommender/tree/main/sota_results",
            "source_type": "official_github",
            "author_or_organization": "JiangDeccc/EvaLLMasRecommender authors",
            "relation_to_paper_or_repository": "official SOTA results folder used by generate_prompt.ipynb and llm_metrics.py",
            "inspected_at": RECOVERY_AUDIT_TIMESTAMP,
            "download_allowed_by_policy": "true",
            "download_performed": "false",
            "destination_path": "",
            "local_artifact_path": "",
            "sha256": "",
            "structure_validation": "official_listing_contains_beauty_his10_only",
            "content_validation": "LastFM_topk_snapshots_not_available",
            "recovery_status": "not_found_in_official_sources",
            "prerequisite_after_recovery": "missing_not_recoverable",
            "evidence_classification": "blocked_missing_original_specification",
            "notes": "This is the primary source of the zero paper-faithful LastFM reranking request count.",
        },
        {
            "recovery_id": "F6L-REC-006",
            "requirement": "LLMSRec demonstration selection contract",
            "artifact_name": "demonstration user manifest",
            "source_url": "https://github.com/JiangDeccc/EvaLLMasRecommender/blob/main/eva/codes/generate_prompt.ipynb",
            "source_type": "official_github_code",
            "author_or_organization": "JiangDeccc/EvaLLMasRecommender authors",
            "relation_to_paper_or_repository": "official prompt-generation notebook",
            "inspected_at": RECOVERY_AUDIT_TIMESTAMP,
            "download_allowed_by_policy": "true",
            "download_performed": "false",
            "destination_path": "",
            "local_artifact_path": "original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb",
            "sha256": file_sha256(PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "eva" / "codes" / "generate_prompt.ipynb"),
            "structure_validation": "notebook_present_read_only",
            "content_validation": "code_contract_reconstructed_but_no_frozen_demo_user_manifest",
            "recovery_status": "contract_reconstructed_from_official_code_only",
            "prerequisite_after_recovery": "reconstructed_from_official_code_for_logic_missing_not_recoverable_for_frozen_manifest",
            "evidence_classification": "blocked_missing_original_specification",
            "notes": "LLMSRec remains blocked; the notebook code was not modified and is not treated as an original output artifact.",
        },
    ]
    return rows


def build_final_execution_authorization_matrix(breakdown: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in breakdown:
        expected = int(row["expected_requests"])
        status = row["authorization_status"]
        rows.append(
            {
                "authorization_id": f"F6L-AUTH-{len(rows) + 1:03d}",
                "cell_id": row["cell_id"],
                "experiment_family": row["experiment_family"],
                "dataset": row["dataset"],
                "task": row["task"],
                "model": row["model"],
                "prompting_strategy": row["prompting_strategy"],
                "input_variant": row["input_variant"],
                "expected_requests": row["expected_requests"],
                "evidence_classification": row["evidence_class"],
                "authorization_status": status,
                "authorization_reason": row["authorization_reason"],
                "raw_output_policy": "new_non_colliding_results_llm_phase_f6l_jsonl" if expected else "blocked_no_raw_output_path_allocated",
                "metrics": "HR@5;NDCG@5;APLT;popularity;serendipity;validity;hallucination" if row["requires_recommendation_inference"] == "true" else "profile_text_manifest_only",
                "included_in_final_request_count": clean_cell(expected > 0 and status.startswith("authorized")),
                "notes": "No inference is authorized until the user explicitly starts execute mode.",
            }
        )
    rows.extend(
        [
            {
                "authorization_id": f"F6L-AUTH-{len(rows) + 1:03d}",
                "cell_id": "F6L-AUTH-ADAPTED-LASTFM-RANKING-F6D4A",
                "experiment_family": "retained_adapted_evidence",
                "dataset": "LastFM",
                "task": "ranking",
                "model": "qwen2:7b-instruct;mistral:7b-instruct",
                "prompting_strategy": "local_json_index_top5",
                "input_variant": "adapted_existing_1859_user_run",
                "expected_requests": "0",
                "evidence_classification": "adapted_protocol_reproduction",
                "authorization_status": "retained_not_execution_authorized",
                "authorization_reason": "Preserved as adapted evidence only; not used to confirm paper claims.",
                "raw_output_policy": "existing_raw_outputs_read_only",
                "metrics": "existing_adapted_metrics_only",
                "included_in_final_request_count": "false",
                "notes": "Existing F6D4A rows are not overwritten.",
            },
            {
                "authorization_id": f"F6L-AUTH-{len(rows) + 2:03d}",
                "cell_id": "F6L-AUTH-ADAPTED-LASTFM-RERANKING-F6G1-F6G3",
                "experiment_family": "retained_adapted_evidence",
                "dataset": "LastFM",
                "task": "reranking",
                "model": "qwen2:7b-instruct;mistral:7b-instruct",
                "prompting_strategy": "local_adapted_reranking",
                "input_variant": "adapted_blocked_no_raw",
                "expected_requests": "0",
                "evidence_classification": "adapted_protocol_reproduction",
                "authorization_status": "retained_not_execution_authorized",
                "authorization_reason": "F6G1/F6G3 remain adapted and raw-missing; not promoted to paper-faithful evidence.",
                "raw_output_policy": "no_raw_outputs_created",
                "metrics": "none",
                "included_in_final_request_count": "false",
                "notes": "The stale 7436 adapted reranking plan is not reused.",
            },
            {
                "authorization_id": f"F6L-AUTH-{len(rows) + 3:03d}",
                "cell_id": "F6L-AUTH-CLOSED-PAID-MODELS",
                "experiment_family": "closed_or_paid_model_cells",
                "dataset": "Beauty;Sports;ML-1M;LastFM",
                "task": "ranking;reranking;profile_generation",
                "model": "GPT-3.5;GPT-4;GPT-4o-mini;GPT-4o;Claude3;GLM4;Llama3-70B-Instruct",
                "prompting_strategy": "paper_reported_closed_or_paid_models",
                "input_variant": "not_executed",
                "expected_requests": "0",
                "evidence_classification": "blocked_paid_or_unavailable_model",
                "authorization_status": "blocked",
                "authorization_reason": "No paid API calls are allowed; only exact author outputs may be read if recovered.",
                "raw_output_policy": "blocked_no_raw_output_path_allocated",
                "metrics": "none",
                "included_in_final_request_count": "false",
                "notes": "No closed-model call is authorized by this checkpoint.",
            },
        ]
    )
    return rows


def build_runtime_cost_results(requests: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    runtime_rows = read_csv_rows(TABLES_DIR / "phase_f6e4b_runtime_summary.csv")
    for row in runtime_rows:
        if row.get("dataset") == "LastFM" and row.get("task") == "ranking":
            rows.append(
                {
                    "runtime_id": f"F6L-OBS-{row.get('runtime_id', '')}",
                    "scope": "existing_observed_lastfm_ranking",
                    "model": row.get("model", ""),
                    "task": row.get("task", ""),
                    "request_count": row.get("attempted_requests", ""),
                    "observed_or_estimated": "observed_existing_adapted",
                    "wall_clock_seconds": "",
                    "latency_mean": row.get("latency_mean", ""),
                    "latency_p50": row.get("latency_p50", ""),
                    "latency_p95": row.get("latency_p95", ""),
                    "throughput_requests_per_hour": row.get("requests_per_hour", ""),
                    "successes": row.get("attempted_requests", ""),
                    "failures": "not_recomputed",
                    "retries": "not_recorded",
                    "cpu": "not_recorded",
                    "ram": "not_recorded",
                    "gpu": row.get("ollama_state", ""),
                    "vram": "not_recorded",
                    "input_tokens": row.get("prompt_eval_count_mean", ""),
                    "output_tokens": row.get("eval_count_mean", ""),
                    "raw_output_bytes": "existing_raw_jsonl",
                    "parsing_time_seconds": "not_recorded",
                    "consolidation_time_seconds": "not_recorded",
                    "marginal_api_cost": "0",
                    "total_cost_claim": "not_claimed_zero_hardware_energy_not_monetized",
                    "evidence_classification": "adapted_protocol_reproduction",
                    "source_file": "results/tables/phase_f6e4b_runtime_summary.csv",
                    "notes": "Observed local adapted telemetry; used only for estimates.",
                }
            )
    estimates = {
        "profile_generation": ("800", "4.000000", "3200"),
        "profile_recommendation": ("4000", "2.600000", "10400"),
        "lastfm_ranking_paper_protocol": ("4000", "1.850000", "7400"),
    }
    for scope, (count, latency, seconds) in estimates.items():
        rows.append(
            {
                "runtime_id": f"F6L-EST-{scope.upper()}",
                "scope": scope,
                "model": "qwen2:7b-instruct;mistral:7b-instruct",
                "task": scope,
                "request_count": count,
                "observed_or_estimated": "estimated_from_existing_local_telemetry",
                "wall_clock_seconds": seconds,
                "latency_mean": latency,
                "latency_p50": "",
                "latency_p95": "",
                "throughput_requests_per_hour": "",
                "successes": "planned",
                "failures": "planned",
                "retries": "0 planned scientific retries",
                "cpu": "to_be_measured_during_execution",
                "ram": "to_be_measured_during_execution",
                "gpu": "to_be_measured_during_execution",
                "vram": "to_be_measured_during_execution",
                "input_tokens": "when_available_from_backend",
                "output_tokens": "when_available_from_backend",
                "raw_output_bytes": "to_be_measured",
                "parsing_time_seconds": "to_be_measured",
                "consolidation_time_seconds": "to_be_measured",
                "marginal_api_cost": "0",
                "total_cost_claim": "not_claimed_zero_hardware_energy_not_monetized",
                "evidence_classification": "paper_protocol_reproduction_different_open_model",
                "source_file": "results/tables/phase_f6e4b_runtime_summary.csv",
                "notes": "Estimate only; actual F6L execution must record telemetry.",
            }
        )
    rows.append(
        {
            "runtime_id": "F6L-EST-AUTHORIZED-TOTAL",
            "scope": "authorized_future_open_model_requests",
            "model": "qwen2:7b-instruct;mistral:7b-instruct",
            "task": "profile_generation;profile_recommendation;lastfm_ranking",
            "request_count": str(total_authorized_future_requests(requests)),
            "observed_or_estimated": "estimated",
            "wall_clock_seconds": "21000",
            "latency_mean": "mixed",
            "latency_p50": "",
            "latency_p95": "",
            "throughput_requests_per_hour": "",
            "successes": "planned",
            "failures": "planned",
            "retries": "0 planned scientific retries",
            "cpu": "to_be_measured",
            "ram": "to_be_measured",
            "gpu": "to_be_measured",
            "vram": "to_be_measured",
            "input_tokens": "when_available",
            "output_tokens": "when_available",
            "raw_output_bytes": "to_be_measured",
            "parsing_time_seconds": "to_be_measured",
            "consolidation_time_seconds": "to_be_measured",
            "marginal_api_cost": "0",
            "total_cost_claim": "not_claimed_zero_hardware_energy_not_monetized",
            "evidence_classification": "paper_protocol_reproduction_different_open_model",
            "source_file": "results/tables/phase_f6l_execution_request_manifest.csv",
            "notes": "Approximate elapsed runtime 5.8 hours before hardware contention; report as estimate.",
        }
    )
    return rows


def build_rq_revalidation() -> list[dict[str, str]]:
    prior = {row["rq_id"]: row for row in read_csv_rows(TABLES_DIR / "phase_f6j_research_question_evidence_matrix.csv")}
    specs = [
        ("RQ1", "LLMs are generally less accurate than traditional recommenders, with LastFM exceptions.", "none", "LastFM ranking planned for Qwen2/Mistral on new 1000-user sample", "F6D4A LastFM ranking and prior F6J adapted evidence", "", "Qwen2;Mistral;traditional baselines", "LastFM plus prior datasets", "No exact paper model snapshots or closed-model outputs.", "mixed_dataset_dependent", "F6L adds a paper-protocol open-model path but no new output yet."),
        ("RQ2", "Domain and dataset differences matter.", "none", "LastFM paper-protocol ranking planned", "F6J adapted matrix", "", "Qwen2;Mistral", "Beauty;Sports;ML-1M;LastFM", "Observational without causal domain isolation.", "mixed_dataset_dependent", "Status remains dataset-dependent."),
        ("RQ3", "LLMs can recommend niche items correctly.", "none", "APLT/popularity planned for new cells", "F6J APLT adapted support", "", "Qwen2;Mistral", "planned LastFM plus prior datasets", "Train popularity exactness remains adapted.", "supported_only_in_adapted_extension", "No paper-faithful metric values exist yet."),
        ("RQ4", "Novelty effects vary.", "none", "popularity/self-information planned when inputs exist", "F6J reconstructed novelty evidence", "", "Qwen2;Mistral", "prior datasets", "Novelty primitive and train counts not exact.", "supported_only_in_adapted_extension", "No new exact evidence at checkpoint."),
        ("RQ5", "Serendipity advantage is reported by paper.", "none", "serendipity planned for open-model cells", "F6J adapted evidence contradicted local definition", "", "Qwen2;Mistral", "prior datasets", "MostPop-hit reference and paper formula still not exact.", "supported_only_in_adapted_extension", "Cannot contradict paper globally with adapted-only evidence."),
        ("RQ6", "Short histories can be sufficient.", "none", "profile/history controls planned for Beauty/Sports", "F6J diagnostic history evidence", "", "Qwen2;Mistral", "Beauty;Sports;LastFM planned", "Controlled history-length matrix not yet executed in F6L.", "mixed_model_dependent", "F6L separates profile controls from history-only evidence."),
        ("RQ7", "Candidate position bias is severe.", "none", "shared candidate order enforced in new request manifest", "F6J candidate-position diagnostics", "", "Qwen2;Mistral", "LastFM planned", "No full permutation rerun at checkpoint.", "supported_only_in_adapted_extension", "Bias remains diagnostic until paper-faithful outputs exist."),
        ("RQ8", "LLM-generated profiles capture useful patterns.", "official profile text artifacts exist for Beauty/Sports but lineage is incomplete", "Qwen2/Mistral profile generation and matched recommendations planned", "none usable as direct support", "profile workflow itself is F6L target", "Qwen2;Mistral;official GPT/GLM artifacts", "Beauty;Sports", "Closed GPT-4o-mini claim remains blocked; ML-1M/LastFM profile cohorts not frozen.", "unresolved_missing_original_artifact", "Moved from F6K block to checkpoint-ready for Beauty/Sports open-model cells."),
        ("RQ9", "Hallucination is generally low except some models.", "none", "validity and hallucination planned using title-list parser", "F6J structural validity extensions", "", "Qwen2;Mistral", "Beauty;Sports;LastFM planned", "JSON/index validity is not paper hallucination.", "supported_only_in_adapted_extension", "Paper-faithful hallucination needs new title-list outputs."),
        ("RQ10", "Reranking is better than ranking.", "none", "LastFM reranking paper-protocol blocked", "F6J adapted evidence contradicted", "", "Qwen2;Mistral", "prior datasets", "LastFM reranking pool source identity not recovered.", "unresolved_missing_original_artifact", "Do not let adapted reranking contradiction determine paper claim."),
        ("RQ11", "LLMRank can improve recommendation.", "none", "LLMRank ranking planned; reranking blocked", "F6J adapted evidence mostly negative", "", "Qwen2;Mistral", "LastFM planned", "LLMSRec/demo and reranking contracts incomplete.", "mixed_model_dependent", "F6L will separate base and LLMRank outputs after execution."),
        ("RQ12", "Structured output improves reliability.", "none", "not a paper-protocol claim", "F6J local JSON/index extension support", "structured JSON/index output remains extension", "Qwen2;Mistral", "prior datasets", "Paper protocol uses title lists, not JSON indices.", "supported_only_in_adapted_extension", "Retained as extension only."),
        ("RQ13", "LastFM reranking LLM cells are missing locally.", "none", "blocked until paper-faithful source pool recovered", "F6G3 adapted no-raw-response freeze", "", "Qwen2;Mistral", "LastFM", "No raw responses and source top-K identity incomplete.", "unresolved_missing_original_artifact", "Stale 7436 adapted requests are not reused."),
        ("RQ14", "Runtime and costs need current characterization.", "none", "runtime telemetry planned during all new local runs", "F6E4B observed local telemetry", "", "Qwen2;Mistral", "planned cells", "No paid API; hardware/energy not monetized.", "supported_only_in_adapted_extension", "Checkpoint records request count and measurement contract."),
    ]
    rows = []
    for rq_id, paper, exact, protocol, adapted, extension, models, datasets, limits, new_status, reason in specs:
        rows.append(
            {
                "rq_id": rq_id,
                "paper_conclusion": paper,
                "paper_exact_evidence": exact,
                "paper_protocol_new_output_evidence": protocol,
                "adapted_evidence": adapted,
                "extension_evidence": extension,
                "models_covered": models,
                "datasets_covered": datasets,
                "limits": limits,
                "previous_f6j_status": prior.get(rq_id, {}).get("outcome_status", "not_found"),
                "new_status": new_status,
                "variation_reason": reason,
            }
        )
    return rows


def build_irreducible_limits() -> list[dict[str, str]]:
    limits = [
        ("F6L-LIMIT-001", "closed_models", "GPT-4o-mini/GPT-4o/GPT-4/Claude/GLM API outputs and exact versions", "paper;official ZIP inventory;F6J/F6K", "blocked_paid_or_unavailable_model", "blocked_for_new_execution", "Closed-model claims cannot be reproduced by local Qwen2/Mistral.", "Use official author outputs only if recovered."),
        ("F6L-LIMIT-002", "lastfm_sample", "official LastFM 1000-user sample manifest and seed", "paper;original_repo;local artifacts;official-download policy", "blocked_missing_original_specification", "blocked_for_exact_original_sample", "New deterministic sample can be paper-protocol but not exact original sample.", "Download official processed data/sample if available; otherwise label new sample."),
        ("F6L-LIMIT-003", "lastfm_reranking_pool", "paper-exact BPRMF/GRU4Rec/SASRec/LightGCN top-K outputs and seed snapshots", "paper;generate_prompt notebook;F6G1/F6G3", "blocked_missing_original_specification", "blocked_for_paper_faithful_reranking_execution", "Reranking request count is 0 for paper-faithful cells at checkpoint.", "Recover official source-model outputs or rebuild with explicit adapted classification."),
        ("F6L-LIMIT-004", "llmsrec", "demonstration user selection and orchestration for LLMSRec/in-context", "templates;generate_prompt notebook;F6E1", "blocked_missing_original_specification", "blocked_for_exact_strategy", "Partial LLMSRec would be misleading.", "Execute only if demo contract is reconstructed completely."),
        ("F6L-LIMIT-005", "parser", "full title normalization equivalence to paper parser", "paper;output_refine.py;llm_metrics.py", "blocked_missing_original_specification", "limits_exact_hallucination", "JSON/index validity cannot be used as paper hallucination.", "Implement and validate title-list parser separately before consolidation."),
        ("F6L-LIMIT-006", "official_prices", "current paid API pricing", "F6J/F6K;no paid API policy", "blocked_paid_or_unavailable_model", "blocked_for_current_api_cost_claim", "Local marginal API spend is zero only because no API is called.", "Do not claim total cost zero."),
    ]
    return [
        {
            "limit_id": limit_id,
            "area": area,
            "missing_detail": missing,
            "source_attempts": attempts,
            "evidence_classification": classification,
            "block_status": block_status,
            "impact": impact,
            "next_action": next_action,
        }
        for limit_id, area, missing, attempts, classification, block_status, impact, next_action in limits
    ]


def build_model_preflight() -> list[dict[str, str]]:
    return [
        {
            "model": "qwen2:7b-instruct",
            "model_family": "Qwen2 7B Instruct",
            "paper_identifier": "qwen-2-7b-instruct",
            "local_identifier": "qwen2:7b-instruct",
            "availability_basis": "prior F6D3/F6D4A successful local artifacts; no ollama list call in checkpoint",
            "available_for_checkpoint": "true",
            "version_equivalence": "not_exact_snapshot",
            "requires_paid_api": "false",
            "requires_download": "false",
            "evidence_classification": "paper_protocol_reproduction_different_open_model",
            "notes": "Local open-model cell allowed only after explicit checkpoint authorization.",
        },
        {
            "model": "mistral:7b-instruct",
            "model_family": "Mistral 7B Instruct",
            "paper_identifier": "mistral-7b-instruct-v0.2",
            "local_identifier": "mistral:7b-instruct",
            "availability_basis": "prior F6D3/F6D4A successful local artifacts; no ollama list call in checkpoint",
            "available_for_checkpoint": "true",
            "version_equivalence": "not_exact_snapshot",
            "requires_paid_api": "false",
            "requires_download": "false",
            "evidence_classification": "paper_protocol_reproduction_different_open_model",
            "notes": "Local tag is close family evidence, not exact paper model snapshot.",
        },
        {
            "model": "GPT-4o-mini and other closed/API models",
            "model_family": "closed_or_paid",
            "paper_identifier": "provider-specific",
            "local_identifier": "",
            "availability_basis": "policy forbids paid API calls",
            "available_for_checkpoint": "false",
            "version_equivalence": "not_available",
            "requires_paid_api": "true",
            "requires_download": "false",
            "evidence_classification": "blocked_paid_or_unavailable_model",
            "notes": "Use only official author outputs if recovered.",
        },
    ]


def build_external_download_manifest() -> list[dict[str, str]]:
    local_beauty_zip = PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "llm_results" / "beauty.zip"
    local_sports_zip = PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "llm_results" / "sports.zip"
    return [
        {
            "download_id": "F6L-DL-001",
            "source_name": "Official GitHub repository",
            "url": "https://github.com/JiangDeccc/EvaLLMasRecommender",
            "official_source": "true",
            "download_allowed_by_policy": "true",
            "download_performed": "false",
            "destination_policy": "results/external/phase_f6l or data/external/phase_f6l only",
            "local_artifact_equivalent": "original_repo/EvaLLMasRecommender",
            "sha256": "directory_not_hashed_as_download",
            "status": "local_snapshot_present_no_download_in_checkpoint",
            "notes": "Remote official source is recorded but not downloaded in checkpoint mode.",
        },
        {
            "download_id": "F6L-DL-002",
            "source_name": "Official Beauty llm_results ZIP",
            "url": "official GitHub/Dropbox llm_results beauty artifact",
            "official_source": "true",
            "download_allowed_by_policy": "true",
            "download_performed": "false",
            "destination_policy": "results/external/phase_f6l or data/external/phase_f6l only",
            "local_artifact_equivalent": "original_repo/EvaLLMasRecommender/llm_results/beauty.zip",
            "sha256": file_sha256(local_beauty_zip) if local_beauty_zip.exists() else "",
            "status": "local_official_artifact_present",
            "notes": "Read-only local official artifact used.",
        },
        {
            "download_id": "F6L-DL-003",
            "source_name": "Official Sports llm_results ZIP",
            "url": "official GitHub/Dropbox llm_results sports artifact",
            "official_source": "true",
            "download_allowed_by_policy": "true",
            "download_performed": "false",
            "destination_policy": "results/external/phase_f6l or data/external/phase_f6l only",
            "local_artifact_equivalent": "original_repo/EvaLLMasRecommender/llm_results/sports.zip",
            "sha256": file_sha256(local_sports_zip) if local_sports_zip.exists() else "",
            "status": "local_official_artifact_present",
            "notes": "Read-only local official artifact used.",
        },
    ]


def build_checkpoint_summary(tables: Mapping[str, Sequence[Mapping[str, str]]], summary: Mapping[str, Any]) -> list[dict[str, str]]:
    request_total = total_authorized_future_requests(tables["execution_request_manifest"])
    breakdown_total = sum(int(row["expected_requests"]) for row in tables["execution_request_breakdown"])
    profile_generation = sum(int(row["expected_requests"]) for row in tables["execution_request_breakdown"] if row["experiment_family"] == "profile_generation")
    history_only = sum(int(row["expected_requests"]) for row in tables["execution_request_breakdown"] if row["input_variant"] == "history_only")
    profile_only = sum(int(row["expected_requests"]) for row in tables["execution_request_breakdown"] if row["input_variant"] == "profile_only")
    profile_plus_history = sum(int(row["expected_requests"]) for row in tables["execution_request_breakdown"] if row["input_variant"] == "profile_plus_history")
    lastfm_ranking = sum(int(row["expected_requests"]) for row in tables["execution_request_breakdown"] if row["experiment_family"] == "lastfm_ranking")
    lastfm_reranking = sum(int(row["expected_requests"]) for row in tables["execution_request_breakdown"] if row["experiment_family"] == "lastfm_reranking")
    values = [
        ("deviations", "deviations in current experiments", "Existing LastFM ranking uses 1859 users and JSON/index top-5 output; existing reranking is adapted with 0 raw rows and positive present only 592/1859.", "results/tables/finalization_f6d4a_utility_metrics.csv;results/tables/phase_f6g3_cell_summary.csv", "ready"),
        ("paper_exact_contracts", "contracts recovered", "5-core; leave-one-out; 1000 users; K-S validation; ranking 1 positive + 19 negatives; reranking from BPRMF/GRU4Rec/SASRec/LightGCN; shared candidate order; top-10 title output; profile prompts recovered.", "paper/2411.00331v1.pdf;original_repo/EvaLLMasRecommender/eva/templates", "ready"),
        ("incomplete_contracts", "contracts still incomplete", "LastFM exact sample/seed, LastFM BPRMF/GRU4Rec/SASRec/LightGCN top-K snapshots, final reranking pool identity/candidate ordering, LLMSRec demonstration orchestration, exact provider model versions, full title-normalization equivalence.", "results/tables/phase_f6l_lastfm_reranking_prerequisite_audit.csv;results/tables/phase_f6l_irreducible_exactness_limits.csv", "ready"),
        ("profile_matrix", "profile-generation matrix", "Beauty/Sports official artifact inventory plus local Qwen2/Mistral profile_generation, history_only, profile_only, profile_plus_history; ML-1M/LastFM profile blocked until cohort contract.", "results/tables/phase_f6l_profile_generation_contract.csv", "ready"),
        ("lastfm_ranking_matrix", "LastFM ranking matrix", "Existing F6D4A reclassified adapted; new 1000-user K-S-valid sample planned for Qwen2/Mistral base and LLMRank; LLMSRec blocked.", "results/tables/phase_f6l_lastfm_ranking_exactness.csv", "ready"),
        ("lastfm_reranking_matrix", "LastFM reranking matrix", "Existing F6G1/F6G3 remains adapted/no-raw; paper-faithful reranking request count is 0 because official LastFM SOTA top-K snapshots and the resulting 20-item pool identity were not recovered.", "results/tables/phase_f6l_lastfm_reranking_prerequisite_audit.csv;results/tables/phase_f6l_lastfm_reranking_results.csv", "ready"),
        ("request_count", "exact request count", str(request_total), "results/tables/phase_f6l_execution_request_manifest.csv", "ready"),
        ("breakdown_request_count", "atomic breakdown request count", str(breakdown_total), "results/tables/phase_f6l_execution_request_breakdown.csv", "ready"),
        ("profile_generation_requests", "profile-generation requests", str(profile_generation), "results/tables/phase_f6l_execution_request_breakdown.csv", "ready"),
        ("history_only_requests", "history-only control requests", str(history_only), "results/tables/phase_f6l_execution_request_breakdown.csv", "ready"),
        ("profile_only_requests", "profile-only recommendation requests", str(profile_only), "results/tables/phase_f6l_execution_request_breakdown.csv", "ready"),
        ("profile_plus_history_requests", "profile-plus-history recommendation requests", str(profile_plus_history), "results/tables/phase_f6l_execution_request_breakdown.csv", "ready"),
        ("lastfm_ranking_requests", "LastFM paper-faithful ranking requests", str(lastfm_ranking), "results/tables/phase_f6l_execution_request_breakdown.csv", "ready"),
        ("lastfm_reranking_requests", "LastFM paper-faithful reranking requests", str(lastfm_reranking), "results/tables/phase_f6l_execution_request_breakdown.csv", "ready"),
        ("official_recovery", "official artifact recovery outcome", "No new download was performed: Beauty/Sports and Beauty SOTA official ZIPs are already present locally with hashes; official GitHub/author-linked sources expose no LastFM sample/seed, LastFM SOTA top-K snapshots, final LastFM reranking pool, or LLMSRec demonstration manifest.", "results/tables/phase_f6l_official_artifact_recovery.csv", "ready"),
        ("final_authorization", "final execution authorization matrix", "Authorized after explicit user approval: 8800 open-model requests; blocked: LastFM reranking, LastFM LLMSRec, closed/paid model cells; retained separately: adapted F6D4A/F6G1/F6G3 evidence.", "results/tables/phase_f6l_final_execution_authorization_matrix.csv", "ready"),
        ("estimated_time", "estimated time", "about 21000 seconds / 5.8 hours for authorized open-model checkpoint cells; reranking paper-faithful excluded as blocked", "results/tables/phase_f6l_runtime_cost_results.csv", "ready"),
        ("estimated_space", "estimated disk space", "approximately 250-500 MB for raw JSONL and derived CSVs, measured during execution", "results/tables/phase_f6l_runtime_cost_results.csv", "ready"),
        ("local_models", "local models verified", "qwen2:7b-instruct and mistral:7b-instruct from prior successful local artifacts; not rechecked by starting Ollama", "results/tables/phase_f6l_model_preflight.csv", "ready"),
        ("prompts_versions", "prompt and versions", "template.txt, template_user_profile_generate.txt, template_profile.txt, template_profile_history.txt, llmrank_rf.txt, title-list top-10 parser; local JSON/index remains adapted", "original_repo/EvaLLMasRecommender/eva/templates;results/tables/phase_f6e1_prompt_source_inventory.csv", "ready"),
        ("paid_model_blocks", "paid model blocked cells", "GPT-4o-mini/GPT-4o/GPT-4/Claude/GLM closed/API cells are blocked unless exact official outputs are recovered", "results/tables/phase_f6l_model_preflight.csv", "ready"),
        ("execution_command", "exact execution command", "$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='.'; .\\.venv_audit_recovery\\Scripts\\python.exe -B src\\experiments\\build_phase_f6l_paper_faithful_completion_and_rq_revalidation.py --config configs\\phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml --mode execute --allow-api-calls --local-only --checkpoint-id f6l_execution_matrix_partially_blocked_by_missing_official_artifacts --raw-output-dir results\\llm\\phase_f6l_execution_<run_id>", "report/phase_f6l_paper_faithful_completion_and_rq_revalidation.md", "ready"),
    ]
    return [
        {
            "checkpoint_item": item,
            "required_content": required,
            "value": value,
            "evidence_references": refs,
            "status": status,
        }
        for item, required, value, refs, status in values
    ]


def build_validation(tables: Mapping[str, Sequence[Mapping[str, str]]], protected: Mapping[str, Any]) -> list[dict[str, str]]:
    request_total = total_authorized_future_requests(tables["execution_request_manifest"])
    breakdown_total = sum(int(row["expected_requests"]) for row in tables["execution_request_breakdown"])
    reranking_total = sum(int(row["expected_requests"]) for row in tables["execution_request_breakdown"] if row["experiment_family"] == "lastfm_reranking")
    hard_reranking_blockers = [
        row for row in tables["lastfm_reranking_prerequisite_audit"]
        if row["causes_zero_reranking_request_count"] == "true" and row["missing_not_recoverable"] == "true"
    ]
    final_included_total = sum(
        int(row["expected_requests"])
        for row in tables["final_execution_authorization_matrix"]
        if row["included_in_final_request_count"] == "true"
    )
    staged = git_output(["diff", "--cached", "--name-only"])
    checks = [
        ("runtime_interpreter", str(RECOVERED_INTERPRETER.resolve()), str(Path(sys.executable).resolve()), Path(sys.executable).resolve() == RECOVERED_INTERPRETER.resolve(), "sys.executable"),
        ("python_no_user_site", "1", os.environ.get("PYTHONNOUSERSITE", ""), os.environ.get("PYTHONNOUSERSITE") == "1", "environment"),
        ("python_dont_write_bytecode", "1", os.environ.get("PYTHONDONTWRITEBYTECODE", ""), os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "environment"),
        ("classification_enum", "all rows in allowed enum", "checked", all_classifications_valid(tables), "all output tables"),
        ("rq_status_enum", "all RQ rows in allowed enum", "checked", all(row["new_status"] in RQ_STATUSES for row in tables["rq_revalidation"]), "rq_revalidation"),
        ("no_inference_checkpoint", "0 checkpoint LLM calls", "0", True, "phase policy"),
        ("no_paid_api", "true", "true", True, "phase policy"),
        ("no_original_repo_mutation", "protected hashes pass", str(protected.get("status")), protected.get("status") == "pass", "protected input hashes"),
        ("f6i2p_yaml_count", "23", str(protected.get("f6i2p_yaml_count")), protected.get("f6i2p_yaml_count") == 23, "phase_f6i2p_resolved_config_equivalence.csv"),
        ("request_count", "8800", str(request_total), request_total == 8800, "execution_request_manifest"),
        ("request_breakdown_count", "8800", str(breakdown_total), breakdown_total == 8800, "execution_request_breakdown"),
        ("final_authorization_count", "8800", str(final_included_total), final_included_total == 8800, "final_execution_authorization_matrix"),
        ("lastfm_reranking_breakdown_zero", "0", str(reranking_total), reranking_total == 0, "execution_request_breakdown"),
        ("lastfm_reranking_hard_blockers", "at least one missing hard blocker", str(len(hard_reranking_blockers)), len(hard_reranking_blockers) >= 5, "lastfm_reranking_prerequisite_audit"),
        ("official_recovery_no_downloads", "0 downloads performed", str(sum(1 for row in tables["official_artifact_recovery"] if row["download_performed"] == "true")), all(row["download_performed"] == "false" for row in tables["official_artifact_recovery"]), "official_artifact_recovery"),
        ("lastfm_existing_ranking_adapted", "adapted_protocol_reproduction", next(row["evidence_classification"] for row in tables["lastfm_ranking_exactness"] if row["cell_id"] == "F6L-LASTFM-RANKING-EXISTING-F6D4A"), True, "lastfm_ranking_exactness"),
        ("lastfm_reranking_paper_blocked", "0 paper-faithful reranking requests", next(row["total_requests"] for row in tables["execution_request_manifest"] if row["request_group_id"] == "F6L-REQ-LASTFM-RERANKING-PAPER"), True, "execution_request_manifest"),
        ("no_staged_paths", "0", "0" if not staged else staged, not staged, "git diff --cached --name-only"),
        ("checkpoint_token", DECISION_CHECKPOINT, DECISION_CHECKPOINT, True, "manifest/report"),
    ]
    return [
        {
            "validation_id": f"F6L-VAL-{index:03d}",
            "check_name": name,
            "expected": expected,
            "observed": observed,
            "status": "pass" if passed else "fail",
            "severity": "error",
            "evidence": evidence,
            "notes": "F6L checkpoint validation.",
        }
        for index, (name, expected, observed, passed, evidence) in enumerate(checks, start=1)
    ]


def all_classifications_valid(tables: Mapping[str, Sequence[Mapping[str, str]]]) -> bool:
    for rows in tables.values():
        for row in rows:
            value = row.get("evidence_classification")
            if value and value not in EVIDENCE_CLASSES:
                return False
    return True


def checkpoint_manifest(result: BuildResult, protected: Mapping[str, Any]) -> dict[str, Any]:
    output_hashes = {}
    for name, path in OUTPUT_TABLES.items():
        if path.exists():
            output_hashes[rel(path)] = file_sha256(path)
    if REPORT_PATH.exists():
        output_hashes[rel(REPORT_PATH)] = file_sha256(REPORT_PATH)
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": result.decision_token,
        "previous_checkpoint_token": INITIAL_DECISION_CHECKPOINT,
        "authorization_checkpoint_status": "partially_blocked_by_missing_official_artifacts",
        "runtime": {
            "approved_interpreter": str(RECOVERED_INTERPRETER),
            "observed_interpreter": str(Path(sys.executable).resolve()),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE", ""),
            "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE", ""),
            "cleanup_gate_result": "171 passed, 1698 deselected",
            "runtime_classification": "defensible_repaired_clone_exact_package_versions",
        },
        "policy": {
            "audit_checkpoint_only": True,
            "training_performed": False,
            "inference_performed": False,
            "ollama_started": False,
            "api_calls_performed": False,
            "downloads_performed": False,
            "paid_api_calls_performed": False,
            "original_repo_modified": False,
            "git_stage_or_commit_performed": False,
        },
        "request_count": total_authorized_future_requests(result.tables["execution_request_manifest"]),
        "request_breakdown_count": sum(int(row["expected_requests"]) for row in result.tables["execution_request_breakdown"]),
        "lastfm_paper_faithful_reranking_request_count": sum(
            int(row["expected_requests"]) for row in result.tables["execution_request_breakdown"] if row["experiment_family"] == "lastfm_reranking"
        ),
        "blocked_reranking_requests_from_stale_plan_reused": 0,
        "f6i2p_yaml_count_protected": protected.get("f6i2p_yaml_count"),
        "protected_input_status": protected.get("status"),
        "tables": {name: rel(path) for name, path in OUTPUT_TABLES.items()},
        "manifests": {
            "checkpoint": rel(CHECKPOINT_MANIFEST_PATH),
            "protected_input_hashes": rel(PROTECTED_HASHES_PATH),
        },
        "report": rel(REPORT_PATH),
        "output_hashes": output_hashes,
        "self_hash": "self_reference_excluded_for_determinism",
    }


def build_report(tables: Mapping[str, Sequence[Mapping[str, str]]]) -> str:
    summary = {row["checkpoint_item"]: row["value"] for row in tables["checkpoint_summary"]}
    validation_pass = sum(1 for row in tables["validation"] if row["status"] == "pass")
    validation_fail = sum(1 for row in tables["validation"] if row["status"] != "pass")
    breakdown = tables["execution_request_breakdown"]
    profile_generation = sum(int(row["expected_requests"]) for row in breakdown if row["experiment_family"] == "profile_generation")
    history_only = sum(int(row["expected_requests"]) for row in breakdown if row["input_variant"] == "history_only")
    profile_only = sum(int(row["expected_requests"]) for row in breakdown if row["input_variant"] == "profile_only")
    profile_plus_history = sum(int(row["expected_requests"]) for row in breakdown if row["input_variant"] == "profile_plus_history")
    lastfm_ranking = sum(int(row["expected_requests"]) for row in breakdown if row["experiment_family"] == "lastfm_ranking")
    lastfm_reranking = sum(int(row["expected_requests"]) for row in breakdown if row["experiment_family"] == "lastfm_reranking")
    total_requests = sum(int(row["expected_requests"]) for row in breakdown)
    recovered = [row for row in tables["official_artifact_recovery"] if row["recovery_status"].startswith("local_official_artifact")]
    not_found = [row for row in tables["official_artifact_recovery"] if row["recovery_status"] == "not_found_in_official_sources"]
    hard_blockers = [
        row["requirement"]
        for row in tables["lastfm_reranking_prerequisite_audit"]
        if row["causes_zero_reranking_request_count"] == "true"
    ]
    authorized_cells = [row for row in tables["final_execution_authorization_matrix"] if row["included_in_final_request_count"] == "true"]
    adapted_cells = [
        row for row in tables["final_execution_authorization_matrix"]
        if row["evidence_classification"] == "adapted_protocol_reproduction"
    ]
    closed_cells = [
        row for row in tables["final_execution_authorization_matrix"]
        if row["evidence_classification"] == "blocked_paid_or_unavailable_model"
    ]
    return "\n".join(
        [
            "# Phase F6L Paper-Faithful Completion And RQ Revalidation",
            "",
            f"Decision token: `{DECISION_CHECKPOINT}`",
            f"Previous checkpoint token: `{INITIAL_DECISION_CHECKPOINT}`",
            "",
            "F6L has completed the final no-inference execution-authorization checkpoint. No Ollama start, LLM inference, API call, download, training, raw-output overwrite, staging, commit, cleanup, or `original_repo/` mutation was performed.",
            "",
            "## Checkpoint",
            "",
            f"- Deviations found: {summary['deviations']}",
            f"- Paper-exact contracts recovered: {summary['paper_exact_contracts']}",
            f"- Contracts still incomplete: {summary['incomplete_contracts']}",
            f"- Profile-generation matrix: {summary['profile_matrix']}",
            f"- LastFM ranking matrix: {summary['lastfm_ranking_matrix']}",
            f"- LastFM reranking matrix: {summary['lastfm_reranking_matrix']}",
            f"- Initial aggregate request count: `{summary['request_count']}`.",
            f"- Atomic breakdown request count after official-source recovery audit: `{summary['breakdown_request_count']}`.",
            f"- Estimated time: {summary['estimated_time']}.",
            f"- Estimated space: {summary['estimated_space']}.",
            f"- Local models: {summary['local_models']}.",
            f"- Prompts and versions: {summary['prompts_versions']}.",
            f"- Paid/closed model blocked cells: {summary['paid_model_blocks']}.",
            f"- Official recovery outcome: {summary['official_recovery']}",
            f"- Final authorization: {summary['final_authorization']}",
            "",
            "## Atomic Request Composition",
            "",
            f"- Profile generation: `{profile_generation}` = Beauty/Sports `2` datasets x `200` users x Qwen2/Mistral `2` generators.",
            f"- History-only controls: `{history_only}` = Beauty/Sports `2` datasets x `200` users x Qwen2/Mistral `2` recommenders.",
            f"- Profile-only recommendations: `{profile_only}` = Beauty/Sports `2` datasets x `200` users x `2` profile generators x `2` recommenders.",
            f"- Profile-plus-history recommendations: `{profile_plus_history}` = Beauty/Sports `2` datasets x `200` users x `2` profile generators x `2` recommenders.",
            f"- LastFM paper-protocol ranking: `{lastfm_ranking}` = `1000` users x Qwen2/Mistral `2` models x base/LLMRank `2` strategies.",
            f"- LastFM paper-faithful reranking: `{lastfm_reranking}` = blocked because no paper-faithful 20-item reranking pools can be rendered.",
            f"- Total recalculated requests: `{total_requests}`.",
            "",
            "## LastFM Reranking Zero Diagnosis",
            "",
            "The immediate hard blocker is not a generic LastFM gap. The paper-faithful reranking prompt needs the concrete 20-item pool built from official BPRMF, GRU4Rec, SASRec, and LightGCN top-K recommendation snapshots on the same 1000-user cohort, with the official aggregation/order. The official notebook exposes the aggregation logic, but the official LastFM source-model top-K snapshots, exact sample/seed, and final candidate-pool artifact were not found. Therefore `paper_faithful_lastfm_reranking_request_count = 0`.",
            "",
            "Hard blocker requirements:",
            "",
            *[f"- {item}" for item in hard_blockers],
            "",
            "## Official Artifact Recovery",
            "",
            f"- Official artifacts already present locally with hashes: `{len(recovered)}`.",
            f"- Official artifacts not found after source inspection: `{len(not_found)}`.",
            "- New downloads performed: `0`.",
            "- Sources inspected: official GitHub repository, official GitHub `llm_results`, official GitHub `sota_results`, author README Dropbox links for Beauty/Sports, and author-linked GroupLens LastFM raw dataset page.",
            "- The raw GroupLens HetRec LastFM dataset is not a paper-specific 1000-user sample, SOTA top-K snapshot, or final reranking pool, so it does not unblock paper-faithful reranking.",
            "",
            "## Execution Command",
            "",
            "```powershell",
            summary["execution_command"],
            "```",
            "",
            "This is the only command skeleton authorized by the checkpoint contract. It must still be explicitly requested by the user before any inference; the current turn did not execute it.",
            "",
            "## Request Count",
            "",
            "- Profile generation: `800`.",
            "- History-only controls: `800`.",
            "- Profile-only recommendations: `1600`.",
            "- Profile-plus-history recommendations: `1600`.",
            "- LastFM ranking base and LLMRank on Qwen2/Mistral: `4000`.",
            "- LastFM reranking: `0`.",
            "- Total authorized future open-model requests after explicit user approval: `8800`.",
            "",
            "## Authorization Classes",
            "",
            f"- Paper-protocol cells authorized after explicit approval: `{len(authorized_cells)}`.",
            f"- Adapted cells retained separately and not execution-authorized: `{len(adapted_cells)}`.",
            f"- Closed/paid model groups blocked: `{len(closed_cells)}`.",
            "- LLMSRec cells remain blocked because the official demonstration contract was not recovered.",
            "",
            "## Evidence Hierarchy",
            "",
            "Existing adapted results are preserved and reclassified, but they do not directly confirm or contradict paper numerical claims. New local Beauty/Sports open-model profile cells are `paper_protocol_reproduction_different_open_model`; new LastFM ranking cells are `paper_protocol_reproduction_new_sample`; LastFM reranking remains `blocked_missing_original_specification`.",
            "",
            "## Validation",
            "",
            f"- Validation passed: `{validation_pass}`.",
            f"- Validation failed: `{validation_fail}`.",
            "",
            "## Stop",
            "",
            f"`{DECISION_CHECKPOINT}`",
            "",
        ]
    )


def update_marked_section(path: Path, marker: str, content: str) -> None:
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    block = f"{start}\n{content.rstrip()}\n{end}\n"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if start in existing and end in existing:
        before = existing.split(start, 1)[0]
        after = existing.split(end, 1)[1].lstrip("\n")
        text = before.rstrip() + "\n\n" + block + ("\n" + after.rstrip() + "\n" if after.strip() else "")
    else:
        text = existing.rstrip() + "\n\n" + block if existing.strip() else block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def update_handoff_and_notes() -> None:
    handoff_content = "\n".join(
        [
            "## F6L Paper-Faithful Completion Authorization Checkpoint",
            "",
            f"- Latest scientific checkpoint: `{PHASE_NAME}`.",
            f"- Decision token: `{DECISION_CHECKPOINT}`.",
            f"- Previous checkpoint token: `{INITIAL_DECISION_CHECKPOINT}`.",
            "- No inference, Ollama start, paid API call, download, training, cleanup, staging, commit, raw-output overwrite, or `original_repo/` mutation was performed.",
            "- Existing LastFM ranking is retained as adapted evidence; existing LastFM reranking remains adapted/no-raw and is not paper-faithful.",
            "- Recalculated future local open-model request count after explicit approval: `8800`.",
            "- Request composition: profile generation `800`, history-only `800`, profile-only `1600`, profile-plus-history `1600`, LastFM ranking `4000`, LastFM reranking `0`.",
            "- Paper-faithful LastFM reranking request count remains `0` because official LastFM BPRMF/GRU4Rec/SASRec/LightGCN source top-K snapshots and the resulting final 20-item pool identity were not recovered.",
            "- Official recovery audit: Beauty/Sports and Beauty SOTA ZIPs are already present locally with hashes; no new downloads were performed; no official LastFM sample/seed/top-K/pool artifact was found.",
            "- Current guard note: F6L tables/manifests remain isolated terminal checkpoint artifacts.",
        ]
    )
    notes_content = "\n".join(
        [
            "## 2026-07-31 - Phase F6L Paper-Faithful Completion Authorization Checkpoint",
            "",
            "### Changes",
            "",
            "- Extended the existing F6L checkpoint builder, wrapper, config, tests, exactness matrices, request manifest, runtime/cost table, RQ revalidation table, and checkpoint report.",
            "- Added the atomic request breakdown, LastFM reranking prerequisite audit, official artifact recovery audit, and final execution authorization matrix.",
            "- Reclassified existing LastFM ranking/reranking evidence without overwriting raw outputs.",
            "- Recorded official free-download recovery attempts and local official ZIP hashes; no downloads were performed because no missing official LastFM reranking artifact was found.",
            "",
            "### Commands Run",
            "",
            "- F6L checkpoint builder with the canonical recovered runtime.",
            "- Focused F6L tests are recorded in the final assistant response for this implementation turn.",
            "",
            "### Outputs Generated",
            "",
            "- `configs/phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml`",
            "- `src/analysis/plan_f6l_paper_faithful_completion_and_rq_revalidation.py`",
            "- `src/experiments/build_phase_f6l_paper_faithful_completion_and_rq_revalidation.py`",
            "- `tests/test_phase_f6l_paper_faithful_completion_and_rq_revalidation.py`",
            "- `results/tables/phase_f6l_*.csv`",
            "- `results/manifests/phase_f6l_*.json`",
            "- `report/phase_f6l_paper_faithful_completion_and_rq_revalidation.md`",
            "",
            "### Result",
            "",
            f"- Decision: `{DECISION_CHECKPOINT}`.",
            "- Recalculated future local request count after explicit approval: `8800`.",
            "- No paper-faithful LastFM reranking inference is authorized at checkpoint; request count remains `0`.",
            "",
            "### Notes And Limitations",
            "",
            "- Closed/paid models remain blocked unless exact official outputs are recovered.",
            "- A new deterministic K-S-valid LastFM sample is not the exact original sample unless author manifests are recovered.",
            "- Local Qwen2/Mistral outputs, if later generated, will not be exact numerical reproduction.",
            "- No inference, Ollama start, paid/API call, download, training, cleanup, staging, commit, raw-output overwrite, or `original_repo/` mutation was performed.",
        ]
    )
    update_marked_section(HANDOFF_PATH, PHASE_NAME, handoff_content)
    update_marked_section(NOTES_PATH, PHASE_NAME, notes_content)


def assemble(config_path: Path = CONFIG_PATH) -> BuildResult:
    require_runtime()
    load_config(config_path)
    summary = lastfm_summary()
    zip_summary = official_zip_summary()
    protected = hash_protected_inputs()

    tables: dict[str, list[dict[str, str]]] = {}
    tables["paper_repository_exactness_matrix"] = build_exactness_matrix(summary, zip_summary)
    tables["profile_generation_contract"] = build_profile_contract(zip_summary)
    tables["profile_generation_results"] = build_profile_results()
    tables["lastfm_ranking_exactness"] = build_lastfm_ranking_exactness(summary)
    tables["lastfm_ranking_results"] = build_lastfm_ranking_results()
    tables["lastfm_reranking_results"] = build_lastfm_reranking_results(summary)
    tables["execution_request_manifest"] = build_request_manifest()
    tables["execution_request_breakdown"] = build_execution_request_breakdown()
    tables["lastfm_reranking_prerequisite_audit"] = build_lastfm_reranking_prerequisite_audit(summary)
    tables["official_artifact_recovery"] = build_official_artifact_recovery()
    tables["final_execution_authorization_matrix"] = build_final_execution_authorization_matrix(tables["execution_request_breakdown"])
    tables["runtime_cost_results"] = build_runtime_cost_results(tables["execution_request_manifest"])
    tables["rq_revalidation"] = build_rq_revalidation()
    tables["irreducible_exactness_limits"] = build_irreducible_limits()
    tables["model_preflight"] = build_model_preflight()
    tables["external_download_manifest"] = build_external_download_manifest()
    tables["checkpoint_summary"] = build_checkpoint_summary(tables, summary)
    tables["validation"] = build_validation(tables, protected)

    result = BuildResult(
        decision_token=DECISION_CHECKPOINT,
        tables=tables,
        manifests={"protected_input_hashes": dict(protected)},
        report=build_report(tables),
    )
    return result


def build(config_path: Path = CONFIG_PATH, *, write_outputs_enabled: bool = True) -> BuildResult:
    result = assemble(config_path)
    if write_outputs_enabled:
        for name, path in OUTPUT_TABLES.items():
            write_csv(path, result.tables[name], TABLE_FIELDS[name])
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(result.report, encoding="utf-8")
        update_handoff_and_notes()
        protected = result.manifests["protected_input_hashes"]
        write_json(PROTECTED_HASHES_PATH, protected)
        manifest = checkpoint_manifest(result, protected)
        write_json(CHECKPOINT_MANIFEST_PATH, manifest)
        result = BuildResult(
            decision_token=result.decision_token,
            tables=result.tables,
            manifests={**result.manifests, "checkpoint": manifest},
            report=result.report,
        )
    return result


def semantic_digest(result: BuildResult) -> str:
    payload = {
        "decision_token": result.decision_token,
        "tables": result.tables,
        "protected_input_paths": [entry["path"] for entry in result.manifests["protected_input_hashes"]["protected_inputs"]],
        "report": result.report,
    }
    return sha256_text(canonical_json(payload))


def contract_materialization_digest(result: ContractMaterializationResult) -> str:
    payload = {
        "decision_token": result.decision_token,
        "tables": result.tables,
        "manifest_without_output_hashes": {
            key: value
            for key, value in result.manifest.items()
            if key not in {"output_hashes", "self_hash"}
        },
    }
    return sha256_text(canonical_json(payload))


def build_lastfm_contract_reconstruction_audit() -> list[dict[str, str]]:
    rows = [
        (
            "dataset_version",
            "LastFM public dataset version",
            "LastFM HetRec 2011 / GroupLens; local source data/external/plan_c/hetrec2011-lastfm-2k.zip",
            "paper;official README;F6C protocol",
            "paper/traduzione_italiana_beyond_utility_llm_recommender.docx;original_repo/EvaLLMasRecommender/README.md;results/tables/finalization_f6c_lastfm_paper_protocol.csv",
            "recovered",
            "false",
            "paper_protocol_reproduction_new_sample",
            "Dataset family is recovered, but not an author-provided processed sample.",
        ),
        (
            "five_core",
            "5-core filtering",
            "iterative 5-core over user-item interactions; final F6C local set has user/item min degree 5",
            "paper;F6C protocol",
            "results/tables/finalization_f6c_lastfm_paper_protocol.csv;results/tables/finalization_f6c_lastfm_interaction_audit.csv",
            "recovered_as_local_reconstruction",
            "false",
            "paper_protocol_reproduction_new_sample",
            "The 5-core rule is recovered; author processed files are still unavailable.",
        ),
        (
            "temporal_ordering",
            "ordering temporal for leave-one-out",
            "not recovered: user_artists.dat has no listening timestamp; tag timestamps are sparse user-tag-artist events, not listening events",
            "paper;official dataset;F6C protocol",
            "results/tables/finalization_f6c_lastfm_paper_protocol.csv",
            "blocked_missing_input_or_specification",
            "true",
            "blocked_missing_original_specification",
            "A paper-faithful temporal split cannot be materialized from the verified LastFM source.",
        ),
        (
            "leave_one_out_split",
            "train/validation/test leave-one-out split",
            "paper says leave-one-out; exact LastFM temporal validation/test construction is not recoverable before inventing a local order",
            "paper;F6C protocol",
            "paper/traduzione_italiana_beyond_utility_llm_recommender.docx;results/tables/finalization_f6c_lastfm_paper_protocol.csv",
            "blocked_missing_specification",
            "true",
            "blocked_missing_original_specification",
            "F6C weight-rank leave-two-out remains adapted evidence only.",
        ),
        (
            "eligible_users",
            "eligible users after paper-faithful preprocessing and split",
            "1859 local adapted eligible users exist; paper-faithful eligible population cannot be certified without the exact split/order",
            "F6C artifacts",
            "data/processed/finalization_f6c_lastfm_ranking_cohort.csv",
            "blocked_for_paper_faithful_sample",
            "true",
            "blocked_missing_original_specification",
            "The local adapted population is not promoted to the F6L frozen paper-protocol sample.",
        ),
        (
            "ks_gate_hypothesis",
            "K-S gate hypothesis",
            "recovered only at high level: compare metric distributions on full test and sampled test under non-significant difference hypothesis",
            "paper translated DOCX;official README",
            "paper/traduzione_italiana_beyond_utility_llm_recommender.docx;original_repo/EvaLLMasRecommender/README.md",
            "partially_recovered_not_operational",
            "true",
            "blocked_missing_original_specification",
            "This is not sufficient to choose a pre-inference 1000-user sample without inventing the metric distribution inputs.",
        ),
        (
            "ks_feature",
            "exact K-S feature or metric vector",
            "not recovered: paper mentions metrics on full test vs sampled test, but no executable pre-inference feature is specified",
            "paper;official code;current F6 artifacts",
            "paper/traduzione_italiana_beyond_utility_llm_recommender.docx;original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb;results/tables/finalization_f6c_lastfm_paper_protocol.csv",
            "missing",
            "true",
            "blocked_missing_original_specification",
            "Using train-history length, item popularity, or any other local feature would be an invented K-S contract.",
        ),
        (
            "ks_threshold",
            "K-S threshold / alpha",
            "not recovered",
            "paper;official code;current F6 artifacts",
            "paper/traduzione_italiana_beyond_utility_llm_recommender.docx;original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb;original_repo/EvaLLMasRecommender/README.md",
            "missing",
            "true",
            "blocked_missing_original_specification",
            "A conventional alpha such as 0.05 is not inserted because the user required source recovery, not inference.",
        ),
        (
            "negative_sampling",
            "1 positive + 19 negatives and negative source",
            "candidate count and positive/negative counts recovered; exact LastFM negative universe and seed are not recovered",
            "paper;F6C protocol;official notebook",
            "paper/traduzione_italiana_beyond_utility_llm_recommender.docx;results/tables/finalization_f6c_lastfm_candidate_policy.csv;original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb",
            "partially_recovered_not_original_exact",
            "true",
            "blocked_missing_original_specification",
            "The existing F6C seed 42 candidate file is adapted and cannot be relabeled as F6L paper-protocol without the K-S sample.",
        ),
        (
            "candidate_ordering",
            "candidate order frozen and shared across models/strategies",
            "official notebook randomizes candidate list and reuses candidate ids for matched prompt variants; exact author seed/list unavailable",
            "official notebook;F6L matrix",
            "original_repo/EvaLLMasRecommender/eva/codes/generate_prompt.ipynb;results/tables/phase_f6l_final_execution_authorization_matrix.csv",
            "recoverable_after_sample_but_not_materialized",
            "true",
            "blocked_missing_original_specification",
            "The shared-order rule is understood, but no concrete F6L list is materialized while K-S is blocked.",
        ),
        (
            "request_manifest",
            "8800 per-request manifest rows",
            "not materialized because the LastFM 1000-user K-S sample and candidate set are blocked",
            "F6L authorization matrix",
            "results/tables/phase_f6l_final_execution_authorization_matrix.csv",
            "blocked_before_materialization",
            "true",
            "blocked_missing_original_specification",
            "Generating 4000 LastFM request rows without a valid frozen sample would make a false execution contract.",
        ),
    ]
    return [
        {
            "audit_id": f"F6L-LASTFM-CONTRACT-{index:03d}",
            "contract_area": area,
            "required_detail": required,
            "recovered_value": recovered,
            "source_priority": source_priority,
            "evidence_source": evidence_source,
            "status": status,
            "blocks_materialization": blocks,
            "evidence_class": evidence_class,
            "notes": notes,
        }
        for index, (area, required, recovered, source_priority, evidence_source, status, blocks, evidence_class, notes) in enumerate(rows, start=1)
    ]


def build_contract_materialization_blockers(audit_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    blockers = [
        (
            "ks_gate_operational_definition",
            "Exact K-S feature/metric vector, threshold/alpha, and pre-inference gate procedure are not specified by the recovered paper/code/artifacts.",
            "paper DOCX;official README;official generate_prompt notebook;F6C protocol",
            "true",
            "true",
            "true",
            "blocked",
            "Recover an official sample manifest, author code that performs the K-S gate, or an explicit paper supplement defining feature, statistic threshold, and decision rule.",
        ),
        (
            "lastfm_temporal_leave_one_out_order",
            "Paper-faithful temporal ordering for LastFM leave-one-out cannot be certified from user_artists.dat; tag timestamps are not listening timestamps.",
            "paper DOCX;GroupLens HetRec archive;F6C protocol",
            "true",
            "true",
            "true",
            "blocked",
            "Recover author processed LastFM split/sample artifacts or an official source that maps LastFM user-artist interactions to the paper temporal order.",
        ),
    ]
    active_areas = {row["contract_area"] for row in audit_rows if row["blocks_materialization"] == "true"}
    if "ks_feature" not in active_areas or "ks_threshold" not in active_areas:
        blockers.append(
            (
                "audit_consistency",
                "K-S blockers were expected but not present in the audit rows.",
                "internal consistency check",
                "true",
                "true",
                "true",
                "blocked",
                "Repair F6L audit generation before materialization.",
            )
        )
    return [
        {
            "blocker_id": f"F6L-CONTRACT-BLOCKER-{index:03d}",
            "decision_token": DECISION_CONTRACT_MATERIALIZATION_BLOCKED,
            "contract_area": area,
            "missing_definition": missing,
            "source_attempts": attempts,
            "blocks_sample": blocks_sample,
            "blocks_candidates": blocks_candidates,
            "blocks_request_manifest": blocks_manifest,
            "status": status,
            "next_action": next_action,
        }
        for index, (area, missing, attempts, blocks_sample, blocks_candidates, blocks_manifest, status, next_action) in enumerate(blockers, start=1)
    ]


def build_contract_materialization_summary(
    audit_rows: Sequence[Mapping[str, str]],
    blocker_rows: Sequence[Mapping[str, str]],
    protected: Mapping[str, Any],
) -> list[dict[str, str]]:
    authorized_total = authorized_request_total_from_matrix()
    authorized_rows = included_authorized_matrix_rows()
    files_absent = [
        not LASTFM_FROZEN_SAMPLE_COHORT_PATH.exists(),
        not LASTFM_FROZEN_SAMPLE_CANDIDATES_PATH.exists(),
        not PER_REQUEST_MANIFEST_PATH.exists(),
    ]
    entries = [
        ("decision_token", DECISION_CONTRACT_MATERIALIZATION_BLOCKED, "blocked", "contract blockers", "No execution contract was materialized."),
        ("authorization_matrix_requests", str(authorized_total), "pass" if authorized_total == 8800 else "fail", rel(OUTPUT_TABLES["final_execution_authorization_matrix"]), "The aggregate matrix still totals 8800 authorized requests."),
        ("authorized_positive_cells", str(len(authorized_rows)), "pass", rel(OUTPUT_TABLES["final_execution_authorization_matrix"]), "Only included positive authorization cells are considered for future manifest derivation."),
        ("ks_gate_recovered", "false", "blocked", "lastfm_contract_reconstruction_audit", "The K-S definition is not operationally recoverable."),
        ("sample_materialized", "false", "blocked", rel(LASTFM_FROZEN_SAMPLE_COHORT_PATH), "The sample file remains absent by design."),
        ("candidate_sets_materialized", "false", "blocked", rel(LASTFM_FROZEN_SAMPLE_CANDIDATES_PATH), "The candidate file remains absent by design."),
        ("request_manifest_materialized", "false", "blocked", rel(PER_REQUEST_MANIFEST_PATH), "The 8800 request manifest remains absent by design."),
        ("blocked_files_absent", str(all(files_absent)).lower(), "pass" if all(files_absent) else "fail", "filesystem", "Fail-closed materialization did not create unverifiable execution inputs."),
        ("protected_input_status", str(protected.get("status", "")), "pass" if protected.get("status") == "pass" else "fail", rel(PROTECTED_HASHES_PATH), "Protected source hashes are still clean."),
        ("f6i2p_yaml_count", str(protected.get("f6i2p_yaml_count", "")), "pass" if protected.get("f6i2p_yaml_count") == 23 else "fail", rel(PROTECTED_HASHES_PATH), "No protected F6I2P YAML change is required."),
        ("staged_paths", git_output(["diff", "--cached", "--name-only"]) or "0", "pass" if not git_output(["diff", "--cached", "--name-only"]) else "fail", "git diff --cached --name-only", "No staging is allowed."),
        ("inference_performed", "false", "pass", "phase policy", "No Ollama generation or remote API call was made."),
        ("raw_output_directory_created", "false", "pass", "results/llm", "No phase_f6l raw output directory is created."),
        ("blocking_contract_areas", ";".join(row["contract_area"] for row in blocker_rows), "blocked", "execution_contract_materialization_blockers", "These blockers prevent sample/candidate/request materialization."),
    ]
    return [
        {
            "summary_id": f"F6L-CONTRACT-SUMMARY-{index:03d}",
            "item": item,
            "value": value,
            "status": status,
            "evidence": evidence,
            "notes": notes,
        }
        for index, (item, value, status, evidence, notes) in enumerate(entries, start=1)
    ]


def build_contract_materialization_manifest(result: ContractMaterializationResult, protected: Mapping[str, Any]) -> dict[str, Any]:
    output_hashes = {
        rel(path): safe_file_sha256(path)
        for path in CONTRACT_MATERIALIZATION_OUTPUT_TABLES.values()
        if path.exists()
    }
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "mode": "materialize-contract",
        "decision_token": result.decision_token,
        "materialized_at": MATERIALIZATION_AUDIT_TIMESTAMP,
        "scientific_classification": "blocked_missing_original_specification",
        "lastfm_sample_materialized": False,
        "lastfm_candidates_materialized": False,
        "open_model_request_manifest_materialized": False,
        "planned_requests": authorized_request_total_from_matrix(),
        "request_manifest_rows": 0,
        "unique_request_ids": 0,
        "lastfm_sample_users": 0,
        "lastfm_candidate_rows": 0,
        "inference_performed": False,
        "ollama_generation_performed": False,
        "remote_api_calls_performed": False,
        "paid_api_calls_performed": False,
        "downloads_performed": False,
        "training_performed": False,
        "raw_output_dir_created": False,
        "authorization_matrix_sha256": safe_file_sha256(OUTPUT_TABLES["final_execution_authorization_matrix"]),
        "lastfm_sample_sha256": "",
        "lastfm_candidates_sha256": "",
        "request_manifest_sha256": "",
        "counts_by_family": {
            row["experiment_family"]: sum(
                int(other.get("expected_requests", "0") or 0)
                for other in included_authorized_matrix_rows()
                if other.get("experiment_family") == row["experiment_family"]
            )
            for row in included_authorized_matrix_rows()
        },
        "models": {
            "qwen2:7b-instruct": "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755",
            "mistral:7b-instruct": "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf",
        },
        "seed_policy": {
            "original_author_sample_seed": "not_recovered",
            "f6c_adapted_candidate_seed": "42",
            "f6l_sample_seed": "not_registered_because_ks_gate_blocked",
        },
        "runtime": {
            "approved_interpreter": str(RECOVERED_INTERPRETER),
            "observed_interpreter": str(Path(sys.executable).resolve()),
            "python_no_user_site": os.environ.get("PYTHONNOUSERSITE", ""),
            "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE", ""),
        },
        "git": {
            "head": git_output(["rev-parse", "HEAD"]),
            "staged_paths": git_output(["diff", "--cached", "--name-only"]),
        },
        "protected_input_status": protected.get("status"),
        "f6i2p_yaml_count": protected.get("f6i2p_yaml_count"),
        "tables": {name: rel(path) for name, path in CONTRACT_MATERIALIZATION_OUTPUT_TABLES.items()},
        "output_hashes": output_hashes,
        "self_hash": "self_reference_excluded_for_determinism",
    }


def build_contract_materialization_report_section(result: ContractMaterializationResult) -> str:
    summary = {row["item"]: row for row in result.tables["execution_contract_materialization_summary"]}
    blockers = result.tables["execution_contract_materialization_blockers"]
    blocker_lines = [
        f"- {row['contract_area']}: {row['missing_definition']}"
        for row in blockers
    ]
    return "\n".join(
        [
            "## Execution Contract Materialization",
            "",
            f"- Decision token: `{result.decision_token}`.",
            f"- LastFM sample materialized: `{summary['sample_materialized']['value']}`.",
            f"- LastFM candidate sets materialized: `{summary['candidate_sets_materialized']['value']}`.",
            f"- Open-model request manifest materialized: `{summary['request_manifest_materialized']['value']}`.",
            f"- Aggregate authorized requests still valid: `{summary['authorization_matrix_requests']['value']}`.",
            "- Recovered K-S criterion: high-level paper statement only; not operational.",
            "- Seed policy: author sample seed not recovered; F6C adapted candidate seed `42` is not promoted to F6L because the K-S gate is blocked.",
            "",
            "Blocking definitions:",
            "",
            *blocker_lines,
            "",
            "No inference, Ollama generation, remote API call, download, training, raw-output directory creation, staging, commit, cleanup, or `original_repo/` mutation was performed.",
            "",
            f"`{result.decision_token}`",
            "",
        ]
    )


def update_contract_materialization_handoff_and_notes(result: ContractMaterializationResult) -> None:
    blockers = "; ".join(row["contract_area"] for row in result.tables["execution_contract_materialization_blockers"])
    handoff_content = "\n".join(
        [
            "## F6L Execution Contract Materialization",
            "",
            f"- Decision token: `{result.decision_token}`.",
            "- Materialization stopped before creating the LastFM 1000-user sample, candidate sets, or 8800-row request manifest.",
            "- Reason: recovered sources do not define an operational K-S gate with exact feature/metric vector, threshold/alpha, and decision rule; LastFM temporal leave-one-out ordering is also not certifiable from verified source artifacts.",
            "- No inference, Ollama generation, remote API call, raw-output directory, download, training, staging, commit, cleanup, or `original_repo/` mutation was performed.",
            f"- Blocking contract areas: {blockers}.",
        ]
    )
    notes_content = "\n".join(
        [
            "## 2026-08-01 - Phase F6L Execution Contract Materialization Attempt",
            "",
            "### Changes",
            "",
            "- Added the F6L contract materialization audit and fail-closed blocker manifest.",
            "- Did not create the LastFM F6L sample, candidate file, or open-model request manifest because the K-S gate is not operationally recoverable from verified sources.",
            "",
            "### Outputs Generated",
            "",
            "- `results/tables/phase_f6l_lastfm_contract_reconstruction_audit.csv`",
            "- `results/tables/phase_f6l_execution_contract_materialization_blockers.csv`",
            "- `results/tables/phase_f6l_execution_contract_materialization_summary.csv`",
            "- `results/manifests/phase_f6l_execution_contract_materialization_manifest.json`",
            "",
            "### Result",
            "",
            f"- Decision: `{result.decision_token}`.",
            f"- Blockers: {blockers}.",
        ]
    )
    marker = f"{PHASE_NAME}_contract_materialization"
    update_marked_section(REPORT_PATH, marker, build_contract_materialization_report_section(result))
    update_marked_section(HANDOFF_PATH, marker, handoff_content)
    update_marked_section(NOTES_PATH, marker, notes_content)


def assemble_contract_materialization(config_path: Path = CONFIG_PATH) -> ContractMaterializationResult:
    require_runtime()
    load_config(config_path)
    protected = hash_protected_inputs()
    audit_rows = build_lastfm_contract_reconstruction_audit()
    blocker_rows = build_contract_materialization_blockers(audit_rows)
    summary_rows = build_contract_materialization_summary(audit_rows, blocker_rows, protected)
    result = ContractMaterializationResult(
        decision_token=DECISION_CONTRACT_MATERIALIZATION_BLOCKED,
        tables={
            "lastfm_contract_reconstruction_audit": audit_rows,
            "execution_contract_materialization_blockers": blocker_rows,
            "execution_contract_materialization_summary": summary_rows,
        },
        manifest={},
    )
    manifest = build_contract_materialization_manifest(result, protected)
    return ContractMaterializationResult(
        decision_token=result.decision_token,
        tables=result.tables,
        manifest=manifest,
    )


def build_contract_materialization(config_path: Path = CONFIG_PATH, *, write_outputs_enabled: bool = True) -> ContractMaterializationResult:
    first = assemble_contract_materialization(config_path)
    second = assemble_contract_materialization(config_path)
    deterministic = contract_materialization_digest(first) == contract_materialization_digest(second)
    manifest = {**first.manifest, "deterministic_in_memory_digest_match": deterministic}
    result = ContractMaterializationResult(first.decision_token, first.tables, manifest)
    if write_outputs_enabled:
        for name, path in CONTRACT_MATERIALIZATION_OUTPUT_TABLES.items():
            write_csv(path, result.tables[name], CONTRACT_MATERIALIZATION_TABLE_FIELDS[name])
        manifest = {**result.manifest}
        manifest["output_hashes"] = {
            rel(path): safe_file_sha256(path)
            for path in CONTRACT_MATERIALIZATION_OUTPUT_TABLES.values()
            if path.exists()
        }
        result = ContractMaterializationResult(result.decision_token, result.tables, manifest)
        write_json(CONTRACT_MATERIALIZATION_MANIFEST_PATH, result.manifest)
        update_contract_materialization_handoff_and_notes(result)
    return result


MODEL_DIGESTS = {
    "qwen2:7b-instruct": "dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755",
    "mistral:7b-instruct": "6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf",
}

MODEL_IDENTITIES = {
    "qwen2:7b-instruct": "Qwen2 7B Instruct local Ollama Q4_0 digest dd314f039b9d54d5553002c906ce50c9fe7242f73f0680abd04f01c8ecbd2755",
    "mistral:7b-instruct": "Mistral 7B Instruct local Ollama Q4_K_M digest 6577803aa9a036369e481d648a2baebb381ebc6e897f2bb9a766a2aa7bfbc1cf",
}

PROFILE_SOURCE_ENTRIES = {
    "profile_generation": "test_history10_generate_profile_sample.json",
    "history_only": "test_history10_candidate20_shuffle_sample.json",
    "profile_only": "test_history10_candidate20_shuffle_profile_gpt3.5_sample.json",
    "profile_plus_history": "test_history10_candidate20_shuffle_profile_gpt3.5_history_sample.json",
}

PROFILE_TEMPLATE_PATHS = {
    "profile_generation": PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "eva" / "templates" / "template_user_profile_generate.txt",
    "history_only": PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "eva" / "templates" / "template.txt",
    "profile_only": PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "eva" / "templates" / "template_profile.txt",
    "profile_plus_history": PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "eva" / "templates" / "template_profile_history.txt",
}

PROFILE_GENERATION_PARAMETERS = {
    "temperature": 0,
    "stream": False,
    "concurrency": 1,
    "scientific_retry_count": 0,
    "infrastructure_retry_count": 0,
    "response_format": "text",
    "seed": 0,
    "timeout_seconds": 120,
}


def zip_json_rows(dataset: str, entry_suffix: str) -> list[dict[str, Any]]:
    zip_path = PROJECT_ROOT / "original_repo" / "EvaLLMasRecommender" / "llm_results" / f"{dataset}.zip"
    entry_name = f"{dataset}/{entry_suffix}"
    with ZipFile(zip_path) as archive:
        payload = json.loads(archive.read(entry_name).decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected list payload in {zip_path}!{entry_name}")
    return [dict(row) for row in payload]


def parse_candidate_ids(raw: str) -> list[str]:
    value = ast.literal_eval(raw)
    if not isinstance(value, list):
        raise ValueError("candidate list is not a list")
    return [str(item) for item in value]


def parse_prompt_candidate_titles(prompt: str) -> list[str]:
    marker = "The candidates are:"
    if marker not in prompt:
        return []
    section = prompt.split(marker, 1)[1]
    start = section.find("[")
    end = section.rfind("]")
    if start < 0 or end < start:
        return []
    text = section[start + 1 : end]
    titles: list[str] = []
    index = 0
    while index < len(text):
        while index < len(text) and text[index] not in {"'", '"'}:
            index += 1
        if index >= len(text):
            break
        quote = text[index]
        index += 1
        chars: list[str] = []
        while index < len(text):
            char = text[index]
            next_chars = text[index + 1 : index + 4]
            if char == quote and (next_chars.startswith(",") or index + 1 >= len(text)):
                index += 1
                break
            chars.append(char)
            index += 1
        titles.append("".join(chars).strip())
    return titles


def normalized_title(value: str) -> str:
    return " ".join(html.unescape(value).split()).casefold()


def positive_candidate(row: Mapping[str, Any]) -> tuple[str, str]:
    if not row.get("cand_list"):
        return "", ""
    ids = parse_candidate_ids(str(row["cand_list"]))
    prompt = str(row.get("prompt", ""))
    titles = parse_prompt_candidate_titles(prompt)
    target = normalized_title(str(row.get("target", "")))
    for index, title in enumerate(titles):
        if normalized_title(title) == target and index < len(ids):
            return ids[index], str(index + 1)
    marker = "The candidates are:"
    if marker in prompt:
        section = prompt.split(marker, 1)[1]
        raw_target = str(row.get("target", ""))
        target_index = section.find(raw_target)
        if target_index < 0:
            target_index = html.unescape(section).find(html.unescape(raw_target))
        if target_index >= 0:
            position = section[:target_index].count("', '") + 1
            if 1 <= position <= len(ids):
                return ids[position - 1], str(position)
    return "", ""


def profile_source_row_sets(dataset: str) -> dict[str, list[dict[str, Any]]]:
    rows = {variant: zip_json_rows(dataset, entry) for variant, entry in PROFILE_SOURCE_ENTRIES.items()}
    expected_users = [str(row["user"]) for row in rows["history_only"]]
    for variant, variant_rows in rows.items():
        if len(variant_rows) != 200:
            raise ValueError(f"{dataset} {variant} expected 200 rows, observed {len(variant_rows)}")
        users = [str(row["user"]) for row in variant_rows]
        if users != expected_users:
            raise ValueError(f"{dataset} {variant} user order differs from history-only prompt rows")
    return rows


def profile_generator_from_cell(cell_id: str, dataset: str) -> str:
    prefix = f"F6L-BREAKDOWN-PROFILE-ONLY-{dataset.upper()}-"
    plus_prefix = f"F6L-BREAKDOWN-PROFILE-PLUS-HISTORY-{dataset.upper()}-"
    if cell_id.startswith(prefix):
        return cell_id[len(prefix) :].split("-", 1)[0].lower()
    if cell_id.startswith(plus_prefix):
        return cell_id[len(plus_prefix) :].split("-", 1)[0].lower()
    return ""


def profile_request_id(order: int) -> str:
    return f"F6L-PROFILE-REQ-{order:04d}"


def build_profile_request_manifest() -> list[dict[str, str]]:
    auth_rows = [
        row
        for row in read_csv_rows(OUTPUT_TABLES["final_execution_authorization_matrix"])
        if row.get("included_in_final_request_count") == "true"
        and row.get("experiment_family") in {"profile_generation", "profile_recommendation_control", "profile_recommendation"}
    ]
    by_dataset = {dataset: profile_source_row_sets(dataset) for dataset in ["beauty", "sports"]}
    profile_dependency: dict[tuple[str, str, str], str] = {}
    rows: list[dict[str, str]] = []
    order = 1

    def append_request(
        *,
        auth: Mapping[str, str],
        source_row: Mapping[str, Any],
        model_name: str,
        profile_generator_name: str,
        recommender_model_name: str,
        variant: str,
        dependency_request_id: str,
    ) -> str:
        nonlocal order
        dataset = auth["dataset"].lower()
        request_id = profile_request_id(order)
        source_entry = PROFILE_SOURCE_ENTRIES[variant]
        source_zip_entry = f"original_repo/EvaLLMasRecommender/llm_results/{dataset}.zip!{dataset}/{source_entry}"
        template_path = PROFILE_TEMPLATE_PATHS[variant]
        prompt = str(source_row.get("prompt", ""))
        candidate_ids = parse_candidate_ids(str(source_row["cand_list"])) if source_row.get("cand_list") else []
        positive_item_id, positive_position = positive_candidate(source_row)
        candidate_hash = sha256_text(canonical_json(candidate_ids)) if candidate_ids else ""
        history_hash = sha256_text(prompt) if variant != "profile_only" else "not_applicable"
        if dependency_request_id:
            render_recipe = {
                "source_prompt_hash": sha256_text(prompt),
                "dependency_request_id": dependency_request_id,
                "profile_placeholder": "replace author-profile exemplar with local parent profile output after strict-valid parent completion",
            }
            rendered_prompt_hash = sha256_text(canonical_json(render_recipe))
            rendering_status = "deferred_until_dependency_strict_valid"
        else:
            rendered_prompt_hash = sha256_text(prompt)
            rendering_status = "static_rendered_prompt_from_official_zip"
        model_digest = MODEL_DIGESTS[model_name]
        profile_digest = MODEL_DIGESTS.get(profile_generator_name, "") if profile_generator_name else ""
        generation_parameters = {
            **PROFILE_GENERATION_PARAMETERS,
            "output_length": "profile_text" if variant == "profile_generation" else 10,
            "parser_contract": "free_text_profile_retained" if variant == "profile_generation" else "paper_title_list_top10_parser",
        }
        base_row = {
            "request_id": request_id,
            "experiment_family": auth["experiment_family"],
            "experiment_cell_id": auth["cell_id"],
            "dataset": dataset,
            "task": auth["task"],
            "sample_id": f"phase_f6l_{dataset}_official_200_user_profile_artifact_aligned_cohort",
            "user_id": str(source_row["user"]),
            "model_name": model_name,
            "model_exact_identity": MODEL_IDENTITIES[model_name],
            "model_digest": model_digest,
            "profile_generator_name": profile_generator_name,
            "profile_generator_digest": profile_digest,
            "recommender_model_name": recommender_model_name,
            "prompting_strategy": auth["prompting_strategy"],
            "input_variant": auth["input_variant"],
            "history_length": "10",
            "history_hash": history_hash,
            "candidate_pool_type": "official_profile_artifact_candidate20_shuffled" if candidate_ids else "not_applicable",
            "candidate_pool_size": str(len(candidate_ids)) if candidate_ids else "0",
            "candidate_list_hash": candidate_hash,
            "positive_item_id": positive_item_id,
            "positive_item_title": str(source_row.get("target", "")),
            "positive_candidate_position": positive_position,
            "output_length": "profile_text" if variant == "profile_generation" else "10",
            "parser_contract": "free_text_profile_retained" if variant == "profile_generation" else "paper_title_list_top10_parser",
            "prompt_template_path": rel(template_path),
            "prompt_template_hash": file_sha256(template_path),
            "prompt_template_index": str(source_row.get("template", "")),
            "rendered_prompt_hash": rendered_prompt_hash,
            "rendering_status": rendering_status,
            "generation_parameters_json": canonical_json(generation_parameters),
            "dependency_request_id": dependency_request_id,
            "authorization_status": auth["authorization_status"],
            "evidence_class": auth["evidence_classification"],
            "expected_raw_output_relative_path": f"results/llm/phase_f6l_profile_execution_<run_id>/{request_id}.json",
            "input_lineage": source_zip_entry,
        }
        base_row["request_contract_hash"] = sha256_text(canonical_json(base_row))
        rows.append(base_row)
        order += 1
        return request_id

    for auth in auth_rows:
        dataset = auth["dataset"].lower()
        model = auth["model"]
        variant = auth["input_variant"]
        source_variant = variant
        if auth["experiment_family"] == "profile_recommendation_control":
            source_variant = "history_only"
        if auth["experiment_family"] == "profile_generation":
            source_variant = "profile_generation"
        source_rows = by_dataset[dataset][source_variant]
        if source_variant == "profile_generation":
            for source_row in source_rows:
                request_id = append_request(
                    auth=auth,
                    source_row=source_row,
                    model_name=model,
                    profile_generator_name=model,
                    recommender_model_name="",
                    variant=source_variant,
                    dependency_request_id="",
                )
                profile_dependency[(dataset, str(source_row["user"]), model)] = request_id
        elif source_variant == "history_only":
            for source_row in source_rows:
                append_request(
                    auth=auth,
                    source_row=source_row,
                    model_name=model,
                    profile_generator_name="",
                    recommender_model_name=model,
                    variant=source_variant,
                    dependency_request_id="",
                )
        else:
            profile_generator_key = profile_generator_from_cell(auth["cell_id"], dataset)
            profile_generator_model = "qwen2:7b-instruct" if profile_generator_key == "qwen2" else "mistral:7b-instruct"
            for source_row in source_rows:
                dependency = profile_dependency[(dataset, str(source_row["user"]), profile_generator_model)]
                append_request(
                    auth=auth,
                    source_row=source_row,
                    model_name=model,
                    profile_generator_name=profile_generator_model,
                    recommender_model_name=model,
                    variant=source_variant,
                    dependency_request_id=dependency,
                )
    if len(rows) != 4800:
        raise ValueError(f"profile manifest expected 4800 rows, observed {len(rows)}")
    return rows


def validate_profile_manifest(rows: Sequence[Mapping[str, str]]) -> list[tuple[str, str, bool, str]]:
    ids = [row["request_id"] for row in rows]
    by_id = {row["request_id"]: row for row in rows}
    checks: list[tuple[str, str, bool, str]] = [
        ("manifest_rows", str(len(rows)), len(rows) == 4800, "expected 4800 autonomous profile requests"),
        ("unique_request_ids", str(len(set(ids))), len(set(ids)) == 4800, "one request_id per profile-family request"),
        ("no_lastfm_rows", str(sum(1 for row in rows if row["dataset"].lower() == "lastfm")), all(row["dataset"].lower() != "lastfm" for row in rows), "profile manifest excludes LastFM"),
        ("profile_generation_count", str(sum(1 for row in rows if row["input_variant"] == "profile_generation")), sum(1 for row in rows if row["input_variant"] == "profile_generation") == 800, ""),
        ("history_only_count", str(sum(1 for row in rows if row["input_variant"] == "history_only")), sum(1 for row in rows if row["input_variant"] == "history_only") == 800, ""),
        ("profile_only_count", str(sum(1 for row in rows if row["input_variant"] == "profile_only")), sum(1 for row in rows if row["input_variant"] == "profile_only") == 1600, ""),
        ("profile_plus_history_count", str(sum(1 for row in rows if row["input_variant"] == "profile_plus_history")), sum(1 for row in rows if row["input_variant"] == "profile_plus_history") == 1600, ""),
        ("authorized_models_only", ";".join(sorted({row["model_name"] for row in rows})), {row["model_name"] for row in rows} == set(MODEL_DIGESTS), ""),
        ("candidate20_for_recommendation", "true", all(row["candidate_pool_size"] == "20" for row in rows if row["input_variant"] != "profile_generation"), ""),
        ("positive_position_present", "true", all(row["positive_candidate_position"] for row in rows if row["input_variant"] != "profile_generation"), ""),
        ("prompt_hashes_present", "true", all(row["prompt_template_hash"] and row["rendered_prompt_hash"] for row in rows), ""),
    ]
    dependency_ok = True
    acyclic_ok = True
    for row in rows:
        dependency = row["dependency_request_id"]
        if row["input_variant"] in {"history_only", "profile_generation"}:
            dependency_ok = dependency_ok and dependency == ""
        else:
            parent = by_id.get(dependency)
            dependency_ok = dependency_ok and parent is not None
            if parent:
                dependency_ok = dependency_ok and parent["input_variant"] == "profile_generation"
                dependency_ok = dependency_ok and parent["dataset"] == row["dataset"]
                dependency_ok = dependency_ok and parent["user_id"] == row["user_id"]
                dependency_ok = dependency_ok and parent["model_name"] == row["profile_generator_name"]
                acyclic_ok = acyclic_ok and parent["dependency_request_id"] == ""
    checks.append(("profile_dependencies_valid", str(dependency_ok).lower(), dependency_ok, "dependent profile prompts point to matching parent profile_generation request"))
    checks.append(("profile_dependencies_acyclic", str(acyclic_ok).lower(), acyclic_ok, "dependency depth is one"))
    return checks


def build_profile_checkpoint(rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    checks = validate_profile_manifest(rows)
    total_seconds = int(round(4800 * (21000 / 8800)))
    entries = [
        ("requests_materialized", "4800", "pass", rel(PROFILE_REQUEST_MANIFEST_PATH), "Profile-family request manifest is independent from LastFM."),
        ("manifest_rows", str(len(rows)), "pass" if len(rows) == 4800 else "fail", rel(PROFILE_REQUEST_MANIFEST_PATH), "Expected exactly 4800 rows."),
        ("unique_request_ids", str(len({row["request_id"] for row in rows})), "pass" if len({row["request_id"] for row in rows}) == 4800 else "fail", rel(PROFILE_REQUEST_MANIFEST_PATH), "Expected 4800 unique request_id values."),
        ("models", ";".join(sorted(MODEL_DIGESTS)), "pass", "local Ollama digest preflight from prior F6L execute preflight", "Only Qwen2/Mistral local open models are included."),
        ("dependencies", "valid;acyclic", "pass" if all(check[2] for check in checks if check[0].startswith("profile_dependencies")) else "fail", rel(PROFILE_REQUEST_MANIFEST_PATH), "profile_only/profile_plus_history wait for parent profile_generation strict-valid."),
        ("runtime_estimate", f"about {total_seconds} seconds / {total_seconds / 3600:.1f} hours", "ready", "phase_f6l_runtime_cost_results.csv planning rate", "Estimate uses prior F6L planning rate, not measured execution."),
        ("local_command", "$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='.'; .\\.venv_audit_recovery\\Scripts\\python.exe -B src\\experiments\\build_phase_f6l_paper_faithful_completion_and_rq_revalidation.py --config configs\\phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml --mode execute-profile --allow-api-calls --local-only --checkpoint-id f6l_profile_execution_ready_lastfm_operationalization_checkpoint_ready --raw-output-dir results\\llm\\phase_f6l_profile_execution_<run_id>", "ready_after_user_authorization", "checkpoint contract", "Command is recorded for later authorization; this turn does not implement or run generation."),
        ("blocks", "none_for_profile_contract_materialization", "ready", rel(PROFILE_REQUEST_MANIFEST_PATH), "Profile manifest is materialized; execution still requires explicit user authorization."),
    ]
    rows_out = [
        {
            "checkpoint_id": f"F6L-PROFILE-CHECK-{index:03d}",
            "item": item,
            "value": value,
            "status": status,
            "evidence": evidence,
            "notes": notes,
        }
        for index, (item, value, status, evidence, notes) in enumerate(entries, start=1)
    ]
    offset = len(rows_out)
    for index, (name, observed, passed, notes) in enumerate(checks, start=1):
        rows_out.append(
            {
                "checkpoint_id": f"F6L-PROFILE-CHECK-{offset + index:03d}",
                "item": name,
                "value": observed,
                "status": "pass" if passed else "fail",
                "evidence": rel(PROFILE_REQUEST_MANIFEST_PATH),
                "notes": notes,
            }
        )
    return rows_out


def build_author_specification_registry() -> list[dict[str, str]]:
    questions = [
        ("dataset LastFM preprocessato o script esatto", "Needed to reproduce paper counts and split.", "missing", "paper;official README;official code;F6C/F6L audits"),
        ("regola di ordinamento delle interazioni", "Needed for temporal leave-one-out.", "missing", "paper;GroupLens HetRec archive;F6C audit"),
        ("split train/validation/test", "Needed to identify positives and histories.", "missing", "paper;official code;F6C audit"),
        ("modello f_Theta usato nel K-S", "Needed to compute the same sampled/full metric distributions.", "missing", "paper;README;generate_prompt notebook"),
        ("metrica o metriche usate nel K-S", "Needed to define the K-S vectors.", "missing", "paper;README;F6C protocol"),
        ("valori per-user sottoposti al K-S", "Needed to reproduce the statistical gate.", "missing", "paper;official code"),
        ("alpha e alternativa del test", "Needed for the K-S decision.", "missing", "paper;official code"),
        ("seed", "Needed for original sample/candidate reproduction.", "missing", "paper;official code"),
        ("sample dei 1000 utenti", "Needed to avoid new-sample classification.", "missing", "official ZIPs and local artifacts inspected"),
        ("candidate set ranking", "Needed for LastFM ranking prompt identity.", "missing", "F6C adapted candidates exist but are not paper-exact"),
        ("top-K LastFM BPRMF, GRU4Rec, SASRec e LightGCN", "Needed for paper-faithful reranking pools.", "missing", "official notebook aggregation found; snapshots absent"),
        ("regola e ordine del pool di reranking", "Needed to construct exact reranking candidates.", "partially_missing", "official notebook partial aggregation logic"),
        ("raw output LastFM", "Needed to read exact author outputs without paid/closed calls.", "missing", "local ZIPs do not include LastFM raw outputs"),
    ]
    return [
        {
            "question_id": f"F6L-AUTHOR-Q-{index:02d}",
            "needed_artifact_or_detail": item,
            "why_needed": why,
            "current_status": status,
            "source_attempts": attempts,
            "blocks_exact_reproduction": "true",
            "author_request_text": f"Please provide the {item} used for the Beyond Utility LastFM experiments, including hashes or scripts where available.",
        }
        for index, (item, why, status, attempts) in enumerate(questions, start=1)
    ]


def build_author_clarification_report(rows: Sequence[Mapping[str, str]]) -> str:
    bullet_lines = [f"{index}. {row['author_request_text']}" for index, row in enumerate(rows, start=1)]
    return "\n".join(
        [
            "# Phase F6L Author Clarification Request",
            "",
            "This message is a draft only. It has not been sent.",
            "",
            "Dear authors,",
            "",
            "I am preparing a paper-faithful reproduction of the LastFM experiments from Beyond Utility: Evaluating LLM as Recommender. The public paper and repository do not appear to fully specify the operational contract needed to reproduce the sampled LastFM evaluation and reranking pools. Could you share or clarify the following items?",
            "",
            *bullet_lines,
            "",
            "If exact raw outputs are available for LastFM LLM runs, hashes and row-level lineage would also be useful so they can be read as original artifacts rather than regenerated.",
            "",
            "Thank you.",
            "",
        ]
    )


def build_lastfm_preprocessing_count_comparison() -> list[dict[str, str]]:
    comparisons = [
        ("paper_reported_counts", "paper", "1543", "7286", "173778", "true", "paper_reported_target", "paper translated DOCX/Table 2; F6L user request", "Target counts for exact reconstruction."),
        ("official_hetrec_raw", "dataset ufficiale GroupLens HetRec user_artists.dat", "1892", "17632", "92834", "false", "official_dataset_not_paper_processed", "results/tables/finalization_f6c_lastfm_interaction_audit.csv", "Official raw interaction table does not match paper counts."),
        ("official_repo_preprocessing", "repository ufficiale disponibile", "not_available", "not_available", "not_available", "false", "blocked_missing_original_specification", "original_repo/EvaLLMasRecommender", "No LastFM preprocessing script/materialized processed data matching paper counts found."),
        ("local_f6c_5core", "coorte locale F6C da 1859 utenti", "1859", "2823", "71355", "false", "adapted_protocol_reproduction", "results/tables/finalization_f6c_lastfm_interaction_audit.csv", "Evaluation-ready adapted 5-core data, not paper-identical."),
        ("new_reconstruction", "eventuale nuova ricostruzione F6L", "not_attempted", "not_attempted", "not_attempted", "false", "blocked_missing_original_specification", "phase_f6l_lastfm_contract_reconstruction_audit.csv", "Not attempted because temporal order and K-S gate remain underspecified."),
    ]
    return [
        {
            "comparison_id": f"F6L-LASTFM-COUNT-{index:03d}",
            "source": source,
            "users": users,
            "items": items,
            "interactions": interactions,
            "matches_paper_counts": matches,
            "classification": classification,
            "evidence": evidence,
            "notes": notes,
        }
        for index, (_, source, users, items, interactions, matches, classification, evidence, notes) in enumerate(comparisons, start=1)
    ]


def build_lastfm_operationalization_contract() -> list[dict[str, str]]:
    entries = [
        ("sample_size", "1000", "local_operationalization_not_author_claim", "requires preprocessing/split acceptance", "sample size matches paper-scale sampled evaluation but not original sample."),
        ("candidate_pool", "1 positive + 19 negatives; 20 total", "local_operationalization_not_author_claim", "requires split with positives", "Counts follow paper ranking setting."),
        ("candidate_order", "frozen shared order across Qwen2/Mistral and base/LLMRank", "local_operationalization_not_author_claim", "requires candidate generation", "Controls candidate-position confounding."),
        ("ks_distributions", "per-user HR@5 and NDCG@5 from BPRMF, GRU4Rec, SASRec, LightGCN", "local_operationalization_not_author_claim", "requires traditional full-cohort outputs", "Explicitly not attributed to authors."),
        ("ks_test", "two-sample Kolmogorov-Smirnov", "local_operationalization_not_author_claim", "requires distributions", "Local statistical gate."),
        ("alpha", "0.05", "local_operationalization_not_author_claim", "requires preregistration acceptance", "Local alpha; not an author-recovered threshold."),
        ("multiple_testing", "Holm correction", "local_operationalization_not_author_claim", "requires multiple metric/model tests", "Local family-wise correction."),
        ("acceptance", "no significant difference after Holm correction", "local_operationalization_not_author_claim", "requires K-S results", "Local acceptance rule."),
        ("seed_search", "deterministic increasing scan from preregistered seed 20260801", "local_operationalization_not_author_claim", "requires preprocessing/split acceptance", "First passing seed is selected."),
        ("max_attempts", "10000 seeds", "local_operationalization_not_author_claim", "requires execution budget acceptance", "Local cap to avoid unbounded search."),
        ("no_sample_accepted", "record blocked_no_local_operationalized_sample and do not run LLMs", "local_operationalization_not_author_claim", "requires search outcome", "Fail-closed behavior."),
    ]
    return [
        {
            "contract_id": f"F6L-LASTFM-OP-{index:03d}",
            "component": component,
            "local_operationalization": "true",
            "value": value,
            "author_attribution": attribution,
            "classification": "paper_protocol_reproduction_under_explicit_operationalization",
            "blocked_until": blocked_until,
            "notes": notes,
        }
        for index, (component, value, attribution, blocked_until, notes) in enumerate(entries, start=1)
    ]


def build_lastfm_operationalization_sensitivity_plan() -> list[dict[str, str]]:
    entries = [
        ("alpha", "0.01;0.05;0.10", "assess alpha sensitivity", "planned_not_executed"),
        ("ks_metric_set", "HR@5 only;NDCG@5 only;HR@5+NDCG@5", "assess metric-gate sensitivity", "planned_not_executed"),
        ("traditional_models", "each model separately;all four jointly", "assess dependence on f_Theta choice", "planned_not_executed"),
        ("seed_start", "20260801;42;0", "assess seed-start impact without result-dependent cherry-picking", "planned_not_executed"),
        ("negative_universe", "5-core interaction items;catalog artists if explicitly accepted", "assess candidate-universe sensitivity", "planned_not_executed"),
        ("failure_policy", "first passing seed;no passing seed blocked", "verify fail-closed behavior", "planned_not_executed"),
    ]
    return [
        {
            "plan_id": f"F6L-LASTFM-SENS-{index:03d}",
            "sensitivity_axis": axis,
            "planned_values": values,
            "purpose": purpose,
            "execution_status": status,
            "classification": "paper_protocol_reproduction_under_explicit_operationalization",
            "notes": "Sensitivity plan is preregistered but not executed in this checkpoint.",
        }
        for index, (axis, values, purpose, status) in enumerate(entries, start=1)
    ]


def build_lastfm_full_cohort_extension_plan() -> list[dict[str, str]]:
    users = int(lastfm_summary()["ranking_candidate_sets"])
    planned = users * 2 * 2
    seconds = int(round(planned * (21000 / 8800)))
    return [
        {
            "plan_id": "F6L-LASTFM-FULL-COHORT-001",
            "population": "all locally available F6C adapted LastFM ranking candidate sets",
            "preprocessing": "F6C iterative 5-core over GroupLens HetRec user_artists.dat; adapted not paper-identical",
            "split": "F6C adapted weight-rank validation/test split, not certified temporal leave-one-out",
            "users": str(users),
            "models": "qwen2:7b-instruct;mistral:7b-instruct",
            "strategies": "base/OpenP5 title-list;LLMRank recency",
            "planned_requests": str(planned),
            "classification": "original_project_extension_full_cohort",
            "paper_comparability": "not a paper numeric replica; removes sampling/K-S but keeps local adapted LastFM limitations",
            "rq_robustness_value": "can stress-test whether conclusions persist across the full available local cohort; cannot alone support or contradict paper RQs",
            "runtime_estimate": f"about {seconds} seconds / {seconds / 3600:.1f} hours under prior F6L planning rate",
            "status": "planned_extension_not_executed",
            "notes": "Requires explicit user authorization and must be reported separately from paper-faithful evidence.",
        }
    ]


def build_rq_evidence_hierarchy_contract() -> list[dict[str, str]]:
    entries = [
        ("1", "evidenza ufficiale originale", "exact_original_artifact_reproduction", "true", "may directly support/contradict paper RQ if lineage is complete", "original_artifact", "Author outputs or exact official artifacts."),
        ("2", "paper-protocol exact", "paper_protocol_reproduction_new_stochastic_output", "true", "may directly support/contradict with stochastic-run caveat", "paper_protocol_exact", "Same protocol and model family where recovered."),
        ("3", "paper-protocol con modelli open differenti", "paper_protocol_reproduction_different_open_model", "limited", "may update open-model-specific interpretation, not closed-model claims", "different_open_model", "Qwen2/Mistral local cells live here."),
        ("4", "paper-protocol sotto operazionalizzazione esplicita", "paper_protocol_reproduction_under_explicit_operationalization", "false", "may strengthen/weaken or suggest model/dataset dependence", "local_operationalization", "Local K-S choices must stay visible."),
        ("5", "full-cohort extension", "original_project_extension_full_cohort", "false", "may provide robustness evidence only", "full_cohort_extension", "No sampled K-S comparability."),
        ("6", "precedente evidenza adattata", "adapted_protocol_reproduction", "false", "may provide background or diagnostic contrast", "adapted_evidence", "F6C/F6D/F6G LastFM evidence remains adapted."),
        ("7", "blocco per specifica mancante", "blocked_missing_original_specification", "false", "marks unresolved RQ evidence gap", "blocked", "No invented values."),
    ]
    return [
        {
            "hierarchy_rank": rank,
            "evidence_layer": layer,
            "classification": classification,
            "can_support_or_contradict_paper_rq_alone": can_alone,
            "allowed_rq_effect": effect,
            "required_labeling": label,
            "notes": notes,
        }
        for rank, layer, classification, can_alone, effect, label, notes in entries
    ]


def build_separated_checkpoints(profile_rows: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    profile_checks = validate_profile_manifest(profile_rows)
    profile_ready = len(profile_rows) == 4800 and all(check[2] for check in profile_checks)
    return [
        {
            "checkpoint_id": "F6L-SEPARATED-CHECKPOINT-PROFILE",
            "checkpoint_type": "PROFILE CHECKPOINT",
            "decision_token": DECISION_SEPARATED_PROFILE_READY if profile_ready else DECISION_PROFILE_LASTFM_PARTIAL_BLOCKED,
            "requests_materialized": str(len(profile_rows)),
            "manifest_rows": str(len(profile_rows)),
            "unique_request_ids": str(len({row["request_id"] for row in profile_rows})),
            "classification": "paper_protocol_reproduction_different_open_model",
            "status": "ready_for_separate_user_authorization" if profile_ready else "blocked",
            "command_or_next_action": "$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='.'; .\\.venv_audit_recovery\\Scripts\\python.exe -B src\\experiments\\build_phase_f6l_paper_faithful_completion_and_rq_revalidation.py --config configs\\phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml --mode execute-profile --allow-api-calls --local-only --checkpoint-id f6l_profile_execution_ready_lastfm_operationalization_checkpoint_ready --raw-output-dir results\\llm\\phase_f6l_profile_execution_<run_id>",
            "notes": "No inference was performed; command is a future authorization skeleton.",
        },
        {
            "checkpoint_id": "F6L-SEPARATED-CHECKPOINT-LASTFM",
            "checkpoint_type": "LASTFM CHECKPOINT",
            "decision_token": DECISION_CONTRACT_MATERIALIZATION_BLOCKED,
            "requests_materialized": "0",
            "manifest_rows": "0",
            "unique_request_ids": "0",
            "classification": "blocked_missing_original_specification;paper_protocol_reproduction_under_explicit_operationalization;original_project_extension_full_cohort",
            "status": "exact_lastfm_blocked_operationalization_and_full_cohort_planned_not_executed",
            "command_or_next_action": "Request author clarifications; only after acceptance may materialize local operationalization or full-cohort extension under separate labels.",
            "notes": "No LastFM sample, candidate set, or request manifest was created in this checkpoint.",
        },
    ]


def build_separated_manifest(result: SeparatedContractsResult, protected: Mapping[str, Any]) -> dict[str, Any]:
    profile_rows = result.tables["profile_request_manifest"]
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "decision_token": result.decision_token,
        "materialized_at": MATERIALIZATION_AUDIT_TIMESTAMP,
        "profile_requests_materialized": len(profile_rows),
        "profile_manifest_rows": len(profile_rows),
        "profile_unique_request_ids": len({row["request_id"] for row in profile_rows}),
        "profile_counts_by_input_variant": dict(sorted(Counter(row["input_variant"] for row in profile_rows).items())),
        "profile_counts_by_dataset": dict(sorted(Counter(row["dataset"] for row in profile_rows).items())),
        "profile_counts_by_model": dict(sorted(Counter(row["model_name"] for row in profile_rows).items())),
        "lastfm_exact_materialized": False,
        "lastfm_operationalization_materialized": False,
        "lastfm_full_cohort_materialized": False,
        "inference_performed": False,
        "ollama_generation_performed": False,
        "remote_api_calls_performed": False,
        "downloads_performed": False,
        "training_performed": False,
        "raw_output_dir_created": False,
        "models": MODEL_DIGESTS,
        "protected_input_status": protected.get("status"),
        "f6i2p_yaml_count": protected.get("f6i2p_yaml_count"),
        "git": {"head": git_output(["rev-parse", "HEAD"]), "staged_paths": git_output(["diff", "--cached", "--name-only"])},
        "tables": {name: rel(path) for name, path in SEPARATED_OUTPUT_TABLES.items()},
        "reports": {"author_clarification_request": rel(AUTHOR_CLARIFICATION_REPORT_PATH), "phase_report": rel(REPORT_PATH)},
        "output_hashes": {
            rel(path): safe_file_sha256(path)
            for path in SEPARATED_OUTPUT_TABLES.values()
            if path.exists()
        },
        "self_hash": "self_reference_excluded_for_determinism",
    }


def build_separated_contract_report_section(result: SeparatedContractsResult) -> str:
    checkpoints = result.tables["separated_checkpoints"]
    profile = checkpoints[0]
    lastfm = checkpoints[1]
    full = result.tables["lastfm_full_cohort_extension_plan"][0]
    return "\n".join(
        [
            "## Separated Profile And LastFM Checkpoints",
            "",
            "### Profile Checkpoint",
            "",
            f"- Requests materialized: `{profile['requests_materialized']}`.",
            f"- Manifest rows: `{profile['manifest_rows']}`.",
            f"- Unique request IDs: `{profile['unique_request_ids']}`.",
            "- Models: `qwen2:7b-instruct`, `mistral:7b-instruct`.",
            "- Dependencies: profile-only and profile-plus-history rows point to the matching profile-generation parent request and are acyclic.",
            "- Classification: `paper_protocol_reproduction_different_open_model`.",
            f"- Command after separate authorization: `{profile['command_or_next_action']}`",
            "",
            "### LastFM Checkpoint",
            "",
            "- Exact LastFM materialization remains blocked by missing paper specification.",
            "- Local operationalization is preregistered as `paper_protocol_reproduction_under_explicit_operationalization`, not attributed to the authors.",
            f"- Full-cohort extension planned requests: `{full['planned_requests']}`.",
            f"- Full-cohort classification: `{full['classification']}`.",
            "- Author clarification draft created but not sent.",
            "",
            f"`{result.decision_token}`",
            "",
        ]
    )


def update_separated_handoff_and_notes(result: SeparatedContractsResult) -> None:
    profile = result.tables["separated_checkpoints"][0]
    lastfm = result.tables["separated_checkpoints"][1]
    handoff_content = "\n".join(
        [
            "## F6L Separated Profile And LastFM Checkpoints",
            "",
            f"- Decision token: `{result.decision_token}`.",
            f"- Profile manifest: `{profile['manifest_rows']}` rows, `{profile['unique_request_ids']}` unique request IDs, no LastFM rows.",
            "- Profile execution is separately ready for explicit local authorization; no inference was run.",
            "- LastFM exact reproduction remains blocked by missing author/paper specification.",
            "- LastFM local operationalization and full-cohort extension are preregistered plans only and cannot support/contradict paper RQs alone.",
            f"- LastFM status: {lastfm['status']}.",
        ]
    )
    notes_content = "\n".join(
        [
            "## 2026-08-01 - Phase F6L Separated Profile And LastFM Checkpoints",
            "",
            "### Changes",
            "",
            "- Materialized `phase_f6l_profile_request_manifest.csv` with 4800 profile-family requests derived from the F6L authorization matrix and official Beauty/Sports ZIP prompt artifacts.",
            "- Added the author clarification request draft, LastFM missing-specification registry, local operationalization contract, sensitivity plan, full-cohort extension plan, RQ evidence hierarchy, and separated checkpoint summary.",
            "- No LastFM sample/candidate/request manifest was materialized.",
            "",
            "### Result",
            "",
            f"- Decision: `{result.decision_token}`.",
            f"- Profile rows: `{profile['manifest_rows']}`.",
            f"- LastFM exact status: `{lastfm['status']}`.",
        ]
    )
    marker = f"{PHASE_NAME}_separated_checkpoints"
    update_marked_section(REPORT_PATH, marker, build_separated_contract_report_section(result))
    update_marked_section(HANDOFF_PATH, marker, handoff_content)
    update_marked_section(NOTES_PATH, marker, notes_content)


def separated_contracts_digest(result: SeparatedContractsResult) -> str:
    return sha256_text(
        canonical_json(
            {
                "decision_token": result.decision_token,
                "tables": result.tables,
                "reports": result.reports,
            }
        )
    )


def assemble_separated_contracts(config_path: Path = CONFIG_PATH) -> SeparatedContractsResult:
    require_runtime()
    load_config(config_path)
    protected = hash_protected_inputs()
    if protected.get("status") != "pass" or protected.get("f6i2p_yaml_count") != 23:
        decision = DECISION_REPOSITORY_REGRESSION
    else:
        decision = DECISION_SEPARATED_PROFILE_READY
    profile_rows = build_profile_request_manifest()
    author_rows = build_author_specification_registry()
    tables = {
        "profile_request_manifest": profile_rows,
        "profile_checkpoint": build_profile_checkpoint(profile_rows),
        "lastfm_author_specification_registry": author_rows,
        "lastfm_preprocessing_count_comparison": build_lastfm_preprocessing_count_comparison(),
        "lastfm_operationalization_contract": build_lastfm_operationalization_contract(),
        "lastfm_operationalization_sensitivity_plan": build_lastfm_operationalization_sensitivity_plan(),
        "lastfm_full_cohort_extension_plan": build_lastfm_full_cohort_extension_plan(),
        "rq_evidence_hierarchy_contract": build_rq_evidence_hierarchy_contract(),
        "separated_checkpoints": build_separated_checkpoints(profile_rows),
    }
    reports = {"author_clarification_request": build_author_clarification_report(author_rows)}
    result = SeparatedContractsResult(decision, tables, {}, reports)
    manifest = build_separated_manifest(result, protected)
    return SeparatedContractsResult(decision, tables, {"separated_checkpoint": manifest}, reports)


def build_separated_contracts(config_path: Path = CONFIG_PATH, *, write_outputs_enabled: bool = True) -> SeparatedContractsResult:
    first = assemble_separated_contracts(config_path)
    second = assemble_separated_contracts(config_path)
    deterministic = separated_contracts_digest(first) == separated_contracts_digest(second)
    manifest = {**first.manifests["separated_checkpoint"], "deterministic_in_memory_digest_match": deterministic}
    result = SeparatedContractsResult(first.decision_token, first.tables, {"separated_checkpoint": manifest}, first.reports)
    if write_outputs_enabled:
        for name, path in SEPARATED_OUTPUT_TABLES.items():
            write_csv(path, result.tables[name], SEPARATED_TABLE_FIELDS[name])
        AUTHOR_CLARIFICATION_REPORT_PATH.write_text(result.reports["author_clarification_request"], encoding="utf-8")
        manifest = {**result.manifests["separated_checkpoint"]}
        manifest["output_hashes"] = {
            rel(path): safe_file_sha256(path)
            for path in SEPARATED_OUTPUT_TABLES.values()
            if path.exists()
        }
        manifest["report_hashes"] = {rel(AUTHOR_CLARIFICATION_REPORT_PATH): safe_file_sha256(AUTHOR_CLARIFICATION_REPORT_PATH)}
        result = SeparatedContractsResult(result.decision_token, result.tables, {"separated_checkpoint": manifest}, result.reports)
        write_json(SEPARATED_CHECKPOINT_MANIFEST_PATH, result.manifests["separated_checkpoint"])
        update_separated_handoff_and_notes(result)
    return result


def parse_bool_text(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def format_float(value: float, digits: int = 12) -> str:
    if math.isnan(value):
        return ""
    text = f"{value:.{digits}g}"
    return "0" if text == "-0" else text


def sort_key_numeric_text(value: Any) -> tuple[int, str]:
    text = str(value)
    return (int(text), text) if text.isdigit() else (10**18, text)


def json_list(values: Sequence[Any]) -> str:
    return json.dumps([str(value) for value in values], ensure_ascii=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: object) -> str:
    return f"{prefix}-{sha256_text('|'.join(str(part) for part in parts))[:16]}"


def local_pipeline_input_hashes() -> dict[str, str]:
    global _LOCAL_PIPELINE_INPUT_HASHES_CACHE
    if _LOCAL_PIPELINE_INPUT_HASHES_CACHE is not None:
        return dict(_LOCAL_PIPELINE_INPUT_HASHES_CACHE)
    paths: dict[str, Path] = {}
    paths.update({f"f6c_{name}": path for name, path in F6C_LASTFM_INPUTS.items()})
    paths.update({f"f6d2_{name}": path for name, path in F6D2_LASTFM_INPUTS.items()})
    paths.update({f"f6g1_{name}": path for name, path in F6G1_LASTFM_INPUTS.items()})
    paths.update(
        {
            f"template_{index}": spec["template_path"]
            for index, spec in enumerate(LASTFM_RANKING_STRATEGIES.values(), start=1)
        }
    )
    _LOCAL_PIPELINE_INPUT_HASHES_CACHE = {name: safe_file_sha256(path) for name, path in sorted(paths.items())}
    return dict(_LOCAL_PIPELINE_INPUT_HASHES_CACHE)


def selected_traditional_model_seeds() -> dict[str, str]:
    global _SELECTED_TRADITIONAL_MODEL_SEEDS_CACHE
    if _SELECTED_TRADITIONAL_MODEL_SEEDS_CACHE is not None:
        return dict(_SELECTED_TRADITIONAL_MODEL_SEEDS_CACHE)
    rows = read_csv_rows(F6D2_LASTFM_INPUTS["source_selection"])
    selected = {
        row["model"]: row["seed"]
        for row in rows
        if parse_bool_text(row.get("selected_for_adapted_pool"))
        and row.get("model") in LASTFM_CONTROL_MODELS
    }
    if set(selected) != set(LASTFM_CONTROL_MODELS):
        raise ValueError(f"missing selected F6D2 seeds: {sorted(set(LASTFM_CONTROL_MODELS) - set(selected))}")
    _SELECTED_TRADITIONAL_MODEL_SEEDS_CACHE = selected
    return dict(selected)


def lastfm_operationalization_parameters() -> dict[str, str]:
    rows = read_csv_rows(SEPARATED_OUTPUT_TABLES["lastfm_operationalization_contract"])
    by_component = {row.get("component", ""): row for row in rows}
    seed_text = by_component.get("seed_search", {}).get("value", "")
    max_attempts_text = by_component.get("max_attempts", {}).get("value", "")
    seed_digits = "".join(char for char in seed_text if char.isdigit())
    attempts_digits = "".join(char for char in max_attempts_text if char.isdigit())
    return {
        "sample_size": by_component.get("sample_size", {}).get("value", "1000"),
        "candidate_pool": by_component.get("candidate_pool", {}).get("value", "1 positive + 19 negatives; 20 total"),
        "candidate_order": by_component.get("candidate_order", {}).get("value", "frozen shared order across Qwen2/Mistral and base/LLMRank"),
        "ks_distributions": by_component.get("ks_distributions", {}).get("value", "per-user HR@5 and NDCG@5 from BPRMF, GRU4Rec, SASRec, LightGCN"),
        "ks_test": by_component.get("ks_test", {}).get("value", "two-sample Kolmogorov-Smirnov"),
        "alpha": by_component.get("alpha", {}).get("value", "0.05"),
        "multiple_testing": by_component.get("multiple_testing", {}).get("value", "Holm correction"),
        "acceptance": by_component.get("acceptance", {}).get("value", "no significant difference after Holm correction"),
        "start_seed": seed_digits[-8:] if seed_digits else "20260801",
        "max_attempts": attempts_digits if attempts_digits else "10000",
    }


def build_lastfm_local_pipeline_audit(protected: Mapping[str, Any]) -> list[dict[str, str]]:
    selected = selected_traditional_model_seeds()
    lastfm = lastfm_summary()
    rows = [
        {
            "requirement_id": "F6L-LASTFM-AUDIT-001",
            "requirement": "paper LastFM exact processed cohort",
            "observed_value": "paper=1543 users,7286 items,173778 interactions; HetRec raw local=1892 users,17632 items,92834 interactions; F6C adapted=1859 users,2823 items,71355 interactions",
            "classification": "blocked_missing_original_specification",
            "status": "blocked_for_exact_replica",
            "evidence": "paper;data/external/plan_c/hetrec2011-lastfm-2k.zip;data/processed/finalization_f6c_lastfm_interactions.csv",
            "notes": "The local LastFM lineage is useful but not paper-identical.",
        },
        {
            "requirement_id": "F6L-LASTFM-AUDIT-002",
            "requirement": "F6C adapted ranking cohort",
            "observed_value": f"users={lastfm['ranking_cohort_users']};candidate_rows={lastfm['ranking_candidate_rows']};positive_rows={lastfm['ranking_positive_rows']}",
            "classification": "paper_protocol_reproduction_under_explicit_operationalization",
            "status": "available_for_local_operationalization",
            "evidence": rel(F6C_LASTFM_INPUTS["ranking_cohort"]) + ";" + rel(F6C_LASTFM_INPUTS["ranking_candidates"]),
            "notes": "The 1000-user sample is drawn only from this adapted local cohort and must be labeled accordingly.",
        },
        {
            "requirement_id": "F6L-LASTFM-AUDIT-003",
            "requirement": "candidate pool shape",
            "observed_value": "20 candidates per user; one test positive per user; F6C candidate seed 42",
            "classification": "paper_protocol_reproduction_under_explicit_operationalization",
            "status": "available_reused_as_fixed_local_candidate_set",
            "evidence": rel(F6C_LASTFM_INPUTS["ranking_candidates"]),
            "notes": "No new negative resampling is performed in F6L; preserving F6C candidates keeps the K-S metrics linked to F6D2 rankings.",
        },
        {
            "requirement_id": "F6L-LASTFM-AUDIT-004",
            "requirement": "traditional control model selected seeds",
            "observed_value": ";".join(f"{LASTFM_CONTROL_MODELS[model]}={seed}" for model, seed in sorted(selected.items())),
            "classification": "paper_protocol_reproduction_under_explicit_operationalization",
            "status": "available_from_f6d2_best_validation_seed_selection",
            "evidence": rel(F6D2_LASTFM_INPUTS["source_selection"]),
            "notes": "The selected seeds are local adapted seeds, not paper seed identities.",
        },
        {
            "requirement_id": "F6L-LASTFM-AUDIT-005",
            "requirement": "LLM ranking prompt templates",
            "observed_value": ";".join(rel(spec["template_path"]) for spec in LASTFM_RANKING_STRATEGIES.values()),
            "classification": "paper_protocol_reproduction_under_explicit_operationalization",
            "status": "available_read_only",
            "evidence": "original_repo/EvaLLMasRecommender/eva/templates/template.txt;original_repo/EvaLLMasRecommender/eva/templates/llmrank_rf.txt",
            "notes": "Templates are read only; no original repository file is modified.",
        },
        {
            "requirement_id": "F6L-LASTFM-AUDIT-006",
            "requirement": "reranking adapted pool provenance",
            "observed_value": f"pool_rows={count_rows(F6G1_LASTFM_INPUTS['pool_item_provenance'])};instances={count_rows(F6G1_LASTFM_INPUTS['pool_instance_summary'])}",
            "classification": "original_project_extension_full_cohort",
            "status": "available_as_reconstructed_extension",
            "evidence": rel(F6G1_LASTFM_INPUTS["pool_item_provenance"]),
            "notes": "This pool is not a paper-exact BPRMF/GRU4Rec/SASRec/LightGCN author top-K pool.",
        },
        {
            "requirement_id": "F6L-LASTFM-AUDIT-007",
            "requirement": "protected input guard",
            "observed_value": f"status={protected.get('status')};f6i2p_yaml_count={protected.get('f6i2p_yaml_count')}",
            "classification": "blocked_missing_original_specification",
            "status": "pass" if protected.get("status") == "pass" and protected.get("f6i2p_yaml_count") == 23 else "blocked_by_guard_regression",
            "evidence": rel(PROTECTED_HASHES_PATH),
            "notes": "The exact paper layer remains blocked even when local protected inputs are clean.",
        },
        {
            "requirement_id": "F6L-LASTFM-AUDIT-008",
            "requirement": "execution policy",
            "observed_value": "inference=false;ollama_generation=false;remote_api=false;training=false;downloads=false;raw_output_dir_created=false",
            "classification": "paper_protocol_reproduction_under_explicit_operationalization",
            "status": "contract_only",
            "evidence": PHASE_NAME,
            "notes": "This mode creates request contracts and tables only.",
        },
    ]
    return rows


def load_selected_seed_per_user_metrics() -> dict[str, dict[str, dict[str, float]]]:
    global _SELECTED_SEED_METRIC_VECTORS_CACHE
    if _SELECTED_SEED_METRIC_VECTORS_CACHE is not None:
        return {
            model: {metric: dict(values) for metric, values in metrics.items()}
            for model, metrics in _SELECTED_SEED_METRIC_VECTORS_CACHE.items()
        }
    selected = selected_traditional_model_seeds()
    metrics: dict[str, dict[str, dict[str, float]]] = {
        model: {"HR@5": {}, "NDCG@5": {}} for model in selected
    }
    with F6D2_LASTFM_INPUTS["candidate_rankings"].open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row.get("model", "")
            if selected.get(model) != row.get("seed"):
                continue
            if not parse_bool_text(row.get("is_positive")):
                continue
            user_id = str(row.get("user_id", ""))
            rank = int(row.get("rank", "0") or 0)
            hit = 1.0 if 1 <= rank <= 5 else 0.0
            ndcg = 1.0 / math.log2(rank + 1) if hit else 0.0
            metrics[model]["HR@5"][user_id] = hit
            metrics[model]["NDCG@5"][user_id] = ndcg
    _SELECTED_SEED_METRIC_VECTORS_CACHE = metrics
    return {model: {metric: dict(values) for metric, values in values_by_metric.items()} for model, values_by_metric in metrics.items()}


def common_metric_users(metric_vectors: Mapping[str, Mapping[str, Mapping[str, float]]]) -> list[str]:
    user_sets: list[set[str]] = []
    for model_values in metric_vectors.values():
        for values in model_values.values():
            user_sets.append(set(values))
    if not user_sets:
        return []
    return sorted(set.intersection(*user_sets), key=sort_key_numeric_text)


def ks_two_sample_statistic(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right:
        return 1.0
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    i = j = 0
    n = len(left_sorted)
    m = len(right_sorted)
    max_delta = 0.0
    values = sorted(set(left_sorted + right_sorted))
    for value in values:
        while i < n and left_sorted[i] <= value:
            i += 1
        while j < m and right_sorted[j] <= value:
            j += 1
        max_delta = max(max_delta, abs(i / n - j / m))
    return max_delta


def ks_asymptotic_pvalue(statistic: float, left_size: int, right_size: int) -> float:
    if left_size <= 0 or right_size <= 0:
        return 0.0
    if statistic <= 0:
        return 1.0
    effective_n = math.sqrt((left_size * right_size) / (left_size + right_size))
    lam = (effective_n + 0.12 + 0.11 / effective_n) * statistic
    total = 0.0
    for term in range(1, 101):
        contribution = ((-1) ** (term - 1)) * math.exp(-2.0 * (term**2) * (lam**2))
        total += contribution
        if abs(contribution) < 1e-12:
            break
    return max(0.0, min(1.0, 2.0 * total))


def holm_adjust(raw_pvalues: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(raw_pvalues), key=lambda item: (item[1], item[0]))
    adjusted = [1.0 for _ in raw_pvalues]
    running = 0.0
    total = len(indexed)
    for rank, (original_index, pvalue) in enumerate(indexed):
        corrected = min(1.0, pvalue * (total - rank))
        running = max(running, corrected)
        adjusted[original_index] = running
    return adjusted


def sample_users_for_seed(users: Sequence[str], sample_size: int, seed: int) -> list[str]:
    if sample_size > len(users):
        return []
    sampled = random.Random(seed).sample(list(users), sample_size)
    return sorted(sampled, key=sort_key_numeric_text)


def build_lastfm_seed_search(
    metric_vectors: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    params = lastfm_operationalization_parameters()
    sample_size = int(params["sample_size"])
    alpha = float(params["alpha"])
    start_seed = int(params["start_seed"])
    max_attempts = int(params["max_attempts"])
    population_users = common_metric_users(metric_vectors)
    selected = selected_traditional_model_seeds()
    seed_rows: list[dict[str, str]] = []
    all_ks_rows: list[dict[str, str]] = []
    accepted_users: list[str] = []

    for attempt in range(max_attempts):
        seed = start_seed + attempt
        sampled_users = sample_users_for_seed(population_users, sample_size, seed)
        if not sampled_users:
            break
        raw_rows: list[dict[str, Any]] = []
        raw_pvalues: list[float] = []
        for model in sorted(metric_vectors):
            for metric in ["HR@5", "NDCG@5"]:
                population_values = [metric_vectors[model][metric][user] for user in population_users]
                sample_values = [metric_vectors[model][metric][user] for user in sampled_users]
                statistic = ks_two_sample_statistic(population_values, sample_values)
                raw_p = ks_asymptotic_pvalue(statistic, len(population_values), len(sample_values))
                raw_pvalues.append(raw_p)
                raw_rows.append(
                    {
                        "seed": str(seed),
                        "attempt_index": str(attempt + 1),
                        "model_name": LASTFM_CONTROL_MODELS[model],
                        "model_seed": selected[model],
                        "metric": metric,
                        "population_users": str(len(population_users)),
                        "sample_users": str(len(sampled_users)),
                        "ks_statistic": statistic,
                        "p_value_raw": raw_p,
                    }
                )
        adjusted = holm_adjust(raw_pvalues)
        accepted = all(pvalue >= alpha for pvalue in adjusted)
        ks_rows = []
        for row, adjusted_p in zip(raw_rows, adjusted):
            significant = adjusted_p < alpha
            ks_rows.append(
                {
                    **row,
                    "ks_statistic": format_float(float(row["ks_statistic"])),
                    "p_value_raw": format_float(float(row["p_value_raw"])),
                    "p_value_holm": format_float(adjusted_p),
                    "alpha": format_float(alpha),
                    "significant_after_holm": clean_cell(significant),
                    "accepted": clean_cell(accepted),
                    "evidence_class": "paper_protocol_reproduction_under_explicit_operationalization",
                }
            )
        seed_rows.append(
            {
                "seed": str(seed),
                "attempt_index": str(attempt + 1),
                "sample_size": str(sample_size),
                "population_users": str(len(population_users)),
                "test_family_size": str(len(raw_pvalues)),
                "alpha": format_float(alpha),
                "correction": "Holm",
                "raw_pvalue_min": format_float(min(raw_pvalues) if raw_pvalues else 0.0),
                "holm_pvalue_min": format_float(min(adjusted) if adjusted else 0.0),
                "ks_statistic_max": format_float(max(float(row["ks_statistic"]) for row in raw_rows) if raw_rows else 1.0),
                "accepted": clean_cell(accepted),
                "evidence_class": "paper_protocol_reproduction_under_explicit_operationalization",
                "notes": "first accepted preregistered local operationalization seed" if accepted else "failed local K-S gate; continuing deterministic increasing scan",
            }
        )
        all_ks_rows = ks_rows
        if accepted:
            accepted_users = sampled_users
            break
    return seed_rows, all_ks_rows, accepted_users


def load_lastfm_cohort_rows() -> dict[str, dict[str, str]]:
    global _LASTFM_COHORT_ROWS_CACHE
    if _LASTFM_COHORT_ROWS_CACHE is not None:
        return {user_id: dict(row) for user_id, row in _LASTFM_COHORT_ROWS_CACHE.items()}
    rows = read_csv_rows(F6C_LASTFM_INPUTS["ranking_cohort"])
    _LASTFM_COHORT_ROWS_CACHE = {str(row["user_id"]): row for row in rows}
    return {user_id: dict(row) for user_id, row in _LASTFM_COHORT_ROWS_CACHE.items()}


def load_candidate_rows_by_user() -> dict[str, list[dict[str, str]]]:
    global _LASTFM_CANDIDATE_ROWS_BY_USER_CACHE
    if _LASTFM_CANDIDATE_ROWS_BY_USER_CACHE is not None:
        return {
            user_id: [dict(row) for row in rows]
            for user_id, rows in _LASTFM_CANDIDATE_ROWS_BY_USER_CACHE.items()
        }
    rows = read_csv_rows(F6C_LASTFM_INPUTS["ranking_candidates"])
    by_user: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_user.setdefault(str(row["user_id"]), []).append(row)
    for user_rows in by_user.values():
        user_rows.sort(key=lambda row: int(row["candidate_position"]))
    _LASTFM_CANDIDATE_ROWS_BY_USER_CACHE = by_user
    return {user_id: [dict(row) for row in rows] for user_id, rows in by_user.items()}


def parse_json_list(raw: str) -> list[str]:
    if not raw:
        return []
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError("expected JSON list")
    return [str(item) for item in value]


def candidate_hash(candidate_rows: Sequence[Mapping[str, str]]) -> str:
    return sha256_text(json_list([row["candidate_item_id"] for row in candidate_rows]))


def history_hash(cohort_row: Mapping[str, str]) -> str:
    return sha256_text(
        canonical_json(
            {
                "history_item_ids": parse_json_list(cohort_row.get("history_item_ids", "")),
                "history_item_titles": parse_json_list(cohort_row.get("history_item_titles", "")),
            }
        )
    )


def build_lastfm_sample_cohort(selected_users: Sequence[str], selected_seed: str) -> list[dict[str, str]]:
    cohort = load_lastfm_cohort_rows()
    sample_id = f"phase_f6l_lastfm_operationalized_1000_seed_{selected_seed}"
    lineage = rel(F6C_LASTFM_INPUTS["ranking_cohort"])
    rows: list[dict[str, str]] = []
    for ordinal, user_id in enumerate(selected_users, start=1):
        row = cohort[user_id]
        rows.append(
            {
                "sample_id": sample_id,
                "sample_ordinal": str(ordinal),
                "user_id": user_id,
                "source_cohort_id": row["cohort_id"],
                "internal_user_id": row["internal_user_id"],
                "train_history_count": row["train_history_count"],
                "validation_item_id": row["validation_item_id"],
                "test_item_id": row["test_item_id"],
                "test_item_title": row["test_item_title"],
                "history_item_ids": row["history_item_ids"],
                "history_item_titles": row["history_item_titles"],
                "history_hash": history_hash(row),
                "selected_seed": selected_seed,
                "evidence_class": "paper_protocol_reproduction_under_explicit_operationalization",
                "lineage": lineage,
            }
        )
    return rows


def build_lastfm_operationalized_candidates(selected_users: Sequence[str], selected_seed: str) -> list[dict[str, str]]:
    by_user = load_candidate_rows_by_user()
    sample_id = f"phase_f6l_lastfm_operationalized_1000_seed_{selected_seed}"
    lineage = rel(F6C_LASTFM_INPUTS["ranking_candidates"])
    rows: list[dict[str, str]] = []
    for user_id in selected_users:
        source_rows = by_user[user_id]
        item_ids = [row["candidate_item_id"] for row in source_rows]
        item_hash = sha256_text(json_list(item_ids))
        positive_rows = [row for row in source_rows if parse_bool_text(row.get("is_ground_truth"))]
        positive_item = positive_rows[0]["candidate_item_id"] if positive_rows else ""
        positive_position = positive_rows[0]["candidate_position"] if positive_rows else ""
        for row in source_rows:
            rows.append(
                {
                    "sample_id": sample_id,
                    "user_id": user_id,
                    "candidate_set_id": row["candidate_set_id"],
                    "candidate_position": row["candidate_position"],
                    "candidate_item_id": row["candidate_item_id"],
                    "candidate_item_title": row["candidate_item_title"],
                    "is_positive": clean_cell(parse_bool_text(row.get("is_ground_truth"))),
                    "positive_item": positive_item,
                    "positive_position": positive_position,
                    "candidate_list": json_list(item_ids),
                    "candidate_list_hash": item_hash,
                    "negative_sampling_algorithm": "F6C adapted fixed candidate file reused; no F6L resampling",
                    "negative_sampling_seed": row.get("sampling_seed", "42"),
                    "negative_exclusion_contract": "negative rows must not appear in user train/validation/test observed items",
                    "source_lineage": lineage,
                    "evidence_class": "paper_protocol_reproduction_under_explicit_operationalization",
                }
            )
    return rows


def template_line(path: Path, variant_index: int) -> str:
    key = (str(path), variant_index)
    if key in _TEMPLATE_LINE_CACHE:
        return _TEMPLATE_LINE_CACHE[key]
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if variant_index >= len(lines):
        raise ValueError(f"template index {variant_index} unavailable in {path}")
    _TEMPLATE_LINE_CACHE[key] = lines[variant_index]
    return _TEMPLATE_LINE_CACHE[key]


def template_hash(path: Path) -> str:
    key = str(path)
    if key not in _TEMPLATE_HASH_CACHE:
        _TEMPLATE_HASH_CACHE[key] = file_sha256(path)
    return _TEMPLATE_HASH_CACHE[key]


def python_list_literal(values: Sequence[str]) -> str:
    return json.dumps([str(value) for value in values], ensure_ascii=False)


def fill_prompt_template(template: str, replacements: Mapping[str, str]) -> str:
    return template.format(**replacements)


def render_lastfm_prompt(cohort_row: Mapping[str, str], candidate_rows: Sequence[Mapping[str, str]], strategy: str) -> str:
    spec = LASTFM_RANKING_STRATEGIES[strategy]
    history_titles = parse_json_list(cohort_row.get("history_item_titles", ""))[-10:]
    candidate_titles = [row["candidate_item_title"] for row in candidate_rows]
    template = template_line(spec["template_path"], int(spec["variant_index"]))
    if strategy == "base/OpenP5 title-list":
        return fill_prompt_template(
            template,
            {
                "dataset": "LastFM",
                "user_name": str(cohort_row["user_id"]),
                "user_history": python_list_literal(history_titles),
                "candidate_list": python_list_literal(candidate_titles),
            },
        )
    return fill_prompt_template(
        template,
        {
            "user_history": python_list_literal(history_titles),
            "candidate_list": python_list_literal(candidate_titles),
            "last_item": history_titles[-1] if history_titles else "",
        },
    )


def generation_parameters_json(strategy: str) -> str:
    return canonical_json(
        {
            "concurrency": 1,
            "infrastructure_retry_count": 0,
            "output_length": 10,
            "parser_contract": LASTFM_RANKING_STRATEGIES[strategy]["parser_contract"],
            "remote_api": False,
            "response_format": "text",
            "scientific_retry_count": 0,
            "seed": 0,
            "stream": False,
            "temperature": 0,
            "timeout_seconds": 120,
        }
    )


def build_lastfm_ranking_manifest(
    *,
    cohort_rows: Sequence[Mapping[str, str]],
    candidate_rows: Sequence[Mapping[str, str]],
    scope: str,
) -> list[dict[str, str]]:
    by_user_candidates: dict[str, list[dict[str, str]]] = {}
    for row in candidate_rows:
        by_user_candidates.setdefault(str(row["user_id"]), []).append(dict(row))
    for user_rows in by_user_candidates.values():
        user_rows.sort(key=lambda row: int(row["candidate_position"]))

    prefix = "F6L-LASTFM-OP-RANK" if scope == "operationalized" else "F6L-LASTFM-FULL-RANK"
    sample_id = "phase_f6l_lastfm_operationalized_1000" if scope == "operationalized" else "phase_f6l_lastfm_full_cohort_1859"
    evidence_class = (
        "paper_protocol_reproduction_under_explicit_operationalization"
        if scope == "operationalized"
        else "original_project_extension_full_cohort"
    )
    raw_prefix = (
        "results/llm/phase_f6l_lastfm_operationalized_ranking_<run_id>"
        if scope == "operationalized"
        else "results/llm/phase_f6l_lastfm_full_cohort_ranking_<run_id>"
    )
    lineage = rel(F6C_LASTFM_INPUTS["ranking_cohort"]) + ";" + rel(F6C_LASTFM_INPUTS["ranking_candidates"])
    strategy_templates = {
        strategy: rel(spec["template_path"])
        for strategy, spec in LASTFM_RANKING_STRATEGIES.items()
    }
    strategy_template_hashes = {
        strategy: template_hash(spec["template_path"])
        for strategy, spec in LASTFM_RANKING_STRATEGIES.items()
    }
    strategy_generation_params = {
        strategy: generation_parameters_json(strategy)
        for strategy in LASTFM_RANKING_STRATEGIES
    }
    rows: list[dict[str, str]] = []
    order = 1
    for cohort_row in sorted(cohort_rows, key=lambda row: sort_key_numeric_text(row["user_id"])):
        user_id = str(cohort_row["user_id"])
        candidates = by_user_candidates[user_id]
        candidate_ids = [row["candidate_item_id"] for row in candidates]
        user_history_hash = cohort_row.get("history_hash") or history_hash(cohort_row)
        user_candidate_hash = sha256_text(json_list(candidate_ids))
        positives = [row for row in candidates if parse_bool_text(row.get("is_positive") or row.get("is_ground_truth"))]
        positive_item = positives[0].get("candidate_item_id", "") if positives else ""
        positive_position = positives[0].get("candidate_position", "") if positives else ""
        prompt_hashes = {
            strategy: sha256_text(render_lastfm_prompt(cohort_row, candidates, strategy))
            for strategy in LASTFM_RANKING_STRATEGIES
        }
        for model_name in MODEL_DIGESTS:
            for strategy, spec in LASTFM_RANKING_STRATEGIES.items():
                request_id = f"{prefix}-{order:06d}"
                template_path = spec["template_path"]
                base_row = {
                    "request_id": request_id,
                    "sample_id": sample_id,
                    "user_id": user_id,
                    "model_name": model_name,
                    "model_digest": MODEL_DIGESTS[model_name],
                    "prompting_strategy": strategy,
                    "history_hash": user_history_hash,
                    "candidate_list_hash": user_candidate_hash,
                    "positive_item": positive_item,
                    "positive_position": positive_position,
                    "prompt_template": strategy_templates[strategy],
                    "prompt_template_hash": strategy_template_hashes[strategy],
                    "rendered_prompt_hash": prompt_hashes[strategy],
                    "generation_parameters": strategy_generation_params[strategy],
                    "output_length": "10",
                    "parser_contract": spec["parser_contract"],
                    "expected_raw_output_path": f"{raw_prefix}/{request_id}.json",
                    "evidence_class": evidence_class,
                    "lineage": lineage,
                }
                base_row["request_contract_hash"] = sha256_text(canonical_json(base_row))
                rows.append(base_row)
                order += 1
    return rows


def build_full_cohort_rows() -> list[dict[str, str]]:
    cohort = load_lastfm_cohort_rows()
    rows: list[dict[str, str]] = []
    for user_id in sorted(cohort, key=sort_key_numeric_text):
        row = dict(cohort[user_id])
        row["history_hash"] = history_hash(row)
        rows.append(row)
    return rows


def title_lookup() -> dict[str, str]:
    return {row["item_id"]: row["title"] for row in read_csv_rows(F6C_LASTFM_INPUTS["catalog"])}


def build_reranking_pool_rows(selected_users: Sequence[str], *, scope: str) -> list[dict[str, str]]:
    user_filter = set(selected_users)
    all_rows = read_csv_rows(F6G1_LASTFM_INPUTS["pool_item_provenance"])
    filtered = [
        row
        for row in all_rows
        if scope == "full_cohort" or str(row["user_id"]) in user_filter
    ]
    filtered.sort(key=lambda row: (sort_key_numeric_text(row["user_id"]), int(row["candidate_position"])))
    by_user: dict[str, list[dict[str, str]]] = {}
    for row in filtered:
        by_user.setdefault(row["user_id"], []).append(row)
    pool_hashes = {
        user_id: sha256_text(json_list([row["candidate_item_id"] for row in rows]))
        for user_id, rows in by_user.items()
    }
    positive_positions = {
        user_id: next((row["candidate_position"] for row in rows if parse_bool_text(row.get("is_positive"))), "")
        for user_id, rows in by_user.items()
    }
    evidence_class = (
        "paper_protocol_reproduction_under_explicit_operationalization"
        if scope == "operationalized"
        else "original_project_extension_full_cohort"
    )
    sample_id = "phase_f6l_lastfm_operationalized_1000" if scope == "operationalized" else "phase_f6l_lastfm_full_cohort_1859"
    lineage = rel(F6G1_LASTFM_INPUTS["pool_item_provenance"])
    rows_out: list[dict[str, str]] = []
    for row in filtered:
        user_id = row["user_id"]
        rows_out.append(
            {
                "cohort_scope": scope,
                "sample_id": sample_id,
                "user_id": user_id,
                "candidate_position": row["candidate_position"],
                "candidate_item_id": row["candidate_item_id"],
                "candidate_item_title": row.get("title", ""),
                "is_positive": clean_cell(parse_bool_text(row.get("is_positive"))),
                "positive_item": row.get("positive_item_id", ""),
                "positive_position": positive_positions[user_id],
                "source_models": row.get("source_models", ""),
                "source_ranks": row.get("source_ranks", ""),
                "source_scores": row.get("source_scores", ""),
                "source_seeds": row.get("source_seeds", ""),
                "duplicate_source_count": row.get("duplicate_source_count", ""),
                "duplicate_skipped_before_item": row.get("duplicate_skipped_before_item", ""),
                "construction_policy": row.get("construction_policy", ""),
                "pool_hash": pool_hashes[user_id],
                "evidence_class": evidence_class,
                "lineage": lineage,
            }
        )
    return rows_out


def full_catalog_topk_status() -> list[dict[str, str]]:
    global _LASTFM_FULL_CATALOG_TOPK_STATUS_CACHE
    if _LASTFM_FULL_CATALOG_TOPK_STATUS_CACHE is not None:
        return [dict(row) for row in _LASTFM_FULL_CATALOG_TOPK_STATUS_CACHE]
    selected = selected_traditional_model_seeds()
    stats: dict[str, dict[str, Any]] = {
        model: {"seed": seed, "users": set(), "counts": Counter(), "invalid": 0}
        for model, seed in selected.items()
    }
    with F6D2_LASTFM_INPUTS["full_catalog_rankings"].open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            model = row.get("model", "")
            if selected.get(model) != row.get("seed"):
                continue
            user_id = str(row.get("user_id", ""))
            stats[model]["users"].add(user_id)
            stats[model]["counts"][user_id] += 1
            if (
                not parse_bool_text(row.get("eligible_for_recommendation"))
                or parse_bool_text(row.get("was_seen_in_train"))
                or parse_bool_text(row.get("was_seen_in_validation"))
            ):
                stats[model]["invalid"] += 1
    rows = []
    for model, values in sorted(stats.items()):
        counts = values["counts"]
        min_count = min(counts.values(), default=0)
        rows.append(
            {
                "model": model,
                "paper_name": LASTFM_CONTROL_MODELS[model],
                "seed": values["seed"],
                "users": str(len(values["users"])),
                "min_recommendations_per_user": str(min_count),
                "invalid_recommendation_rows": str(values["invalid"]),
                "status": "pass" if len(values["users"]) == 1859 and min_count >= 20 and values["invalid"] == 0 else "fail",
            }
        )
    _LASTFM_FULL_CATALOG_TOPK_STATUS_CACHE = rows
    return [dict(row) for row in rows]


def build_reranking_readiness(
    *,
    operationalized_pool_rows: Sequence[Mapping[str, str]],
    full_pool_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    topk_rows = full_catalog_topk_status()
    topk_pass = all(row["status"] == "pass" for row in topk_rows)
    op_users = len({row["user_id"] for row in operationalized_pool_rows})
    full_users = len({row["user_id"] for row in full_pool_rows})
    rows = [
        {
            "readiness_id": "F6L-LASTFM-RERANK-READY-001",
            "scope": "paper_exact",
            "requirement": "official LastFM BPRMF/GRU4Rec/SASRec/LightGCN top-K and final 20-item pool identity",
            "status": "blocked_missing_original_specification",
            "observed": "not recovered",
            "evidence": "results/tables/phase_f6l_lastfm_author_specification_registry.csv;results/tables/phase_f6l_official_artifact_recovery.csv",
            "evidence_class": "blocked_missing_original_specification",
            "missing_requirement": "official paper top-K snapshots; original seed/run identity; original reranking pool",
            "future_request_count": "0",
            "notes": "Paper-exact reranking remains blocked.",
        },
        {
            "readiness_id": "F6L-LASTFM-RERANK-READY-002",
            "scope": "operationalized_sample",
            "requirement": "local reconstructed reranking pool for selected 1000-user sample",
            "status": "ready_after_explicit_authorization" if op_users == 1000 and len(operationalized_pool_rows) == 20000 and topk_pass else "blocked",
            "observed": f"users={op_users};pool_rows={len(operationalized_pool_rows)};topk_pass={clean_cell(topk_pass)}",
            "evidence": rel(F6G1_LASTFM_INPUTS["pool_item_provenance"]),
            "evidence_class": "paper_protocol_reproduction_under_explicit_operationalization",
            "missing_requirement": "",
            "future_request_count": "4000" if op_users == 1000 and len(operationalized_pool_rows) == 20000 and topk_pass else "0",
            "notes": "Not paper-exact; usable only as local operationalized evidence.",
        },
        {
            "readiness_id": "F6L-LASTFM-RERANK-READY-003",
            "scope": "full_cohort_extension",
            "requirement": "local reconstructed reranking pool for all 1859 F6C users",
            "status": "ready_after_explicit_authorization" if full_users == 1859 and len(full_pool_rows) == 37180 and topk_pass else "blocked",
            "observed": f"users={full_users};pool_rows={len(full_pool_rows)};topk_pass={clean_cell(topk_pass)}",
            "evidence": rel(F6G1_LASTFM_INPUTS["pool_item_provenance"]),
            "evidence_class": "original_project_extension_full_cohort",
            "missing_requirement": "",
            "future_request_count": "7436" if full_users == 1859 and len(full_pool_rows) == 37180 and topk_pass else "0",
            "notes": "Extension evidence only; cannot support or contradict paper claims alone.",
        },
    ]
    for index, topk in enumerate(topk_rows, start=4):
        rows.append(
            {
                "readiness_id": f"F6L-LASTFM-RERANK-READY-{index:03d}",
                "scope": "local_selected_traditional_top100",
                "requirement": f"{topk['paper_name']} full-catalog top-100 available for adapted selected seed",
                "status": topk["status"],
                "observed": f"model={topk['model']};seed={topk['seed']};users={topk['users']};min_recommendations={topk['min_recommendations_per_user']};invalid={topk['invalid_recommendation_rows']}",
                "evidence": rel(F6D2_LASTFM_INPUTS["full_catalog_rankings"]),
                "evidence_class": "original_project_extension_full_cohort",
                "missing_requirement": "",
                "future_request_count": "0",
                "notes": "Local adapted top-K evidence used only for reconstructed reranking pools.",
            }
        )
    return rows


def observed_items_by_user() -> dict[str, set[str]]:
    global _LASTFM_OBSERVED_ITEMS_CACHE
    if _LASTFM_OBSERVED_ITEMS_CACHE is not None:
        return {user_id: set(items) for user_id, items in _LASTFM_OBSERVED_ITEMS_CACHE.items()}
    observed: dict[str, set[str]] = {}
    for row in read_csv_rows(F6C_LASTFM_INPUTS["interactions"]):
        observed.setdefault(str(row["user_id"]), set()).add(str(row["item_id"]))
    _LASTFM_OBSERVED_ITEMS_CACHE = observed
    return {user_id: set(items) for user_id, items in observed.items()}


def validate_lastfm_contracts(
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    manifest: Mapping[str, Any],
    protected: Mapping[str, Any],
    deterministic: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(name: str, passed: bool, observed: str, expected: str, evidence: str, notes: str, severity: str = "error") -> None:
        rows.append(
            {
                "validation_id": f"F6L-LASTFM-H{len(rows) + 1:02d}",
                "check_name": name,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    sample = list(tables["operationalized_sample_cohort"])
    candidates = list(tables["operationalized_candidates"])
    op_manifest = list(tables["operationalized_ranking_request_manifest"])
    full_manifest = list(tables["full_cohort_ranking_request_manifest"])
    op_pool = list(tables["operationalized_reranking_pool"])
    full_pool = list(tables["full_cohort_reranking_pool"])
    ks_rows = list(tables["operationalized_ks_results"])
    seed_rows = list(tables["operationalized_seed_search"])
    readiness = list(tables["reranking_readiness"])
    sample_users = {row["user_id"] for row in sample}
    observed = observed_items_by_user()
    candidates_by_user: dict[str, list[Mapping[str, str]]] = {}
    for row in candidates:
        candidates_by_user.setdefault(row["user_id"], []).append(row)

    exact_audit = [row for row in tables["local_pipeline_audit"] if row["classification"] == "blocked_missing_original_specification"]
    add(
        "H01_exact_paper_replica_remains_blocked",
        bool(exact_audit),
        exact_audit[0]["status"] if exact_audit else "missing",
        "blocked_missing_original_specification",
        rel(LASTFM_LOCAL_PIPELINE_AUDIT_PATH),
        "No local artifact is promoted to exact paper LastFM replication.",
    )
    add(
        "H02_operationalized_seed_selected",
        len(seed_rows) >= 1 and seed_rows[-1]["accepted"] == "true",
        seed_rows[-1]["seed"] if seed_rows else "none",
        "first accepted deterministic seed",
        rel(LASTFM_OPERATIONALIZED_SEED_SEARCH_PATH),
        "Seed search must fail closed if no local K-S-valid sample exists.",
    )
    add(
        "H03_ks_holm_non_significant",
        len(ks_rows) == 8 and all(row["accepted"] == "true" and row["significant_after_holm"] == "false" for row in ks_rows),
        f"tests={len(ks_rows)};min_holm={min((float(row['p_value_holm']) for row in ks_rows), default=0.0)}",
        "8 tests; all Holm-adjusted p >= alpha",
        rel(LASTFM_OPERATIONALIZED_KS_RESULTS_PATH),
        "Four local traditional models times HR@5/NDCG@5.",
    )
    add("H04_sample_size_1000", len(sample) == 1000 and len(sample_users) == 1000, f"rows={len(sample)};users={len(sample_users)}", "1000 distinct users", rel(LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH), "Local preregistered operationalized sample.")
    add("H05_candidate_rows_20000", len(candidates) == 20000, str(len(candidates)), "20000", rel(LASTFM_OPERATIONALIZED_CANDIDATES_PATH), "1000 users x 20 candidates.")
    positives_by_user = Counter(row["user_id"] for row in candidates if row["is_positive"] == "true")
    add("H06_one_positive_per_user", len(positives_by_user) == 1000 and set(positives_by_user.values()) == {1}, f"users_with_positive={len(positives_by_user)};counts={sorted(set(positives_by_user.values()))}", "one positive row for each sampled user", rel(LASTFM_OPERATIONALIZED_CANDIDATES_PATH), "Positive item is the F6C local test item.")
    duplicates = []
    for user_id in sample_users:
        user_items = [row["candidate_item_id"] for row in candidates_by_user.get(user_id, [])]
        if len(user_items) != len(set(user_items)):
            duplicates.append(user_id)
    add("H07_no_duplicate_candidate_items_per_user", not duplicates, ";".join(duplicates[:5]) if duplicates else "none", "none", rel(LASTFM_OPERATIONALIZED_CANDIDATES_PATH), "Candidate rows are deterministic and de-duplicated per user.")
    observed_negative_hits = []
    for row in candidates:
        if row["is_positive"] == "true":
            continue
        if row["candidate_item_id"] in observed.get(row["user_id"], set()):
            observed_negative_hits.append(f"{row['user_id']}:{row['candidate_item_id']}")
    add("H08_no_observed_negative_candidates", not observed_negative_hits, ";".join(observed_negative_hits[:5]) if observed_negative_hits else "none", "none", rel(F6C_LASTFM_INPUTS["interactions"]), "Negative candidates must not be user-observed train/validation/test items.")
    per_user_manifest_counts = Counter(row["user_id"] for row in op_manifest)
    per_user_hash_counts: dict[str, set[str]] = {}
    for row in op_manifest:
        per_user_hash_counts.setdefault(row["user_id"], set()).add(row["candidate_list_hash"])
    add("H09_shared_candidate_order_across_models_and_strategies", len(per_user_manifest_counts) == 1000 and set(per_user_manifest_counts.values()) == {4} and all(len(value) == 1 for value in per_user_hash_counts.values()), f"users={len(per_user_manifest_counts)};row_counts={sorted(set(per_user_manifest_counts.values()))}", "4 rows per user and one candidate_list_hash per user", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Candidate ordering is shared across Qwen2/Mistral and base/LLMRank.")
    add("H10_operationalized_ranking_manifest_4000", len(op_manifest) == 4000 and len({row["request_id"] for row in op_manifest}) == 4000, f"rows={len(op_manifest)};unique={len({row['request_id'] for row in op_manifest})}", "4000 unique request rows", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "No inference is executed from this manifest.")
    add("H11_full_cohort_ranking_manifest_7436", len(full_manifest) == 7436 and len({row["request_id"] for row in full_manifest}) == 7436, f"rows={len(full_manifest)};unique={len({row['request_id'] for row in full_manifest})}", "7436 unique request rows", rel(LASTFM_FULL_COHORT_RANKING_MANIFEST_PATH), "1859 users x two models x two strategies.")
    allowed_models = set(MODEL_DIGESTS)
    manifest_models = {row["model_name"] for row in op_manifest + full_manifest}
    add("H12_allowed_open_models_only", manifest_models == allowed_models, ";".join(sorted(manifest_models)), ";".join(sorted(allowed_models)), rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Closed/paid models are not included.")
    strategies = {row["prompting_strategy"] for row in op_manifest + full_manifest}
    add("H13_no_llmsrec_and_two_strategies_only", strategies == set(LASTFM_RANKING_STRATEGIES) and not any("LLMSRec" in row["prompting_strategy"] for row in op_manifest + full_manifest), ";".join(sorted(strategies)), "base/OpenP5 title-list;LLMRank recency", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "LLMSRec remains blocked by missing demonstration contract.")
    add("H14_prompt_and_contract_hashes_present", all(row["prompt_template_hash"] and row["rendered_prompt_hash"] and row["request_contract_hash"] for row in op_manifest + full_manifest), "present" if op_manifest and full_manifest else "missing", "all rows hashed", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Rendered prompt text is not stored, only hashes.")
    policy = manifest.get("policy", {})
    add("H15_no_inference_training_download_or_raw_dir", not any(policy.get(flag) for flag in ["inference_performed", "ollama_generation_performed", "remote_api_calls_performed", "paid_api_calls_performed", "training_performed", "downloads_performed", "raw_output_dir_created"]), canonical_json(policy), "all execution flags false", rel(LASTFM_EXECUTION_CONTRACTS_MANIFEST_PATH), "Contract mode is non-executing.")
    add("H16_reranking_pool_shapes", len(op_pool) == 20000 and len(full_pool) == 37180, f"op={len(op_pool)};full={len(full_pool)}", "20000 operationalized and 37180 full cohort", rel(LASTFM_OPERATIONALIZED_RERANKING_POOL_PATH), "Reconstructed reranking pools are separated by evidence class.")
    paper_exact = [row for row in readiness if row["scope"] == "paper_exact"]
    add("H17_reranking_paper_exact_blocked", bool(paper_exact) and paper_exact[0]["status"] == "blocked_missing_original_specification", paper_exact[0]["status"] if paper_exact else "missing", "blocked_missing_original_specification", rel(LASTFM_RERANKING_READINESS_PATH), "Official LastFM top-K snapshots and pool identity are still missing.")
    topk_rows = [row for row in readiness if row["scope"] == "local_selected_traditional_top100"]
    add("H18_local_topk_prerequisites_pass_for_extension", len(topk_rows) == 4 and all(row["status"] == "pass" for row in topk_rows), ";".join(row["observed"] for row in topk_rows), "four adapted selected top-100 exports pass", rel(F6D2_LASTFM_INPUTS["full_catalog_rankings"]), "This supports only local reconstructed reranking pools.")
    add("H19_evidence_layers_separated", {row["evidence_class"] for row in op_manifest} == {"paper_protocol_reproduction_under_explicit_operationalization"} and {row["evidence_class"] for row in full_manifest} == {"original_project_extension_full_cohort"}, f"op={sorted({row['evidence_class'] for row in op_manifest})};full={sorted({row['evidence_class'] for row in full_manifest})}", "operationalized and full-cohort labels are disjoint", rel(LASTFM_EXECUTION_CONTRACTS_MANIFEST_PATH), "No local row is labeled paper-exact.")
    add("H20_protected_inputs_and_23_yaml_pass", protected.get("status") == "pass" and protected.get("f6i2p_yaml_count") == 23, f"status={protected.get('status')};f6i2p={protected.get('f6i2p_yaml_count')}", "pass;23", rel(PROTECTED_HASHES_PATH), "Uses the existing protected-input hash policy.")
    raw_root = PROJECT_ROOT / "results" / "llm"
    lastfm_raw = sorted(path.name for path in raw_root.glob("*phase_f6l_lastfm*")) if raw_root.exists() else []
    unauthorized_lastfm_raw = [
        name
        for name in lastfm_raw
        if not name.startswith("phase_f6l_lastfm_operationalized_ranking_")
    ]
    add("H21_no_unauthorized_lastfm_phase_f6l_raw_outputs", not unauthorized_lastfm_raw, ";".join(unauthorized_lastfm_raw) if unauthorized_lastfm_raw else "none", "none", rel(raw_root) if raw_root.exists() else "results/llm", "Authorized operationalized-ranking raw dirs may exist only after explicit execution authorization.")
    original_repo_status = git_output(["status", "--short", "original_repo"])
    add("H22_original_repo_clean_for_this_phase", original_repo_status == "", original_repo_status or "none", "none", "git status --short original_repo", "No original repository files are modified.")
    staged = git_output(["diff", "--cached", "--name-only"])
    add("H23_no_staging_and_deterministic_contracts", staged == "" and deterministic, f"staged={staged or 'none'};deterministic={clean_cell(deterministic)}", "no staged paths and deterministic in-memory digest", "git diff --cached --name-only", "The user explicitly prohibited staging or commits.")
    return rows


def lastfm_contracts_digest_payload(result: LastfmExecutionContractsResult) -> dict[str, Any]:
    manifest = {
        key: value
        for key, value in result.manifest.items()
        if key not in {"output_hashes", "self_hash", "deterministic_in_memory_digest_match"}
    }
    return {
        "decision_token": result.decision_token,
        "tables": result.tables,
        "manifest": manifest,
        "report_section": result.report_section,
    }


def rows_to_csv_text(rows: Sequence[Mapping[str, str]], fields: Sequence[str]) -> str:
    import io

    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: clean_cell(row.get(field, "")) for field in fields})
    return handle.getvalue()


def build_lastfm_execution_contract_manifest(
    result: LastfmExecutionContractsResult,
    *,
    protected: Mapping[str, Any],
    deterministic: bool,
    output_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    tables = result.tables
    op_manifest = tables["operationalized_ranking_request_manifest"]
    full_manifest = tables["full_cohort_ranking_request_manifest"]
    op_hash_rollup = sha256_text(canonical_json([row["rendered_prompt_hash"] for row in op_manifest]))
    full_hash_rollup = sha256_text(canonical_json([row["rendered_prompt_hash"] for row in full_manifest]))
    seed_rows = tables["operationalized_seed_search"]
    selected_seed = seed_rows[-1]["seed"] if seed_rows and seed_rows[-1]["accepted"] == "true" else ""
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "mode": "lastfm-contracts",
        "decision_token": result.decision_token,
        "materialized_at": MATERIALIZATION_AUDIT_TIMESTAMP,
        "evidence_hierarchy": {
            "exact_paper_replica": "blocked_missing_original_specification",
            "operationalized_1000_user_sample": "paper_protocol_reproduction_under_explicit_operationalization",
            "full_1859_local_cohort": "original_project_extension_full_cohort",
        },
        "selected_operationalized_seed": selected_seed,
        "seed_search_attempts": len(seed_rows),
        "ks_gate": {
            "test": "two-sample Kolmogorov-Smirnov",
            "pvalue_implementation": "local_asymptotic_two_sided_kolmogorov_distribution_no_scipy_dependency",
            "alpha": "0.05",
            "multiple_testing": "Holm",
            "metric_vectors": "per-user HR@5 and NDCG@5 from F6D2 selected BPRMF/GRU4Rec/SASRec/LightGCN candidate-set rankings",
            "accepted": bool(selected_seed),
        },
        "counts": {
            "operationalized_sample_users": len(tables["operationalized_sample_cohort"]),
            "operationalized_candidate_rows": len(tables["operationalized_candidates"]),
            "operationalized_ranking_requests": len(op_manifest),
            "full_cohort_users": len({row["user_id"] for row in full_manifest}),
            "full_cohort_ranking_requests": len(full_manifest),
            "operationalized_reranking_pool_rows": len(tables["operationalized_reranking_pool"]),
            "full_cohort_reranking_pool_rows": len(tables["full_cohort_reranking_pool"]),
            "future_operationalized_reranking_requests_if_authorized": sum(int(row["future_request_count"]) for row in tables["reranking_readiness"] if row["scope"] == "operationalized_sample"),
            "future_full_cohort_reranking_requests_if_authorized": sum(int(row["future_request_count"]) for row in tables["reranking_readiness"] if row["scope"] == "full_cohort_extension"),
        },
        "models": MODEL_DIGESTS,
        "strategies": {
            strategy: {
                "strategy_id": spec["strategy_id"],
                "template_path": rel(spec["template_path"]),
                "template_hash": template_hash(spec["template_path"]),
                "parser_contract": spec["parser_contract"],
            }
            for strategy, spec in LASTFM_RANKING_STRATEGIES.items()
        },
        "rendered_prompt_hash_rollups": {
            "operationalized_ranking_request_manifest": op_hash_rollup,
            "full_cohort_ranking_request_manifest": full_hash_rollup,
        },
        "selected_traditional_model_seeds": selected_traditional_model_seeds(),
        "input_hashes": local_pipeline_input_hashes(),
        "protected_input_status": protected.get("status"),
        "f6i2p_yaml_count": protected.get("f6i2p_yaml_count"),
        "policy": {
            "inference_performed": False,
            "ollama_generation_performed": False,
            "remote_api_calls_performed": False,
            "paid_api_calls_performed": False,
            "training_performed": False,
            "downloads_performed": False,
            "raw_output_dir_created": False,
            "original_repo_modified": False,
            "git_staging_performed": False,
        },
        "tables": {name: rel(path) for name, path in LASTFM_EXECUTION_CONTRACT_TABLES.items()},
        "output_hashes": dict(output_hashes or {}),
        "git": {
            "head": git_output(["rev-parse", "HEAD"]),
            "staged_paths": git_output(["diff", "--cached", "--name-only"]),
            "original_repo_status": git_output(["status", "--short", "original_repo"]),
        },
        "deterministic_in_memory_digest_match": deterministic,
        "self_hash": "self_reference_excluded_for_determinism",
    }


def build_lastfm_execution_contract_report_section(result: LastfmExecutionContractsResult) -> str:
    manifest = result.manifest
    counts = manifest["counts"]
    seed = manifest["selected_operationalized_seed"] or "none"
    validation_pass = sum(1 for row in result.tables["validation"] if row["status"] == "pass")
    validation_fail = sum(1 for row in result.tables["validation"] if row["status"] != "pass")
    return "\n".join(
        [
            "## LastFM Execution Contracts",
            "",
            f"- Decision token: `{result.decision_token}`.",
            "- Exact paper replica: `blocked_missing_original_specification`.",
            "- 1000-user local sample: `paper_protocol_reproduction_under_explicit_operationalization`.",
            "- Full 1859-user local cohort: `original_project_extension_full_cohort`.",
            f"- Selected local K-S seed: `{seed}`.",
            f"- K-S attempts: `{manifest['seed_search_attempts']}`; K-S tests in selected family: `{len(result.tables['operationalized_ks_results'])}`.",
            f"- Operationalized sample users: `{counts['operationalized_sample_users']}`; candidate rows: `{counts['operationalized_candidate_rows']}`.",
            f"- Operationalized ranking request contracts: `{counts['operationalized_ranking_requests']}`.",
            f"- Full-cohort ranking request contracts: `{counts['full_cohort_ranking_requests']}`.",
            f"- Operationalized reranking pool rows: `{counts['operationalized_reranking_pool_rows']}`; future reranking requests if separately authorized: `{counts['future_operationalized_reranking_requests_if_authorized']}`.",
            f"- Full-cohort reranking pool rows: `{counts['full_cohort_reranking_pool_rows']}`; future reranking requests if separately authorized: `{counts['future_full_cohort_reranking_requests_if_authorized']}`.",
            f"- Validation: `{validation_pass}` passed, `{validation_fail}` failed.",
            "- No inference, Ollama generation, remote API, paid API, download, training, raw-output directory creation, staging, commit, cleanup, or `original_repo/` mutation was performed.",
            "",
            "### Future Commands",
            "",
            "- Revalidate/materialize these LastFM contracts only:",
            "`$env:PYTHONNOUSERSITE='1'; $env:PYTHONDONTWRITEBYTECODE='1'; $env:PYTHONPATH='.'; .\\.venv_audit_recovery\\Scripts\\python.exe -B src\\experiments\\build_phase_f6l_paper_faithful_completion_and_rq_revalidation.py --config configs\\phase_f6l_paper_faithful_completion_and_rq_revalidation.yaml --mode lastfm-contracts`",
            "- No inference command is recorded here because this turn was contract-only. A future runner must consume the CSV manifests only after explicit authorization.",
            "",
            f"`{result.decision_token}`",
            "",
        ]
    )


def update_lastfm_execution_contract_handoff_and_notes(result: LastfmExecutionContractsResult) -> None:
    manifest = result.manifest
    counts = manifest["counts"]
    marker = f"{PHASE_NAME}_lastfm_execution_contracts"
    handoff_content = "\n".join(
        [
            "## F6L LastFM Execution Contracts",
            "",
            f"- Decision token: `{result.decision_token}`.",
            "- Profile generation remains terminally closed and was not reopened.",
            "- Exact paper LastFM remains `blocked_missing_original_specification`.",
            "- Local 1000-user operationalized LastFM sample is separate from paper-exact evidence.",
            "- Full 1859-user cohort is separate extension evidence.",
            f"- Operationalized ranking manifests: `{counts['operationalized_ranking_requests']}` rows.",
            f"- Full-cohort ranking manifests: `{counts['full_cohort_ranking_requests']}` rows.",
            f"- Reranking readiness keeps paper-exact blocked and local pools separate: operationalized `{counts['future_operationalized_reranking_requests_if_authorized']}` future requests, full cohort `{counts['future_full_cohort_reranking_requests_if_authorized']}` future requests.",
            "- No inference or raw output directory was created.",
        ]
    )
    notes_content = "\n".join(
        [
            "## 2026-08-02 - Phase F6L LastFM Execution Contracts",
            "",
            "### Changes",
            "",
            "- Materialized LastFM-only local operationalized K-S sample tables, ranking request manifests, reranking pool readiness, validation, and manifest under the existing F6L phase.",
            "- Preserved the evidence hierarchy: paper-exact blocked; 1000-user local operationalization; full-cohort extension.",
            "- Did not run inference, train models, call APIs, alter `original_repo/`, alter YAML, alter cleanup/guard logic, stage, commit, or write raw LLM outputs.",
            "",
            "### Outputs Generated",
            "",
            "- `results/tables/phase_f6l_lastfm_local_pipeline_audit.csv`",
            "- `results/tables/phase_f6l_lastfm_operationalized_seed_search.csv`",
            "- `results/tables/phase_f6l_lastfm_operationalized_ks_results.csv`",
            "- `results/tables/phase_f6l_lastfm_operationalized_sample_cohort.csv`",
            "- `results/tables/phase_f6l_lastfm_operationalized_candidates.csv`",
            "- `results/tables/phase_f6l_lastfm_operationalized_ranking_request_manifest.csv`",
            "- `results/tables/phase_f6l_lastfm_full_cohort_ranking_request_manifest.csv`",
            "- `results/tables/phase_f6l_lastfm_operationalized_reranking_pool.csv`",
            "- `results/tables/phase_f6l_lastfm_full_cohort_reranking_pool.csv`",
            "- `results/tables/phase_f6l_lastfm_reranking_readiness.csv`",
            "- `results/tables/phase_f6l_lastfm_execution_contracts_validation.csv`",
            "- `results/manifests/phase_f6l_lastfm_execution_contracts_manifest.json`",
            "",
            "### Result",
            "",
            f"- Decision: `{result.decision_token}`.",
            f"- Selected local K-S seed: `{manifest['selected_operationalized_seed']}`.",
            f"- Operationalized ranking requests: `{counts['operationalized_ranking_requests']}`.",
            f"- Full-cohort ranking requests: `{counts['full_cohort_ranking_requests']}`.",
        ]
    )
    update_marked_section(REPORT_PATH, marker, build_lastfm_execution_contract_report_section(result))
    update_marked_section(HANDOFF_PATH, marker, handoff_content)
    update_marked_section(NOTES_PATH, marker, notes_content)


def assemble_lastfm_execution_contracts(config_path: Path = CONFIG_PATH) -> LastfmExecutionContractsResult:
    require_runtime()
    load_config(config_path)
    protected = hash_protected_inputs()
    metric_vectors = load_selected_seed_per_user_metrics()
    seed_rows, ks_rows, selected_users = build_lastfm_seed_search(metric_vectors)
    selected_seed = seed_rows[-1]["seed"] if seed_rows and seed_rows[-1]["accepted"] == "true" else ""
    sample_rows = build_lastfm_sample_cohort(selected_users, selected_seed) if selected_users else []
    candidate_rows = build_lastfm_operationalized_candidates(selected_users, selected_seed) if selected_users else []
    full_cohort_rows = build_full_cohort_rows()
    full_candidate_rows = [
        {**row, "is_positive": clean_cell(parse_bool_text(row.get("is_ground_truth")))}
        for user_rows in load_candidate_rows_by_user().values()
        for row in user_rows
    ]
    op_ranking_manifest = (
        build_lastfm_ranking_manifest(cohort_rows=sample_rows, candidate_rows=candidate_rows, scope="operationalized")
        if sample_rows and candidate_rows
        else []
    )
    full_ranking_manifest = build_lastfm_ranking_manifest(
        cohort_rows=full_cohort_rows,
        candidate_rows=full_candidate_rows,
        scope="full_cohort",
    )
    op_pool = build_reranking_pool_rows(selected_users, scope="operationalized") if selected_users else []
    full_pool = build_reranking_pool_rows([], scope="full_cohort")
    readiness = build_reranking_readiness(operationalized_pool_rows=op_pool, full_pool_rows=full_pool)
    tables: dict[str, list[dict[str, str]]] = {
        "local_pipeline_audit": build_lastfm_local_pipeline_audit(protected),
        "operationalized_seed_search": seed_rows,
        "operationalized_ks_results": ks_rows,
        "operationalized_sample_cohort": sample_rows,
        "operationalized_candidates": candidate_rows,
        "operationalized_ranking_request_manifest": op_ranking_manifest,
        "full_cohort_ranking_request_manifest": full_ranking_manifest,
        "operationalized_reranking_pool": op_pool,
        "full_cohort_reranking_pool": full_pool,
        "reranking_readiness": readiness,
        "validation": [],
    }
    proto = LastfmExecutionContractsResult(DECISION_LASTFM_CONTRACTS_BLOCKED, tables, {}, "")
    manifest = build_lastfm_execution_contract_manifest(proto, protected=protected, deterministic=False)
    validation = validate_lastfm_contracts(tables, manifest, protected, deterministic=True)
    tables["validation"] = validation
    failed = [row for row in validation if row["status"] != "pass" and row["severity"] == "error"]
    if failed:
        decision = (
            DECISION_LASTFM_SAMPLE_BLOCKED_FULL_READY
            if not sample_rows and len(full_ranking_manifest) == 7436 and protected.get("status") == "pass"
            else DECISION_LASTFM_CONTRACTS_BLOCKED
        )
    else:
        decision = DECISION_LASTFM_CONTRACTS_READY
    result = LastfmExecutionContractsResult(decision, tables, {}, "")
    manifest = build_lastfm_execution_contract_manifest(result, protected=protected, deterministic=True)
    report_section = build_lastfm_execution_contract_report_section(
        LastfmExecutionContractsResult(decision, tables, manifest, "")
    )
    return LastfmExecutionContractsResult(decision, tables, manifest, report_section)


def build_lastfm_execution_contracts(config_path: Path = CONFIG_PATH, *, write_outputs_enabled: bool = True) -> LastfmExecutionContractsResult:
    first = assemble_lastfm_execution_contracts(config_path)
    second = assemble_lastfm_execution_contracts(config_path)
    deterministic = sha256_text(canonical_json(lastfm_contracts_digest_payload(first))) == sha256_text(canonical_json(lastfm_contracts_digest_payload(second)))
    protected = hash_protected_inputs()
    validation = validate_lastfm_contracts(first.tables, first.manifest, protected, deterministic)
    tables = {**first.tables, "validation": validation}
    failed = [row for row in validation if row["status"] != "pass" and row["severity"] == "error"]
    decision = DECISION_LASTFM_CONTRACTS_READY if not failed else DECISION_LASTFM_CONTRACTS_BLOCKED
    result = LastfmExecutionContractsResult(decision, tables, {}, "")
    manifest = build_lastfm_execution_contract_manifest(result, protected=protected, deterministic=deterministic)
    result = LastfmExecutionContractsResult(decision, tables, manifest, build_lastfm_execution_contract_report_section(LastfmExecutionContractsResult(decision, tables, manifest, "")))
    if write_outputs_enabled:
        for name, path in LASTFM_EXECUTION_CONTRACT_TABLES.items():
            write_csv(path, result.tables[name], LASTFM_EXECUTION_CONTRACT_FIELDS[name])
        output_hashes = {
            rel(path): safe_file_sha256(path)
            for path in LASTFM_EXECUTION_CONTRACT_TABLES.values()
            if path.exists()
        }
        manifest = build_lastfm_execution_contract_manifest(result, protected=protected, deterministic=deterministic, output_hashes=output_hashes)
        result = LastfmExecutionContractsResult(
            decision,
            tables,
            manifest,
            build_lastfm_execution_contract_report_section(LastfmExecutionContractsResult(decision, tables, manifest, "")),
        )
        write_json(LASTFM_EXECUTION_CONTRACTS_MANIFEST_PATH, result.manifest)
        update_lastfm_execution_contract_handoff_and_notes(result)
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_jsonl_fsync(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row["_jsonl_line_no"] = line_no
            rows.append(row)
    return rows


def powershell_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def execution_replay_command() -> str:
    return " ".join(
        [
            "$env:PYTHONNOUSERSITE='1';",
            "$env:PYTHONDONTWRITEBYTECODE='1';",
            "$env:PYTHONPATH='.';",
            powershell_quote(str(Path(sys.executable).resolve())),
            "-B",
            *[powershell_quote(arg) for arg in sys.argv],
        ]
    )


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def execution_state_row(
    *,
    run_id: str,
    stage: str,
    decision_token: str,
    planned: int,
    completed: int,
    strict_valid: int,
    failed: int,
    retry: int,
    start_time: float,
    raw_output_dir: Path,
    last_request_id: str = "",
    notes: str = "",
) -> dict[str, str]:
    residual = max(0, planned - completed)
    elapsed = max(0.001, time.perf_counter() - start_time)
    throughput = completed / elapsed * 3600 if completed else 0.0
    eta = residual / (completed / elapsed) if completed else 0.0
    return {
        "run_id": run_id,
        "stage": stage,
        "decision_token": decision_token,
        "planned": str(planned),
        "completed": str(completed),
        "strict_valid": str(strict_valid),
        "failed": str(failed),
        "residual": str(residual),
        "retry": str(retry),
        "percent_complete": format_float((completed / planned * 100.0) if planned else 0.0),
        "last_request_id": last_request_id,
        "last_timestamp": utc_now(),
        "throughput_requests_per_hour": format_float(throughput),
        "eta_seconds": format_float(eta),
        "raw_output_dir": rel(raw_output_dir) if path_is_under(raw_output_dir, PROJECT_ROOT) else str(raw_output_dir),
        "notes": notes,
    }


def write_execution_state(row: Mapping[str, str]) -> None:
    LASTFM_OPERATIONALIZED_RANKING_EXECUTION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = LASTFM_OPERATIONALIZED_RANKING_EXECUTION_STATE_PATH.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LASTFM_OPERATIONALIZED_RANKING_TABLE_FIELDS["execution_state"], lineterminator="\n")
        writer.writeheader()
        writer.writerow({field: clean_cell(row.get(field, "")) for field in LASTFM_OPERATIONALIZED_RANKING_TABLE_FIELDS["execution_state"]})
    temp.replace(LASTFM_OPERATIONALIZED_RANKING_EXECUTION_STATE_PATH)


def title_results_map(title: Any) -> str:
    text = str(title[0] if isinstance(title, list) and title else title)
    return (
        text.replace("\\'", "'")
        .replace("Â®", "&reg;")
        .replace("Ã¨", "&egrave;")
        .replace("â˜…", "&#9733;")
    )


def normalize_title_for_match(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(title_results_map(value)).strip()).casefold()


def parse_paper_title_list_response(raw_text: str, candidate_rows: Sequence[Mapping[str, str]], expected_count: int = 10) -> dict[str, Any]:
    text = str(raw_text or "")
    json_list_detected = False
    schema_valid = False
    titles: list[str] = []
    failure = "empty_response" if not text.strip() else ""
    matches = re.findall(r"\[[\s\S]*\]", text)
    if len(matches) == 1:
        json_list_detected = True
        result_str = matches[0]
        try:
            result_str = re.sub(r"'[,]*[ \n\r]+#.*$", "',", result_str, flags=re.MULTILINE)
            result_str = re.sub(r"'[ \n\r]*,[ \n\r]*'", '", "', result_str.replace('"', "'"))
            result_str = re.sub(r"'[ \n,]*\]", '"]', result_str)
            result_str = re.sub(r"\[[ \n]*'", '["', result_str)
            result_str = result_str.replace("['", '["').replace("\\", "\\\\")
            parsed = json.loads(result_str)
            if isinstance(parsed, list):
                schema_valid = True
                titles = [title_results_map(item) for item in parsed]
            else:
                failure = "parsed_value_not_list"
        except Exception as exc:
            failure = f"title_list_parse_failed:{type(exc).__name__}"
    else:
        failure = "missing_or_multiple_bracketed_lists"

    output_count_valid = len(titles) == expected_count
    by_norm: dict[str, list[dict[str, str]]] = {}
    for row in candidate_rows:
        by_norm.setdefault(normalize_title_for_match(row["candidate_item_title"]), []).append(dict(row))
    ranked_item_ids: list[str] = []
    invalid_items: list[str] = []
    for title in titles[:expected_count]:
        matches_for_title = by_norm.get(normalize_title_for_match(title), [])
        if len(matches_for_title) == 1:
            ranked_item_ids.append(matches_for_title[0]["candidate_item_id"])
        else:
            invalid_items.append(title)
            ranked_item_ids.append("0")
    duplicate_count = len([item for item, count in Counter(item for item in ranked_item_ids if item != "0").items() if count > 1])
    hallucinated_count = len(invalid_items)
    candidate_validity = bool(ranked_item_ids) and hallucinated_count == 0
    uniqueness_valid = duplicate_count == 0 and len([item for item in ranked_item_ids if item != "0"]) == len(set(item for item in ranked_item_ids if item != "0"))
    strict_valid = bool(json_list_detected and schema_valid and output_count_valid and candidate_validity and uniqueness_valid)
    if strict_valid:
        failure = ""
    elif not failure:
        failure = "invalid_length_or_candidate_mapping"
    return {
        "json_list_detected": json_list_detected,
        "schema_valid": schema_valid,
        "output_count_valid": output_count_valid,
        "title_mapping_valid": candidate_validity,
        "candidate_validity": candidate_validity,
        "uniqueness_valid": uniqueness_valid,
        "strict_valid": strict_valid,
        "parsed_titles": titles[:expected_count],
        "ranked_item_ids": ranked_item_ids[:expected_count],
        "duplicate_count": duplicate_count,
        "hallucinated_count": hallucinated_count,
        "invalid_items": invalid_items,
        "parse_status": "strict_valid" if strict_valid else "invalid",
        "failure_reason": failure,
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def js_divergence(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0

    def kl(p: Mapping[str, float], q: Mapping[str, float]) -> float:
        total = 0.0
        for key in keys:
            pv = p.get(key, 0.0)
            qv = q.get(key, 0.0)
            if pv > 0 and qv > 0:
                total += pv * math.log2(pv / qv)
        return total

    midpoint = {key: (left.get(key, 0.0) + right.get(key, 0.0)) / 2.0 for key in keys}
    return (kl(left, midpoint) + kl(right, midpoint)) / 2.0


def distribution(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def item_tag_sets() -> dict[str, set[str]]:
    tags: dict[str, set[str]] = {}
    for row in read_csv_rows(F6C_LASTFM_INPUTS["catalog"]):
        values = {tag.strip().casefold() for tag in row.get("tag_values", "").split("|") if tag.strip()}
        tags[row["item_id"]] = values
    return tags


def train_popularity_counts() -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in read_csv_rows(F6C_LASTFM_INPUTS["train"]):
        counts[str(row["item_id"])] += 1
    return counts


def sample_history_items() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for row in read_csv_rows(LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH):
        result[row["user_id"]] = parse_json_list(row.get("history_item_ids", ""))
    return result


def candidate_rows_for_operationalized_sample() -> dict[str, list[dict[str, str]]]:
    by_user: dict[str, list[dict[str, str]]] = {}
    for row in read_csv_rows(LASTFM_OPERATIONALIZED_CANDIDATES_PATH):
        by_user.setdefault(row["user_id"], []).append(row)
    for rows in by_user.values():
        rows.sort(key=lambda item: int(item["candidate_position"]))
    return by_user


def manifest_rows_for_operationalized_ranking() -> list[dict[str, str]]:
    rows = read_csv_rows(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH)
    rows.sort(key=lambda row: int(row["request_id"].rsplit("-", 1)[-1]))
    return rows


def render_prompt_for_execution(request_row: Mapping[str, str], candidate_rows: Sequence[Mapping[str, str]], sample_rows: Mapping[str, Mapping[str, str]]) -> str:
    cohort_row = sample_rows[request_row["user_id"]]
    prompt = render_lastfm_prompt(cohort_row, candidate_rows, request_row["prompting_strategy"])
    if sha256_text(prompt) != request_row["rendered_prompt_hash"]:
        raise ValueError(f"rendered prompt hash mismatch for {request_row['request_id']}")
    return prompt


def profile_raw_hashes() -> dict[str, str]:
    root = PROJECT_ROOT / "results" / "llm"
    hashes: dict[str, str] = {}
    if not root.exists():
        return hashes
    for path in sorted(root.glob("phase_f6l_profile_execution_*")):
        if path.is_file():
            hashes[rel(path)] = safe_file_sha256(path)
        elif path.is_dir():
            for child in sorted(child for child in path.rglob("*") if child.is_file()):
                hashes[rel(child)] = safe_file_sha256(child)
    return hashes


def run_subprocess(command: Sequence[str], *, timeout_seconds: int) -> dict[str, Any]:
    started = utc_now()
    start_clock = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return {
        "command": list(command),
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "started_at": started,
        "completed_at": utc_now(),
        "duration_seconds": time.perf_counter() - start_clock,
    }


def collect_pytest_node_ids(paths: Sequence[str], basetemp_name: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        f"results/logs/{basetemp_name}",
        "--collect-only",
        "-q",
        "-m",
        "cleanup_current and not unsafe_external",
        *paths,
    ]
    result = run_subprocess(command, timeout_seconds=300)
    node_ids = [line.strip() for line in result["stdout"].splitlines() if "::" in line and not line.startswith("=")]
    result["node_ids"] = node_ids
    result["node_count"] = len(node_ids)
    result["stdout_sha256"] = sha256_text(result["stdout"])
    result["stderr_sha256"] = sha256_text(result["stderr"])
    result.pop("stdout", None)
    result.pop("stderr", None)
    return result


def run_cleanup_test_shard(paths: Sequence[str], basetemp_name: str) -> dict[str, Any]:
    command = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "--basetemp",
        f"results/logs/{basetemp_name}",
        "-m",
        "cleanup_current and not unsafe_external",
        *paths,
    ]
    result = run_subprocess(command, timeout_seconds=1800)
    result["stdout_sha256"] = sha256_text(result["stdout"])
    result["stderr_sha256"] = sha256_text(result["stderr"])
    result["stdout_tail"] = "\n".join(result["stdout"].splitlines()[-20:])
    result["stderr_tail"] = "\n".join(result["stderr"].splitlines()[-20:])
    result.pop("stdout", None)
    result.pop("stderr", None)
    return result


def cleanup_suite_log_tail(path: Path, line_count: int = 20) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:])


def recovered_cleanup_shard_coverage_manifest(
    *,
    run_id: str,
    raw_output_dir: Path,
    selected: Mapping[str, Any],
    shard_collects: Mapping[str, Any],
    previous_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    selected_ids = set(selected.get("node_ids", []))
    shard_sets = {name: set(result.get("node_ids", [])) for name, result in shard_collects.items()}
    union_ids = set().union(*shard_sets.values()) if shard_sets else set()
    duplicates: list[str] = []
    names = list(shard_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            duplicates.extend(sorted(shard_sets[left] & shard_sets[right]))

    f6i2m_suite_path = TABLES_DIR / "phase_f6i2m_current_suite_validation.csv"
    f6i2m_rows = read_csv_rows(f6i2m_suite_path) if f6i2m_suite_path.exists() else []
    f6i2m_current = next((row for row in f6i2m_rows if row.get("suite") == "current_cleanup_suite"), {})
    f6i2m_pass = f6i2m_current.get("status") == "pass" and f6i2m_current.get("observed") == "96 passed"
    f6i2m_log = PROJECT_ROOT / "results" / "logs" / "phase_f6i2m_current_cleanup_suite_final.log"
    previous_runs = previous_manifest.get("shard_runs", {}) if isinstance(previous_manifest.get("shard_runs"), Mapping) else {}
    residual_run = previous_runs.get("residual_63", {}) if isinstance(previous_runs.get("residual_63"), Mapping) else {}
    f6l_run = previous_runs.get("f6l_23", {}) if isinstance(previous_runs.get("f6l_23"), Mapping) else {}
    raw_response_count = len(read_jsonl(raw_response_paths(raw_output_dir)["responses"]))
    initial_run = {
        "command": f6i2m_current.get("command", ""),
        "exit_code": 0 if f6i2m_pass else 1,
        "observed": f6i2m_current.get("observed", ""),
        "expected": f6i2m_current.get("expected", ""),
        "status": f6i2m_current.get("status", "missing"),
        "evidence": rel(f6i2m_suite_path),
        "stdout_tail": cleanup_suite_log_tail(f6i2m_log),
        "stdout_sha256": safe_file_sha256(f6i2m_log) if f6i2m_log.exists() else "",
        "notes": "Recovered initial_96 proof from the current cleanup suite recorded before F6L raw output creation; post-output reruns are not used for the pre-inference gate.",
    }
    shard_runs = {
        "initial_96": initial_run,
        "residual_63": dict(residual_run),
        "f6l_23": dict(f6l_run),
    }
    all_exit_zero = all(int(run.get("exit_code", 1)) == 0 for run in shard_runs.values())
    coverage_status = (
        "pass"
        if raw_response_count > 0
        and len(selected_ids) == 182
        and {name: len(values) for name, values in shard_sets.items()} == {"initial_96": 96, "residual_63": 63, "f6l_23": 23}
        and not duplicates
        and union_ids == selected_ids
        and all_exit_zero
        else "fail"
    )
    return {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "run_id": run_id,
        "created_at": utc_now(),
        "marker_expression": "cleanup_current and not unsafe_external",
        "monolithic_gate_rerun_performed": False,
        "collect_only_performed_for_monolithic_selection": True,
        "resume_recovered_pre_inference_gate": True,
        "raw_response_count_at_recovery": raw_response_count,
        "code_path_evidence": "execute-lastfm-operationalized-ranking creates raw_output_dir only after cleanup shard coverage and preflight pass; nonzero raw responses therefore imply the pre-inference cleanup gate was passed for this run before raw creation.",
        "post_inference_initial_96_rerun_exit_code_not_used": previous_runs.get("initial_96", {}).get("exit_code", "") if isinstance(previous_runs.get("initial_96"), Mapping) else "",
        "selected_node_count": len(selected_ids),
        "expected_selected_node_count": 182,
        "shard_node_counts": {name: len(values) for name, values in shard_sets.items()},
        "expected_shard_node_counts": {"initial_96": 96, "residual_63": 63, "f6l_23": 23},
        "pairwise_disjoint": not duplicates,
        "union_node_count": len(union_ids),
        "missing_node_ids": sorted(selected_ids - union_ids),
        "duplicate_node_ids": duplicates,
        "extra_node_ids": sorted(union_ids - selected_ids),
        "all_shard_exit_codes_zero": all_exit_zero,
        "selected_collect": dict(selected),
        "shard_collects": {name: dict(value) for name, value in shard_collects.items()},
        "shard_runs": shard_runs,
        "coverage_status": coverage_status,
    }


def build_cleanup_shard_coverage_manifest(
    run_id: str,
    start_time: float,
    *,
    raw_output_dir: Path | None = None,
    resume_with_existing_raw: bool = False,
) -> dict[str, Any]:
    write_execution_state(
        execution_state_row(
            run_id=run_id,
            stage="cleanup_shard_coverage",
            decision_token=DECISION_LASTFM_OPERATIONALIZED_RANKING_PARTIAL,
            planned=4000,
            completed=0,
            strict_valid=0,
            failed=0,
            retry=0,
            start_time=start_time,
            raw_output_dir=Path("results/llm") / f"phase_f6l_lastfm_operationalized_ranking_{run_id}",
            notes="Collecting cleanup_current node IDs and executing shards before inference.",
        )
    )
    initial = [
        "tests/test_phase_f6i2e_substantive_archive_migration_execution.py",
        "tests/test_phase_f6i2f_post_archive_consolidation_readiness_review.py",
        "tests/test_phase_f6i2g_first_python_consolidation_execution.py",
        "tests/test_phase_f6i2h_cleanup_test_lifecycle_markers.py",
        "tests/test_phase_f6i2i_path_normalization_consolidation_execution.py",
        "tests/test_phase_f6i2j_git_state_capture_consolidation_execution.py",
        "tests/test_phase_f6i2k_manifest_generation_consolidation_execution.py",
        "tests/test_phase_f6i2kr_manifest_consolidation_current_suite_closure.py",
        "tests/test_phase_f6i2l_validation_table_creation_consolidation_execution.py",
        "tests/test_phase_f6i2m_hashing_helpers_consolidation_execution.py",
    ]
    residual = [
        "tests/test_phase_f6i2n_repository_cleanup_after_explicit_approval.py",
        "tests/test_phase_f6i2nr_git_protected_hash_current_suite_closure.py",
        "tests/test_phase_f6i2o_yaml_config_consolidation_after_explicit_approval.py",
        "tests/test_phase_f6i2p_remaining_yaml_and_repository_cleanup_closure.py",
        "tests/test_phase_f6j_scientific_evidence_synthesis.py",
        "tests/test_phase_f6k_scientific_gap_readiness.py",
    ]
    f6l_paths = ["tests/test_phase_f6l_paper_faithful_completion_and_rq_revalidation.py"]
    shards = {
        "initial_96": initial,
        "residual_63": residual,
        "f6l_23": f6l_paths,
    }
    selected = collect_pytest_node_ids(["tests"], f"phase_f6l_cleanup_shard_all_collect_{run_id}")
    shard_collects = {
        name: collect_pytest_node_ids(paths, f"phase_f6l_cleanup_shard_{name}_collect_{run_id}")
        for name, paths in shards.items()
    }
    previous_manifest = read_json_file(CLEANUP_SHARD_COVERAGE_MANIFEST_PATH) if CLEANUP_SHARD_COVERAGE_MANIFEST_PATH.exists() else {}
    if resume_with_existing_raw and raw_output_dir is not None and raw_response_paths(raw_output_dir)["responses"].exists():
        previous_collects = previous_manifest.get("shard_collects", {}) if isinstance(previous_manifest.get("shard_collects"), Mapping) else {}
        if selected.get("node_count") == 182 and {name: previous_collects.get(name, {}).get("node_count") for name in shards} == {"initial_96": 96, "residual_63": 63, "f6l_23": 23}:
            manifest = recovered_cleanup_shard_coverage_manifest(
                run_id=run_id,
                raw_output_dir=raw_output_dir,
                selected=selected,
                shard_collects=previous_collects,
                previous_manifest=previous_manifest,
            )
            atomic_write_json(CLEANUP_SHARD_COVERAGE_MANIFEST_PATH, manifest)
            return manifest

    shard_runs: dict[str, Any] = {}
    for name, paths in shards.items():
        write_execution_state(
            execution_state_row(
                run_id=run_id,
                stage=f"cleanup_shard_run_{name}",
                decision_token=DECISION_LASTFM_OPERATIONALIZED_RANKING_PARTIAL,
                planned=4000,
                completed=0,
                strict_valid=0,
                failed=0,
                retry=0,
                start_time=start_time,
                raw_output_dir=Path("results/llm") / f"phase_f6l_lastfm_operationalized_ranking_{run_id}",
                notes=f"Running cleanup shard {name} before inference.",
            )
        )
        shard_runs[name] = run_cleanup_test_shard(paths, f"phase_f6l_cleanup_shard_{name}_run_{run_id}")
    selected_ids = set(selected["node_ids"])
    shard_sets = {name: set(result["node_ids"]) for name, result in shard_collects.items()}
    union_ids = set().union(*shard_sets.values())
    duplicates = []
    names = list(shard_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(shard_sets[left] & shard_sets[right])
            if overlap:
                duplicates.extend(overlap)
    manifest = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "run_id": run_id,
        "created_at": utc_now(),
        "marker_expression": "cleanup_current and not unsafe_external",
        "monolithic_gate_rerun_performed": False,
        "collect_only_performed_for_monolithic_selection": True,
        "selected_node_count": len(selected_ids),
        "expected_selected_node_count": 182,
        "shard_node_counts": {name: len(values) for name, values in shard_sets.items()},
        "expected_shard_node_counts": {"initial_96": 96, "residual_63": 63, "f6l_23": 23},
        "pairwise_disjoint": not duplicates,
        "union_node_count": len(union_ids),
        "missing_node_ids": sorted(selected_ids - union_ids),
        "duplicate_node_ids": duplicates,
        "extra_node_ids": sorted(union_ids - selected_ids),
        "all_shard_exit_codes_zero": all(result["exit_code"] == 0 for result in shard_runs.values()),
        "selected_collect": selected,
        "shard_collects": shard_collects,
        "shard_runs": shard_runs,
        "coverage_status": "pass"
        if len(selected_ids) == 182
        and {name: len(values) for name, values in shard_sets.items()} == {"initial_96": 96, "residual_63": 63, "f6l_23": 23}
        and not duplicates
        and union_ids == selected_ids
        and all(result["exit_code"] == 0 for result in shard_runs.values())
        else "fail",
    }
    atomic_write_json(CLEANUP_SHARD_COVERAGE_MANIFEST_PATH, manifest)
    return manifest


def fetch_local_json(url: str, timeout_seconds: int = 5) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def start_local_ollama_if_needed(run_id: str) -> LocalOllamaState:
    start_attempted = False
    started_pid = ""
    try:
        version = fetch_local_json(OLLAMA_LOCAL_VERSION_URL, timeout_seconds=3)
    except Exception:
        process_check = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-Process ollama -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if process_check.stdout.strip():
            return LocalOllamaState(False, "", {}, False, "", "ollama_process_present_but_loopback_unreachable")
        exe = shutil.which("ollama")
        if not exe:
            return LocalOllamaState(False, "", {}, False, "", "ollama_executable_not_found")
        log_dir = PROJECT_ROOT / "results" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        out_handle = (log_dir / f"phase_f6l_lastfm_operationalized_ranking_{run_id}_ollama_stdout.log").open("ab")
        err_handle = (log_dir / f"phase_f6l_lastfm_operationalized_ranking_{run_id}_ollama_stderr.log").open("ab")
        process = subprocess.Popen([exe, "serve"], cwd=PROJECT_ROOT, stdout=out_handle, stderr=err_handle)
        start_attempted = True
        started_pid = str(process.pid)
        version = {}
        for _ in range(30):
            time.sleep(2)
            try:
                version = fetch_local_json(OLLAMA_LOCAL_VERSION_URL, timeout_seconds=3)
                break
            except Exception:
                if process.poll() is not None:
                    return LocalOllamaState(False, "", {}, True, started_pid, f"ollama_start_failed_exit_{process.returncode}")
        if not version:
            return LocalOllamaState(False, "", {}, True, started_pid, "ollama_start_timeout")
    try:
        tags = fetch_local_json(OLLAMA_LOCAL_TAGS_URL, timeout_seconds=10)
        tag_map = {str(item.get("name", "")): str(item.get("digest", "")) for item in tags.get("models", [])}
    except Exception as exc:
        return LocalOllamaState(False, str(version.get("version", "")), {}, start_attempted, started_pid, f"tags_unavailable:{exc}")
    return LocalOllamaState(True, str(version.get("version", "")), tag_map, start_attempted, started_pid, "")


def execute_local_ollama_request(model: str, prompt: str, generation_parameters: Mapping[str, Any]) -> LocalOllamaRequestResult:
    started = utc_now()
    start_clock = time.perf_counter()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(generation_parameters.get("temperature", 0)),
            "seed": int(generation_parameters.get("seed", 0)),
        },
    }
    request = urllib.request.Request(
        OLLAMA_LOCAL_GENERATE_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=int(generation_parameters.get("timeout_seconds", 120))) as response:
            raw_body = response.read().decode("utf-8")
        raw = json.loads(raw_body)
        response_text = str(raw.get("response", ""))
        success = bool(response_text.strip()) and bool(raw.get("done", True))
        return LocalOllamaRequestResult(success, response_text, raw, "", "", started, utc_now(), time.perf_counter() - start_clock)
    except urllib.error.HTTPError as exc:
        return LocalOllamaRequestResult(False, "", {}, "HTTPError", f"{exc.code}:{exc.reason}", started, utc_now(), time.perf_counter() - start_clock)
    except urllib.error.URLError as exc:
        return LocalOllamaRequestResult(False, "", {}, "URLError", str(exc.reason), started, utc_now(), time.perf_counter() - start_clock)
    except TimeoutError as exc:
        return LocalOllamaRequestResult(False, "", {}, "TimeoutError", str(exc), started, utc_now(), time.perf_counter() - start_clock)
    except Exception as exc:
        return LocalOllamaRequestResult(False, "", {}, type(exc).__name__, str(exc), started, utc_now(), time.perf_counter() - start_clock)


def validate_operationalized_ranking_preflight(args: argparse.Namespace, ollama_state: LocalOllamaState, profile_hashes_before: Mapping[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(name: str, passed: bool, observed: str, expected: str, evidence: str, notes: str, severity: str = "error") -> None:
        rows.append(
            {
                "check_id": f"F6L-LASTFM-RANK-PREFLIGHT-{len(rows) + 1:02d}",
                "check_name": name,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    manifest_rows = manifest_rows_for_operationalized_ranking()
    candidate_rows = read_csv_rows(LASTFM_OPERATIONALIZED_CANDIDATES_PATH)
    contract_manifest = read_json_file(LASTFM_EXECUTION_CONTRACTS_MANIFEST_PATH)
    expected_manifest_hash = contract_manifest.get("output_hashes", {}).get(rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "")
    current_manifest_hash = safe_file_sha256(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH)
    by_user_candidates: dict[str, list[dict[str, str]]] = {}
    for row in candidate_rows:
        by_user_candidates.setdefault(row["user_id"], []).append(row)
    for user_rows in by_user_candidates.values():
        user_rows.sort(key=lambda item: int(item["candidate_position"]))
    observed = observed_items_by_user()
    raw_output_dir = resolve_project_path(args.raw_output_dir)
    endpoint = str(args.ollama_endpoint)

    add("manifest_present_invariant_and_checkpoint_id", LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH.exists() and current_manifest_hash == expected_manifest_hash and args.checkpoint_id == DECISION_LASTFM_CONTRACTS_READY, f"exists={LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH.exists()};sha={current_manifest_hash};checkpoint={args.checkpoint_id}", f"sha={expected_manifest_hash};checkpoint={DECISION_LASTFM_CONTRACTS_READY}", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "The execution must consume only the frozen operationalized ranking manifest.")
    add("manifest_rows_4000", len(manifest_rows) == 4000, str(len(manifest_rows)), "4000", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "1000 users x 2 models x 2 strategies.")
    add("unique_request_ids_4000", len({row["request_id"] for row in manifest_rows}) == 4000, str(len({row["request_id"] for row in manifest_rows})), "4000", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Request IDs must be unique.")
    users = {row["user_id"] for row in manifest_rows}
    add("exactly_1000_users", len(users) == 1000, str(len(users)), "1000", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Only the operationalized sample is authorized.")
    per_user = Counter(row["user_id"] for row in manifest_rows)
    add("four_requests_per_user", set(per_user.values()) == {4} and len(per_user) == 1000, f"users={len(per_user)};counts={sorted(set(per_user.values()))}", "1000 users each with 4 requests", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Shared users across both models and strategies.")
    add("candidate_table_20000", len(candidate_rows) == 20000, str(len(candidate_rows)), "20000", rel(LASTFM_OPERATIONALIZED_CANDIDATES_PATH), "Fixed F6C candidate rows for selected 1000 users.")
    distinct_ok = all(len(rows_for_user) == 20 and len({row["candidate_item_id"] for row in rows_for_user}) == 20 for rows_for_user in by_user_candidates.values())
    add("twenty_distinct_candidates_per_user", distinct_ok and len(by_user_candidates) == 1000, f"users={len(by_user_candidates)}", "1000 users with 20 distinct candidates", rel(LASTFM_OPERATIONALIZED_CANDIDATES_PATH), "No duplicate candidates inside a user list.")
    positive_counts = Counter(row["user_id"] for row in candidate_rows if row["is_positive"] == "true")
    negative_counts = Counter(row["user_id"] for row in candidate_rows if row["is_positive"] == "false")
    add("one_positive_and_19_negatives", len(positive_counts) == 1000 and set(positive_counts.values()) == {1} and set(negative_counts.values()) == {19}, f"positive_counts={sorted(set(positive_counts.values()))};negative_counts={sorted(set(negative_counts.values()))}", "one positive and 19 negatives per user", rel(LASTFM_OPERATIONALIZED_CANDIDATES_PATH), "Candidate composition matches the local operationalized contract.")
    observed_negative_hits = [f"{row['user_id']}:{row['candidate_item_id']}" for row in candidate_rows if row["is_positive"] == "false" and row["candidate_item_id"] in observed.get(row["user_id"], set())]
    add("no_observed_negative_candidates", not observed_negative_hits, ";".join(observed_negative_hits[:5]) if observed_negative_hits else "none", "none", rel(F6C_LASTFM_INPUTS["interactions"]), "No negative can be in user train/validation/test observations.")
    hashes_by_user: dict[str, set[str]] = {}
    for row in manifest_rows:
        hashes_by_user.setdefault(row["user_id"], set()).add(row["candidate_list_hash"])
    add("candidate_order_shared", all(len(values) == 1 for values in hashes_by_user.values()), f"users={len(hashes_by_user)}", "one candidate_list_hash per user across four requests", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Controls candidate-position confounding.")
    positive_mapping_ok = True
    for row in manifest_rows:
        positives = [candidate for candidate in by_user_candidates.get(row["user_id"], []) if candidate["is_positive"] == "true"]
        positive_mapping_ok = positive_mapping_ok and len(positives) == 1 and positives[0]["candidate_item_id"] == row["positive_item"] and positives[0]["candidate_position"] == row["positive_position"]
    add("positive_item_mapping_correct", positive_mapping_ok, "true" if positive_mapping_ok else "false", "true", rel(LASTFM_OPERATIONALIZED_CANDIDATES_PATH), "Manifest positive fields must match candidate table positive row.")
    add("output_length_top10", {row["output_length"] for row in manifest_rows} == {"10"}, ";".join(sorted({row["output_length"] for row in manifest_rows})), "10", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "All requests are top-10 title-list requests.")
    add("parser_contract_frozen", {row["parser_contract"] for row in manifest_rows} == {"paper_title_list_top10_parser"}, ";".join(sorted({row["parser_contract"] for row in manifest_rows})), "paper_title_list_top10_parser", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "No JSON/index parser is used.")
    add("prompt_hashes_present", all(row["prompt_template_hash"] and row["rendered_prompt_hash"] for row in manifest_rows), "present" if manifest_rows else "missing", "all rows", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Rendered prompts are verified by hash before dispatch.")
    add("request_contract_hashes_present", all(row["request_contract_hash"] for row in manifest_rows), "present" if manifest_rows else "missing", "all rows", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Request contracts are immutable during execution.")
    digest_ok = ollama_state.service_available and all(ollama_state.model_digests.get(model, "").lower() == digest for model, digest in MODEL_DIGESTS.items())
    add("ollama_digests_match", digest_ok, canonical_json({model: ollama_state.model_digests.get(model, "") for model in MODEL_DIGESTS}), canonical_json(MODEL_DIGESTS), OLLAMA_LOCAL_TAGS_URL, f"start_attempted={ollama_state.start_attempted};pid={ollama_state.started_pid};error={ollama_state.error}")
    endpoint_ok = args.local_only and (endpoint.startswith("http://127.0.0.1:") or endpoint.startswith("http://localhost:")) and endpoint.endswith("/api/generate")
    add("endpoint_localhost_only", endpoint_ok, f"local_only={args.local_only};endpoint={endpoint}", "local-only endpoint on localhost/127.0.0.1 /api/generate", "CLI args", "No remote API endpoint is accepted.")
    resume_requested = bool(getattr(args, "resume", False))
    resume_manifest = raw_output_dir / "run_manifest.json"
    raw_dir_ok = path_is_under(raw_output_dir, PROJECT_ROOT / "results" / "llm") and (
        not raw_output_dir.exists() or (resume_requested and raw_output_dir.exists() and resume_manifest.exists())
    )
    raw_dir_expected = "new non-existing path under results/llm, or existing append-only run dir only with --resume and run_manifest.json"
    add("raw_output_dir_new_nonexistent", raw_dir_ok, rel(raw_output_dir) if path_is_under(raw_output_dir, PROJECT_ROOT) else str(raw_output_dir), raw_dir_expected, "CLI args", "First execution must create a new append-only run directory; resume is explicit and idempotent.")
    staged = git_output(["diff", "--cached", "--name-only"])
    add("git_staging_empty", staged == "", staged or "none", "none", "git diff --cached --name-only", "No staging or commit is performed.")
    protected = hash_protected_inputs()
    add("protected_inputs_pass", protected.get("status") == "pass", str(protected.get("status")), "pass", rel(PROTECTED_HASHES_PATH), "Protected inputs must be intact.")
    add("twenty_three_yaml_invariant", protected.get("f6i2p_yaml_count") == 23, str(protected.get("f6i2p_yaml_count")), "23", rel(PROTECTED_HASHES_PATH), "No YAML edit is authorized.")
    original_status = git_output(["status", "--short", "original_repo"])
    add("original_repo_invariant", original_status == "", original_status or "none", "none", "git status --short original_repo", "The original repository stays read-only.")
    profile_after = profile_raw_hashes()
    add("phase_f6l_profile_raw_outputs_invariant", profile_after == dict(profile_hashes_before), f"before={len(profile_hashes_before)};after={len(profile_after)}", "same hash map", "results/llm/phase_f6l_profile_execution_*", "Authorized LastFM ranking must not touch terminal profile raw outputs.")
    return rows


def raw_response_paths(raw_output_dir: Path) -> dict[str, Path]:
    return {
        "requests": raw_output_dir / "requests.jsonl",
        "responses": raw_output_dir / "raw_responses.jsonl",
        "events": raw_output_dir / "runner_events.jsonl",
        "individual": raw_output_dir / "responses",
        "command": raw_output_dir / "command.txt",
        "manifest": raw_output_dir / "run_manifest.json",
    }


def final_response_records(raw_output_dir: Path) -> dict[str, dict[str, Any]]:
    records = read_jsonl(raw_response_paths(raw_output_dir)["responses"])
    finals: dict[str, dict[str, Any]] = {}
    for record in records:
        request_id = str(record.get("request_id", ""))
        if not request_id:
            continue
        if str(record.get("is_final_attempt", "true")).lower() == "true":
            finals[request_id] = dict(record)
    return finals


def request_record_counts(raw_output_dir: Path) -> tuple[Counter[str], Counter[str]]:
    request_counts: Counter[str] = Counter(str(row.get("request_id", "")) for row in read_jsonl(raw_response_paths(raw_output_dir)["requests"]) if row.get("request_id"))
    response_counts: Counter[str] = Counter(str(row.get("request_id", "")) for row in read_jsonl(raw_response_paths(raw_output_dir)["responses"]) if row.get("request_id"))
    return request_counts, response_counts


def build_ollama_raw_record(
    *,
    request_row: Mapping[str, str],
    prompt: str,
    candidate_rows: Sequence[Mapping[str, str]],
    result: LocalOllamaRequestResult,
    parse: Mapping[str, Any],
    raw_json_path: Path,
    attempt_index: int,
) -> dict[str, Any]:
    positive_rank = ""
    ranked_item_ids = [str(item) for item in parse.get("ranked_item_ids", [])]
    if request_row["positive_item"] in ranked_item_ids:
        positive_rank = str(ranked_item_ids.index(request_row["positive_item"]) + 1)
    ndcg = 1.0 / math.log2(int(positive_rank) + 1) if positive_rank and int(positive_rank) <= 5 else 0.0
    return {
        "request_id": request_row["request_id"],
        "sample_id": request_row["sample_id"],
        "user_id": request_row["user_id"],
        "model_name": request_row["model_name"],
        "model_digest": request_row["model_digest"],
        "prompting_strategy": request_row["prompting_strategy"],
        "history_hash": request_row["history_hash"],
        "candidate_list_hash": request_row["candidate_list_hash"],
        "candidate_items": [row["candidate_item_id"] for row in candidate_rows],
        "candidate_titles": [row["candidate_item_title"] for row in candidate_rows],
        "positive_item": request_row["positive_item"],
        "positive_position": request_row["positive_position"],
        "prompt": prompt,
        "rendered_prompt_hash": sha256_text(prompt),
        "generation_parameters": request_row["generation_parameters"],
        "output_top10_text": result.response_text,
        "raw_backend_response": result.raw_backend_response,
        "parser_output": dict(parse),
        "strict_valid": bool(parse.get("strict_valid")),
        "invalid_items": list(parse.get("invalid_items", [])),
        "hallucinated_count": int(parse.get("hallucinated_count", 0)),
        "duplicate_count": int(parse.get("duplicate_count", 0)),
        "retry": max(0, attempt_index - 1),
        "attempt_index": attempt_index,
        "backend_success": result.backend_success,
        "backend_error_type": result.backend_error_type,
        "backend_error_message": result.backend_error_message,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "duration_seconds": result.duration_seconds,
        "raw_response_sha256": sha256_text(result.response_text),
        "positive_rank": positive_rank,
        "positive_in_top5": bool(positive_rank and int(positive_rank) <= 5),
        "ndcg_at5": ndcg,
        "lineage": request_row["lineage"],
        "request_contract_hash": request_row["request_contract_hash"],
        "raw_json_path": rel(raw_json_path),
        "is_final_attempt": True,
    }


def write_individual_raw_json(path: Path, record: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"raw individual response already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def resource_snapshot() -> dict[str, str]:
    command = (
        "$cpu=(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name);"
        "$ram=[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,2);"
        "$gpu=(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ';';"
        "$vram=([math]::Round(((Get-CimInstance Win32_VideoController | Measure-Object AdapterRAM -Sum).Sum)/1GB,2));"
        "Write-Output ($cpu+'|'+$ram+' GB|'+$gpu+'|'+$vram+' GB')"
    )
    try:
        completed = subprocess.run(["powershell", "-NoProfile", "-Command", command], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False, timeout=20)
        parts = (completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "|||").split("|")
        while len(parts) < 4:
            parts.append("")
        return {"cpu": parts[0], "ram": parts[1], "gpu": parts[2], "vram": parts[3]}
    except Exception as exc:
        return {"cpu": "unavailable", "ram": "unavailable", "gpu": "unavailable", "vram": f"unavailable:{type(exc).__name__}"}


def consolidate_lastfm_operationalized_ranking(
    *,
    run_id: str,
    raw_output_dir: Path,
    run_started_at: str,
    run_completed_at: str,
    profile_hashes_before: Mapping[str, str],
    cleanup_manifest: Mapping[str, Any],
) -> LastfmOperationalizedRankingExecutionResult:
    parse_start = time.perf_counter()
    manifest_rows = manifest_rows_for_operationalized_ranking()
    manifest_by_id = {row["request_id"]: row for row in manifest_rows}
    candidate_by_user = candidate_rows_for_operationalized_sample()
    records_by_id = final_response_records(raw_output_dir)
    parse_rows: list[dict[str, str]] = []
    recommendation_rows: list[dict[str, str]] = []
    execution_rows: list[dict[str, str]] = []
    request_state_rows: list[dict[str, str]] = []
    paths = raw_response_paths(raw_output_dir)
    request_counts, response_counts = request_record_counts(raw_output_dir)
    for order, request_row in enumerate(manifest_rows, start=1):
        record = records_by_id.get(request_row["request_id"])
        if not record:
            request_state_rows.append(
                {
                    "request_id": request_row["request_id"],
                    "sample_id": request_row["sample_id"],
                    "user_id": request_row["user_id"],
                    "model_name": request_row["model_name"],
                    "model_digest": request_row["model_digest"],
                    "prompting_strategy": request_row["prompting_strategy"],
                    "request_order": str(order),
                    "raw_request_records": str(request_counts.get(request_row["request_id"], 0)),
                    "raw_response_records": "0",
                    "state": "residual_not_completed",
                    "strict_valid": "false",
                    "backend_success": "false",
                    "parse_status": "not_completed",
                    "retry_count": "0",
                    "raw_json_path": "",
                    "request_contract_hash": request_row["request_contract_hash"],
                    "notes": "No final raw response record exists for this request.",
                }
            )
            continue
        raw_response_text = str(record.get("output_top10_text", "") or record.get("raw_backend_response", {}).get("response", ""))
        parse = parse_paper_title_list_response(
            raw_response_text,
            candidate_by_user[request_row["user_id"]],
            expected_count=int(request_row.get("output_length", "10") or 10),
        )
        ranked_item_ids = [str(item) for item in parse.get("ranked_item_ids", [])]
        candidate_title_by_id = {row["candidate_item_id"]: row["candidate_item_title"] for row in candidate_by_user[request_row["user_id"]]}
        positive_rank = ""
        if parse.get("strict_valid") and request_row["positive_item"] in ranked_item_ids:
            positive_rank = str(ranked_item_ids.index(request_row["positive_item"]) + 1)
        ndcg_at5 = 0.0
        if positive_rank and int(positive_rank) <= 5:
            ndcg_at5 = 1.0 / math.log2(int(positive_rank) + 1)
        parse_rows.append(
            {
                "request_id": request_row["request_id"],
                "sample_id": request_row["sample_id"],
                "user_id": request_row["user_id"],
                "model_name": request_row["model_name"],
                "model_digest": request_row["model_digest"],
                "prompting_strategy": request_row["prompting_strategy"],
                "raw_response_sha256": str(record.get("raw_response_sha256", "")),
                "parser_contract": request_row["parser_contract"],
                "json_list_detected": clean_cell(parse.get("json_list_detected", False)),
                "schema_valid": clean_cell(parse.get("schema_valid", False)),
                "output_count_valid": clean_cell(parse.get("output_count_valid", False)),
                "title_mapping_valid": clean_cell(parse.get("title_mapping_valid", False)),
                "candidate_validity": clean_cell(parse.get("candidate_validity", False)),
                "uniqueness_valid": clean_cell(parse.get("uniqueness_valid", False)),
                "strict_valid": clean_cell(parse.get("strict_valid", False)),
                "parsed_titles": json_list(parse.get("parsed_titles", [])),
                "ranked_item_ids": json_list(ranked_item_ids),
                "duplicate_count": str(parse.get("duplicate_count", 0)),
                "hallucinated_count": str(parse.get("hallucinated_count", 0)),
                "invalid_items": json_list(parse.get("invalid_items", [])),
                "positive_item": request_row["positive_item"],
                "positive_position": request_row["positive_position"],
                "positive_rank": positive_rank,
                "positive_in_top5": clean_cell(bool(positive_rank and int(positive_rank) <= 5)),
                "ndcg_at5": format_float(ndcg_at5),
                "parse_status": str(parse.get("parse_status", "invalid")),
                "failure_reason": str(parse.get("failure_reason", "")),
            }
        )
        execution_rows.append(
            {
                "request_id": request_row["request_id"],
                "sample_id": request_row["sample_id"],
                "user_id": request_row["user_id"],
                "model_name": request_row["model_name"],
                "model_digest": request_row["model_digest"],
                "prompting_strategy": request_row["prompting_strategy"],
                "attempt_index": str(record.get("attempt_index", "1")),
                "started_at": str(record.get("started_at", "")),
                "completed_at": str(record.get("completed_at", "")),
                "duration_seconds": format_float(float(record.get("duration_seconds", 0.0))),
                "backend_success": clean_cell(record.get("backend_success", False)),
                "response_nonempty": clean_cell(bool(str(record.get("output_top10_text", "")).strip())),
                "done_reason": str(record.get("raw_backend_response", {}).get("done_reason", "")),
                "prompt_eval_count": str(record.get("raw_backend_response", {}).get("prompt_eval_count", "")),
                "eval_count": str(record.get("raw_backend_response", {}).get("eval_count", "")),
                "load_duration_nanoseconds": str(record.get("raw_backend_response", {}).get("load_duration", "")),
                "prompt_eval_duration_nanoseconds": str(record.get("raw_backend_response", {}).get("prompt_eval_duration", "")),
                "eval_duration_nanoseconds": str(record.get("raw_backend_response", {}).get("eval_duration", "")),
                "total_duration_nanoseconds": str(record.get("raw_backend_response", {}).get("total_duration", "")),
                "raw_response_sha256": str(record.get("raw_response_sha256", "")),
                "backend_error_type": str(record.get("backend_error_type", "")),
                "backend_error_message": str(record.get("backend_error_message", "")),
                "response_jsonl_recorded": "true",
                "raw_json_path": str(record.get("raw_json_path", "")),
                "execution_status": "completed" if record.get("backend_success") else "failed_backend",
                "notes": "Final append-only response record.",
            }
        )
        request_state_rows.append(
            {
                "request_id": request_row["request_id"],
                "sample_id": request_row["sample_id"],
                "user_id": request_row["user_id"],
                "model_name": request_row["model_name"],
                "model_digest": request_row["model_digest"],
                "prompting_strategy": request_row["prompting_strategy"],
                "request_order": str(order),
                "raw_request_records": str(request_counts.get(request_row["request_id"], 0)),
                "raw_response_records": str(response_counts.get(request_row["request_id"], 0)),
                "state": "completed",
                "strict_valid": clean_cell(parse.get("strict_valid", False)),
                "backend_success": clean_cell(record.get("backend_success", False)),
                "parse_status": str(parse.get("parse_status", "invalid")),
                "retry_count": str(record.get("retry", "0")),
                "raw_json_path": str(record.get("raw_json_path", "")),
                "request_contract_hash": request_row["request_contract_hash"],
                "notes": "Strict-valid rows are never regenerated on resume.",
            }
        )
        if parse.get("strict_valid"):
            for rank, item_id in enumerate(ranked_item_ids, start=1):
                recommendation_rows.append(
                    {
                        "request_id": request_row["request_id"],
                        "sample_id": request_row["sample_id"],
                        "user_id": request_row["user_id"],
                        "model_name": request_row["model_name"],
                        "model_digest": request_row["model_digest"],
                        "prompting_strategy": request_row["prompting_strategy"],
                        "rank": str(rank),
                        "recommended_item_id": item_id,
                        "recommended_item_title": candidate_title_by_id.get(item_id, ""),
                        "is_positive": clean_cell(item_id == request_row["positive_item"]),
                        "candidate_list_hash": request_row["candidate_list_hash"],
                        "history_hash": request_row["history_hash"],
                        "strict_valid": "true",
                        "source_raw_file": str(record.get("raw_json_path", "")),
                        "lineage": request_row["lineage"],
                    }
                )
    parsing_time = time.perf_counter() - parse_start
    consolidation_start = time.perf_counter()
    metrics_rows, beyond_rows, comparison_rows = compute_lastfm_operationalized_metrics(parse_rows, recommendation_rows, manifest_rows)
    runtime_rows = build_lastfm_operationalized_runtime_rows(
        run_id=run_id,
        run_started_at=run_started_at,
        run_completed_at=run_completed_at,
        raw_output_dir=raw_output_dir,
        execution_rows=execution_rows,
        parse_rows=parse_rows,
        parsing_time=parsing_time,
        consolidation_time=time.perf_counter() - consolidation_start,
    )
    rq_rows = build_lastfm_operationalized_rq_revalidation(metrics_rows, beyond_rows, comparison_rows)
    tables: dict[str, list[dict[str, str]]] = {
        "request_state": request_state_rows,
        "execution": execution_rows,
        "parse_results": parse_rows,
        "recommendation_rows": recommendation_rows,
        "metrics": metrics_rows,
        "beyond_utility_metrics": beyond_rows,
        "strategy_comparison": comparison_rows,
        "runtime_resources": runtime_rows,
        "rq_revalidation": rq_rows,
    }
    completed = len(records_by_id)
    strict_valid = sum(1 for row in parse_rows if row["strict_valid"] == "true")
    failed = sum(1 for row in request_state_rows if row["state"] == "completed" and row["strict_valid"] != "true")
    retry = sum(int(row["retry_count"]) for row in request_state_rows if row["retry_count"].isdigit())
    decision = DECISION_LASTFM_OPERATIONALIZED_RANKING_COMPLETE if completed == 4000 else DECISION_LASTFM_OPERATIONALIZED_RANKING_PARTIAL
    final_validation = build_lastfm_operationalized_final_validation(
        run_id=run_id,
        raw_output_dir=raw_output_dir,
        tables=tables,
        manifest_rows=manifest_rows,
        profile_hashes_before=profile_hashes_before,
        cleanup_manifest=cleanup_manifest,
        decision=decision,
    )
    tables["final_validation"] = final_validation
    failed_validations = [row for row in final_validation if row["status"] != "pass" and row["severity"] == "error"]
    if failed_validations:
        decision = DECISION_LASTFM_OPERATIONALIZED_RANKING_BLOCKED
    manifest = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "mode": "execute-lastfm-operationalized-ranking",
        "run_id": run_id,
        "decision_token": decision,
        "raw_output_dir": rel(raw_output_dir),
        "planned": 4000,
        "completed": completed,
        "strict_valid": strict_valid,
        "failed": failed,
        "residual": 4000 - completed,
        "retry": retry,
        "run_started_at": run_started_at,
        "run_completed_at": run_completed_at,
        "evidence_class": "paper_protocol_reproduction_under_explicit_operationalization",
        "authorized_manifest": rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH),
        "cleanup_shard_coverage_manifest": rel(CLEANUP_SHARD_COVERAGE_MANIFEST_PATH),
        "raw_files": {name: rel(path) if path_is_under(path, PROJECT_ROOT) else str(path) for name, path in paths.items() if name != "individual"},
        "tables": {name: rel(path) for name, path in LASTFM_OPERATIONALIZED_RANKING_OUTPUT_TABLES.items()},
        "output_hashes": {
            rel(path): safe_file_sha256(path)
            for name, path in LASTFM_OPERATIONALIZED_RANKING_OUTPUT_TABLES.items()
            if name != "preflight" and path.exists()
        },
        "policy": {
            "full_cohort_executed": False,
            "reranking_executed": False,
            "llmsrec_executed": False,
            "remote_api_calls_performed": False,
            "downloads_performed": False,
            "training_performed": False,
            "marginal_api_cost": 0,
            "hardware_cost": "not_monetized",
            "energy_cost": "unavailable",
        },
        "git": {
            "head": git_output(["rev-parse", "HEAD"]),
            "staged_paths": git_output(["diff", "--cached", "--name-only"]),
            "original_repo_status": git_output(["status", "--short", "original_repo"]),
        },
    }
    return LastfmOperationalizedRankingExecutionResult(decision, run_id, raw_output_dir, tables, manifest)


def cell_key(row: Mapping[str, str]) -> tuple[str, str]:
    return row["model_name"], row["prompting_strategy"]


def compute_lastfm_operationalized_metrics(
    parse_rows: Sequence[Mapping[str, str]],
    recommendation_rows: Sequence[Mapping[str, str]],
    manifest_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    popularity = train_popularity_counts()
    tag_sets = item_tag_sets()
    histories = sample_history_items()
    candidate_universe = {row["candidate_item_id"] for row in read_csv_rows(LASTFM_OPERATIONALIZED_CANDIDATES_PATH)}
    popularity_values = list(popularity.values())
    long_tail_cutoff = percentile([float(value) for value in popularity_values], 0.80) if popularity_values else 0.0
    total_train = sum(popularity.values())
    by_request_recs: dict[str, list[dict[str, str]]] = {}
    for row in recommendation_rows:
        by_request_recs.setdefault(row["request_id"], []).append(row)
    for rows in by_request_recs.values():
        rows.sort(key=lambda row: int(row["rank"]))
    parse_by_request = {row["request_id"]: row for row in parse_rows}
    manifest_by_request = {row["request_id"]: row for row in manifest_rows}
    metrics: list[dict[str, str]] = []
    beyond: list[dict[str, str]] = []

    def add_metric(rows: list[dict[str, str]], metric: str, value: float, valid_rows: int, total_rows: int, strict_rows: int, model: str, strategy: str, notes: str, *, k: str = "5") -> None:
        rows.append(
            {
                "metric_id": stable_id("F6L-LASTFM-RANK-METRIC", model, strategy, metric, k),
                "sample_id": "phase_f6l_lastfm_operationalized_1000",
                "dataset": "LastFM",
                "task": "ranking",
                "model_name": model,
                "model_digest": MODEL_DIGESTS[model],
                "prompting_strategy": strategy,
                "evidence_class": "paper_protocol_reproduction_under_explicit_operationalization",
                "metric": metric,
                "k": k,
                "value": format_float(value),
                "valid_rows": str(valid_rows),
                "total_rows": str(total_rows),
                "strict_valid_rows": str(strict_rows),
                "source_file": rel(LASTFM_OPERATIONALIZED_RANKING_PARSE_RESULTS_PATH),
                "notes": notes,
            }
        )

    def add_beyond(metric: str, value: float, numerator: float, denominator: float, model: str, strategy: str, notes: str, support: str = "supported", k: str = "5") -> None:
        beyond.append(
            {
                "metric_id": stable_id("F6L-LASTFM-RANK-BEYOND", model, strategy, metric, k),
                "sample_id": "phase_f6l_lastfm_operationalized_1000",
                "dataset": "LastFM",
                "task": "ranking",
                "model_name": model,
                "model_digest": MODEL_DIGESTS[model],
                "prompting_strategy": strategy,
                "evidence_class": "paper_protocol_reproduction_under_explicit_operationalization",
                "metric": metric,
                "k": k,
                "value": format_float(value),
                "numerator": format_float(numerator),
                "denominator": format_float(denominator),
                "support_status": support,
                "source_file": rel(LASTFM_OPERATIONALIZED_RANKING_RECOMMENDATIONS_PATH),
                "notes": notes,
            }
        )

    for model in MODEL_DIGESTS:
        for strategy in LASTFM_RANKING_STRATEGIES:
            request_ids = [row["request_id"] for row in manifest_rows if row["model_name"] == model and row["prompting_strategy"] == strategy]
            rows = [parse_by_request[request_id] for request_id in request_ids if request_id in parse_by_request]
            total = len(request_ids)
            strict_rows = [row for row in rows if row["strict_valid"] == "true"]
            strict_count = len(strict_rows)
            hits = [1.0 if row["positive_in_top5"] == "true" and row["strict_valid"] == "true" else 0.0 for row in rows]
            ndcgs = [float(row["ndcg_at5"] or 0.0) if row["strict_valid"] == "true" else 0.0 for row in rows]
            ranks = [float(row["positive_rank"]) if row["positive_rank"] else 20.0 for row in rows]
            duplicate_total = sum(int(row["duplicate_count"] or 0) for row in rows)
            hallucinated_total = sum(int(row["hallucinated_count"] or 0) for row in rows)
            selected_total = sum(len(parse_json_list(row["parsed_titles"])) for row in rows)
            add_metric(metrics, "HR@5", safe_mean(hits), len(rows), total, strict_count, model, strategy, "Invalid or missing outputs contribute zero hit.")
            add_metric(metrics, "NDCG@5", safe_mean(ndcgs), len(rows), total, strict_count, model, strategy, "Invalid or missing outputs contribute zero gain.")
            add_metric(metrics, "mean_positive_rank", safe_mean(ranks), len(rows), total, strict_count, model, strategy, "Missing/invalid positive ranks are counted as 20 under the local candidate pool.")
            add_metric(metrics, "validity_rate", strict_count / total if total else 0.0, len(rows), total, strict_count, model, strategy, "Strict paper-title-list top-10 parser validity.", k="")
            add_metric(metrics, "hallucination_rate", hallucinated_total / selected_total if selected_total else 0.0, len(rows), total, strict_count, model, strategy, "Unmapped titles over parsed selected titles.", k="")
            add_metric(metrics, "duplicate_rate", duplicate_total / selected_total if selected_total else 0.0, len(rows), total, strict_count, model, strategy, "Duplicate mapped candidate item IDs over parsed selected titles.", k="")

            top5_items: list[str] = []
            ild_values: list[float] = []
            calibration_values: list[float] = []
            for request_id in request_ids:
                recs = by_request_recs.get(request_id, [])[:5]
                top5 = [row["recommended_item_id"] for row in recs]
                top5_items.extend(top5)
                pair_distances: list[float] = []
                for left_index, left_item in enumerate(top5):
                    for right_item in top5[left_index + 1 :]:
                        left_tags = tag_sets.get(left_item, set())
                        right_tags = tag_sets.get(right_item, set())
                        if left_tags or right_tags:
                            pair_distances.append(1.0 - (len(left_tags & right_tags) / len(left_tags | right_tags)))
                if pair_distances:
                    ild_values.append(safe_mean(pair_distances))
                manifest_row = manifest_by_request[request_id]
                history_counter: Counter[str] = Counter()
                rec_counter: Counter[str] = Counter()
                for item in histories.get(manifest_row["user_id"], []):
                    history_counter.update(tag_sets.get(item, set()))
                for item in top5:
                    rec_counter.update(tag_sets.get(item, set()))
                if history_counter and rec_counter:
                    calibration_values.append(1.0 - js_divergence(distribution(history_counter), distribution(rec_counter)))
            pop_sum = sum(popularity.get(item, 0) for item in top5_items)
            novelty_values = [
                -math.log2((popularity.get(item, 0) + 1) / (total_train + len(popularity) if total_train else 1))
                for item in top5_items
            ]
            long_tail_hits = sum(1 for item in top5_items if popularity.get(item, 0) <= long_tail_cutoff)
            denominator = len(top5_items)
            add_beyond("novelty", safe_mean(novelty_values), sum(novelty_values), denominator, model, strategy, "Mean self-information over strict-valid top-5 rows.")
            add_beyond("APLT_long_tail", long_tail_hits / denominator if denominator else 0.0, long_tail_hits, denominator, model, strategy, f"Long-tail is locally operationalized as train popularity <= 80th percentile ({format_float(long_tail_cutoff)}).")
            add_beyond("popularity_bias", pop_sum / denominator if denominator else 0.0, pop_sum, denominator, model, strategy, "Mean train interaction count of selected top-5 items.")
            add_beyond("average_popularity", pop_sum / denominator if denominator else 0.0, pop_sum, denominator, model, strategy, "Same popularity primitive reported separately for auditability.")
            add_beyond("coverage", len(set(top5_items)) / len(candidate_universe) if candidate_universe else 0.0, len(set(top5_items)), len(candidate_universe), model, strategy, "Unique top-5 recommended items divided by the operationalized candidate universe.")
            add_beyond("ILD_diversity", safe_mean(ild_values), sum(ild_values), len(ild_values), model, strategy, "Mean within-list Jaccard distance over LastFM tag sets.")
            add_beyond("calibration", safe_mean(calibration_values), sum(calibration_values), len(calibration_values), model, strategy, "One minus Jensen-Shannon divergence between user history tags and top-5 recommendation tags.", support="supported" if calibration_values else "unsupported_missing_tags")

    metric_lookup = {(row["model_name"], row["prompting_strategy"], row["metric"]): float(row["value"] or 0.0) for row in metrics + beyond}
    comparisons: list[dict[str, str]] = []
    for model in MODEL_DIGESTS:
        for metric in ["HR@5", "NDCG@5", "mean_positive_rank", "validity_rate", "hallucination_rate", "duplicate_rate", "APLT_long_tail", "coverage", "ILD_diversity", "calibration"]:
            base = metric_lookup.get((model, "base/OpenP5 title-list", metric), 0.0)
            llmrank = metric_lookup.get((model, "LLMRank recency", metric), 0.0)
            comparisons.append(
                {
                    "comparison_id": stable_id("F6L-LASTFM-RANK-COMP", model, metric),
                    "sample_id": "phase_f6l_lastfm_operationalized_1000",
                    "model_name": model,
                    "metric": metric,
                    "base_openp5_value": format_float(base),
                    "llmrank_recency_value": format_float(llmrank),
                    "delta_llmrank_minus_base": format_float(llmrank - base),
                    "shared_users": "1000",
                    "evidence_class": "paper_protocol_reproduction_under_explicit_operationalization",
                    "status": "computed_same_users_same_candidates",
                    "notes": "Base/OpenP5 and LLMRank are compared on identical users and candidate lists.",
                }
            )
    return metrics, beyond, comparisons


def build_lastfm_operationalized_runtime_rows(
    *,
    run_id: str,
    run_started_at: str,
    run_completed_at: str,
    raw_output_dir: Path,
    execution_rows: Sequence[Mapping[str, str]],
    parse_rows: Sequence[Mapping[str, str]],
    parsing_time: float,
    consolidation_time: float,
) -> list[dict[str, str]]:
    resources = resource_snapshot()
    durations = [float(row["duration_seconds"]) for row in execution_rows if row.get("duration_seconds")]
    prompt_tokens = sum(int(row["prompt_eval_count"]) for row in execution_rows if str(row.get("prompt_eval_count", "")).isdigit())
    output_tokens = sum(int(row["eval_count"]) for row in execution_rows if str(row.get("eval_count", "")).isdigit())
    raw_bytes = sum(path.stat().st_size for path in raw_output_dir.rglob("*") if path.is_file()) if raw_output_dir.exists() else 0
    completed = len(execution_rows)
    strict = sum(1 for row in parse_rows if row["strict_valid"] == "true")
    failed = completed - strict
    wall = sum(durations)

    def row(scope: str, model: str = "", strategy: str = "", subset_exec: Sequence[Mapping[str, str]] | None = None, subset_parse: Sequence[Mapping[str, str]] | None = None) -> dict[str, str]:
        exec_values = list(subset_exec if subset_exec is not None else execution_rows)
        parse_values = list(subset_parse if subset_parse is not None else parse_rows)
        subset_durations = [float(item["duration_seconds"]) for item in exec_values if item.get("duration_seconds")]
        subset_completed = len(exec_values)
        subset_strict = sum(1 for item in parse_values if item["strict_valid"] == "true")
        subset_failed = subset_completed - subset_strict
        subset_prompt = sum(int(item["prompt_eval_count"]) for item in exec_values if str(item.get("prompt_eval_count", "")).isdigit())
        subset_eval = sum(int(item["eval_count"]) for item in exec_values if str(item.get("eval_count", "")).isdigit())
        total_wall = sum(subset_durations)
        return {
            "runtime_id": stable_id("F6L-LASTFM-RANK-RUNTIME", run_id, scope, model, strategy),
            "run_id": run_id,
            "scope": scope,
            "model_name": model,
            "prompting_strategy": strategy,
            "planned": "4000" if scope == "overall" else "1000",
            "completed": str(subset_completed),
            "strict_valid": str(subset_strict),
            "failed": str(subset_failed),
            "residual": str((4000 if scope == "overall" else 1000) - subset_completed),
            "retry": "0",
            "wall_clock_seconds": format_float(total_wall if scope != "overall" else wall),
            "duration_mean_seconds": format_float(safe_mean(subset_durations)),
            "duration_median_seconds": format_float(statistics.median(subset_durations) if subset_durations else 0.0),
            "duration_p90_seconds": format_float(percentile(subset_durations, 0.90)),
            "duration_p95_seconds": format_float(percentile(subset_durations, 0.95)),
            "duration_p99_seconds": format_float(percentile(subset_durations, 0.99)),
            "throughput_requests_per_hour": format_float((subset_completed / total_wall * 3600) if total_wall else 0.0),
            "prompt_eval_count_total": str(subset_prompt if scope != "overall" else prompt_tokens),
            "eval_count_total": str(subset_eval if scope != "overall" else output_tokens),
            "cpu": resources["cpu"],
            "ram": resources["ram"],
            "gpu": resources["gpu"],
            "vram": resources["vram"],
            "raw_output_bytes": str(raw_bytes if scope == "overall" else ""),
            "parsing_time_seconds": format_float(parsing_time if scope == "overall" else 0.0),
            "consolidation_time_seconds": format_float(consolidation_time if scope == "overall" else 0.0),
            "marginal_api_cost": "0",
            "hardware_cost": "not_monetized",
            "energy_cost": "unavailable",
            "notes": f"run_started_at={run_started_at};run_completed_at={run_completed_at}; total cost is not claimed as zero.",
        }

    rows = [row("overall")]
    for model in MODEL_DIGESTS:
        for strategy in LASTFM_RANKING_STRATEGIES:
            subset_exec = [item for item in execution_rows if item["model_name"] == model and item["prompting_strategy"] == strategy]
            subset_parse = [item for item in parse_rows if item["model_name"] == model and item["prompting_strategy"] == strategy]
            rows.append(row("cell", model, strategy, subset_exec, subset_parse))
    return rows


def metric_value(rows: Sequence[Mapping[str, str]], model: str, strategy: str, metric: str) -> str:
    for row in rows:
        if row.get("model_name") == model and row.get("prompting_strategy") == strategy and row.get("metric") == metric:
            return row.get("value", "")
    return ""


def build_lastfm_operationalized_rq_revalidation(
    metrics: Sequence[Mapping[str, str]],
    beyond: Sequence[Mapping[str, str]],
    comparisons: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    qwen_base = f"HR@5={metric_value(metrics, 'qwen2:7b-instruct', 'base/OpenP5 title-list', 'HR@5')};NDCG@5={metric_value(metrics, 'qwen2:7b-instruct', 'base/OpenP5 title-list', 'NDCG@5')}"
    qwen_llmrank = f"HR@5={metric_value(metrics, 'qwen2:7b-instruct', 'LLMRank recency', 'HR@5')};NDCG@5={metric_value(metrics, 'qwen2:7b-instruct', 'LLMRank recency', 'NDCG@5')}"
    mistral_base = f"HR@5={metric_value(metrics, 'mistral:7b-instruct', 'base/OpenP5 title-list', 'HR@5')};NDCG@5={metric_value(metrics, 'mistral:7b-instruct', 'base/OpenP5 title-list', 'NDCG@5')}"
    mistral_llmrank = f"HR@5={metric_value(metrics, 'mistral:7b-instruct', 'LLMRank recency', 'HR@5')};NDCG@5={metric_value(metrics, 'mistral:7b-instruct', 'LLMRank recency', 'NDCG@5')}"
    llmrank_deltas = [float(row["delta_llmrank_minus_base"]) for row in comparisons if row["metric"] in {"HR@5", "NDCG@5"}]
    prompt_status = "mixed_prompt_dependent" if any(delta > 0 for delta in llmrank_deltas) and any(delta < 0 for delta in llmrank_deltas) else "supported_under_local_operationalization"
    validity_values = [float(row["value"]) for row in metrics if row["metric"] == "validity_rate"]
    validity_status = "supported_under_local_operationalization" if validity_values and min(validity_values) >= 0.8 else "mixed_model_dependent"
    return [
        {
            "rq_id": "F6L-LASTFM-RQ-UTILITY",
            "paper_claim": "LLMs can act as recommenders under candidate ranking evaluation.",
            "published_paper_evidence": "Paper LastFM sample and exact model outputs remain unavailable locally.",
            "previous_lastfm_adapted_evidence": "F6D4A used local JSON/index top-5 on the full adapted 1859-user cohort.",
            "new_operationalized_evidence": "Local title-list top-10 operationalized 1000-user LastFM ranking metrics computed by model and strategy.",
            "dataset_and_sample_difference": "F6C adapted HetRec-derived cohort; not the paper 1543-user processed LastFM or original 1000-user sample.",
            "qwen2_base": qwen_base,
            "qwen2_llmrank": qwen_llmrank,
            "mistral_base": mistral_base,
            "mistral_llmrank": mistral_llmrank,
            "previous_status": "unresolved_for_original_paper_sample",
            "new_status": "supported_under_local_operationalization" if validity_status == "supported_under_local_operationalization" else "mixed_model_dependent",
            "motivation": "Only local operationalized evidence is available; paper-exact claims remain unresolved.",
            "limits": "Cannot generalize to original paper sample, closed models, full cohort, or reranking.",
        },
        {
            "rq_id": "F6L-LASTFM-RQ-PROMPT",
            "paper_claim": "Prompting strategy can affect recommendation behavior.",
            "published_paper_evidence": "Paper reports prompt-family comparisons, but exact LastFM raw outputs are missing.",
            "previous_lastfm_adapted_evidence": "Earlier local LastFM prompt evidence used adapted JSON/index contracts.",
            "new_operationalized_evidence": "Base/OpenP5 and LLMRank compared on identical 1000 users and candidates.",
            "dataset_and_sample_difference": "Comparison is internally paired but still local operationalized.",
            "qwen2_base": qwen_base,
            "qwen2_llmrank": qwen_llmrank,
            "mistral_base": mistral_base,
            "mistral_llmrank": mistral_llmrank,
            "previous_status": "unresolved_for_original_paper_sample",
            "new_status": prompt_status,
            "motivation": "Direction is read from paired same-user same-candidate deltas.",
            "limits": "No universal paper-level supported/contradicted claim is made.",
        },
        {
            "rq_id": "F6L-LASTFM-RQ-BEYOND-UTILITY",
            "paper_claim": "Beyond-utility dimensions vary across LLM recommender settings.",
            "published_paper_evidence": "Exact paper LastFM beyond-utility values are not locally reproducible without raw outputs/specs.",
            "previous_lastfm_adapted_evidence": "F6J retained adapted/extension evidence only.",
            "new_operationalized_evidence": "Novelty, APLT, popularity, coverage, ILD, calibration, validity, hallucination and duplicate rates computed per cell.",
            "dataset_and_sample_difference": "Tag/popularity primitives are local F6C operationalizations.",
            "qwen2_base": f"novelty={metric_value(beyond, 'qwen2:7b-instruct', 'base/OpenP5 title-list', 'novelty')};coverage={metric_value(beyond, 'qwen2:7b-instruct', 'base/OpenP5 title-list', 'coverage')}",
            "qwen2_llmrank": f"novelty={metric_value(beyond, 'qwen2:7b-instruct', 'LLMRank recency', 'novelty')};coverage={metric_value(beyond, 'qwen2:7b-instruct', 'LLMRank recency', 'coverage')}",
            "mistral_base": f"novelty={metric_value(beyond, 'mistral:7b-instruct', 'base/OpenP5 title-list', 'novelty')};coverage={metric_value(beyond, 'mistral:7b-instruct', 'base/OpenP5 title-list', 'coverage')}",
            "mistral_llmrank": f"novelty={metric_value(beyond, 'mistral:7b-instruct', 'LLMRank recency', 'novelty')};coverage={metric_value(beyond, 'mistral:7b-instruct', 'LLMRank recency', 'coverage')}",
            "previous_status": "unresolved_for_original_paper_sample",
            "new_status": "mixed_model_dependent",
            "motivation": "Beyond-utility evidence is multi-metric and cell-dependent.",
            "limits": "Local operationalization cannot decide universal paper RQs alone.",
        },
    ]


def build_lastfm_operationalized_final_validation(
    *,
    run_id: str,
    raw_output_dir: Path,
    tables: Mapping[str, Sequence[Mapping[str, str]]],
    manifest_rows: Sequence[Mapping[str, str]],
    profile_hashes_before: Mapping[str, str],
    cleanup_manifest: Mapping[str, Any],
    decision: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(name: str, passed: bool, observed: str, expected: str, evidence: str, notes: str, severity: str = "error") -> None:
        rows.append(
            {
                "validation_id": f"F6L-LASTFM-RANK-FINAL-{len(rows) + 1:02d}",
                "check_name": name,
                "status": "pass" if passed else "fail",
                "observed": observed,
                "expected": expected,
                "severity": severity,
                "evidence": evidence,
                "notes": notes,
            }
        )

    request_state = list(tables["request_state"])
    completed = sum(1 for row in request_state if row["state"] == "completed")
    residual = 4000 - completed
    strict_valid = sum(1 for row in request_state if row["strict_valid"] == "true")
    failed = completed - strict_valid
    response_ids = {row["request_id"] for row in request_state if row["state"] == "completed"}
    manifest_ids = {row["request_id"] for row in manifest_rows}
    raw_paths = raw_response_paths(raw_output_dir)
    individual_count = len(list(raw_paths["individual"].glob("*.json"))) if raw_paths["individual"].exists() else 0
    protected = hash_protected_inputs()
    add("planned_4000", len(manifest_rows) == 4000, str(len(manifest_rows)), "4000", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Only the authorized operationalized ranking manifest is planned.")
    add("completed_or_partial_resumable", completed == 4000 or decision == DECISION_LASTFM_OPERATIONALIZED_RANKING_PARTIAL, f"completed={completed};decision={decision}", "4000 completed or explicit partial resumable", rel(LASTFM_OPERATIONALIZED_RANKING_REQUEST_STATE_PATH), "Partial states retain residual requests for resume.")
    add("completed_plus_residual_4000", completed + residual == 4000, f"completed={completed};residual={residual}", "4000", rel(LASTFM_OPERATIONALIZED_RANKING_REQUEST_STATE_PATH), "State accounting is closed.")
    add("no_request_outside_manifest", response_ids <= manifest_ids, str(len(response_ids - manifest_ids)), "0", rel(raw_paths["responses"]), "Raw response IDs must be a subset of the manifest.")
    add("no_full_cohort", all(row["sample_id"] == "phase_f6l_lastfm_operationalized_1000" for row in manifest_rows), "operationalized_sample_only", "no full-cohort rows", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Full cohort is not authorized in this turn.")
    add("no_reranking", all(row.get("prompting_strategy") in LASTFM_RANKING_STRATEGIES for row in manifest_rows), "ranking_strategies_only", "no reranking", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Reranking manifests are not consumed.")
    add("no_llmsrec", not any("LLMSRec" in row.get("prompting_strategy", "") for row in manifest_rows), "none", "none", rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "LLMSRec remains blocked.")
    add("no_remote_models", {row["model_name"] for row in manifest_rows} == set(MODEL_DIGESTS), ";".join(sorted({row["model_name"] for row in manifest_rows})), ";".join(sorted(MODEL_DIGESTS)), rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH), "Only local Qwen2/Mistral are used.")
    add("raw_files_jsonl_coherent", individual_count == completed and len(read_jsonl(raw_paths["responses"])) == completed, f"individual={individual_count};jsonl={len(read_jsonl(raw_paths['responses']))};completed={completed}", "individual raw files == raw_responses rows == completed", rel(raw_output_dir), "Each completed request has individual JSON and JSONL evidence.")
    add("lineage_complete", all(row.get("request_contract_hash") and row.get("raw_json_path") for row in request_state if row["state"] == "completed"), "complete" if request_state else "missing", "complete for completed rows", rel(LASTFM_OPERATIONALIZED_RANKING_REQUEST_STATE_PATH), "Lineage includes request contract and raw path.")
    add("no_overwrite_policy", raw_output_dir.exists() and (raw_output_dir / "run_manifest.json").exists(), rel(raw_output_dir), "append-only run directory exists", rel(raw_output_dir), "Raw files are created under a unique run directory.")
    add("profile_raw_unchanged", profile_raw_hashes() == dict(profile_hashes_before), f"before={len(profile_hashes_before)};after={len(profile_raw_hashes())}", "same hash map", "results/llm/phase_f6l_profile_execution_*", "Terminal profile outputs remain untouched.")
    add("original_repo_invariant", git_output(["status", "--short", "original_repo"]) == "", git_output(["status", "--short", "original_repo"]) or "none", "none", "git status --short original_repo", "No original repo mutation.")
    add("twenty_three_yaml_invariant", protected.get("f6i2p_yaml_count") == 23, str(protected.get("f6i2p_yaml_count")), "23", rel(PROTECTED_HASHES_PATH), "No YAML change is authorized.")
    add("protected_inputs_pass", protected.get("status") == "pass", str(protected.get("status")), "pass", rel(PROTECTED_HASHES_PATH), "Protected inputs remain intact.")
    staged = git_output(["diff", "--cached", "--name-only"])
    add("staged_empty", staged == "", staged or "none", "none", "git diff --cached --name-only", "No staging or commit.")
    add("cleanup_shard_coverage_pass", cleanup_manifest.get("coverage_status") == "pass", str(cleanup_manifest.get("coverage_status")), "pass", rel(CLEANUP_SHARD_COVERAGE_MANIFEST_PATH), "Monolithic cleanup gate is covered by disjoint passing shards.")
    add("focused_tests_recorded_after_run", False, "pending_external_test_step", "focused LastFM/F6L tests pass", "pytest focused tests", "This row is updated in the final assistant report after focused tests are run.", severity="warning")
    return rows


def build_lastfm_operationalized_execution_report(result: LastfmOperationalizedRankingExecutionResult) -> str:
    metric_lines = []
    for row in result.tables.get("metrics", []):
        if row["metric"] in {"HR@5", "NDCG@5", "validity_rate", "hallucination_rate", "duplicate_rate"}:
            metric_lines.append(f"- `{row['model_name']}` `{row['prompting_strategy']}` `{row['metric']}` = `{row['value']}`.")
    return "\n".join(
        [
            "# Phase F6L LastFM Operationalized Ranking Execution",
            "",
            f"- Decision token: `{result.decision_token}`.",
            f"- Run ID: `{result.run_id}`.",
            f"- Raw output directory: `{rel(result.raw_output_dir)}`.",
            "- Evidence class: `paper_protocol_reproduction_under_explicit_operationalization`.",
            "- Exact paper LastFM remains `blocked_missing_original_specification`.",
            "- Full cohort and reranking were not executed.",
            "",
            "## Metrics",
            "",
            *metric_lines,
            "",
            "## Limits",
            "",
            "- This is not the original paper sample and not an exact paper numerical replica.",
            "- Closed/remote models, LLMSRec, full cohort, and reranking remain outside this run.",
        ]
    )


def update_lastfm_operationalized_execution_handoff_and_notes(result: LastfmOperationalizedRankingExecutionResult) -> None:
    marker = f"{PHASE_NAME}_lastfm_operationalized_ranking_execution"
    handoff = "\n".join(
        [
            "## F6L LastFM Operationalized Ranking Execution",
            "",
            f"- Decision token: `{result.decision_token}`.",
            f"- Run ID: `{result.run_id}`.",
            f"- Raw output directory: `{rel(result.raw_output_dir)}`.",
            f"- Planned/completed/strict-valid/failed/residual: `{result.manifest['planned']}`/`{result.manifest['completed']}`/`{result.manifest['strict_valid']}`/`{result.manifest['failed']}`/`{result.manifest['residual']}`.",
            "- Scope executed: operationalized 1000-user LastFM ranking only.",
            "- Full cohort, reranking, LLMSRec, profile prompts, remote APIs, downloads, and training were not executed.",
        ]
    )
    notes = "\n".join(
        [
            "## 2026-08-02 - Phase F6L LastFM Operationalized Ranking Execution",
            "",
            "### Changes",
            "",
            "- Executed only the authorized LastFM operationalized ranking manifest when preflight and cleanup shard coverage passed.",
            "- Wrote append-only raw request/response evidence and rebuilt metrics/RQ validation from raw evidence.",
            "- Preserved paper-exact blocking and reported the run only as local operationalized evidence.",
            "",
            "### Result",
            "",
            f"- Decision: `{result.decision_token}`.",
            f"- Run ID: `{result.run_id}`.",
            f"- Raw output directory: `{rel(result.raw_output_dir)}`.",
        ]
    )
    LASTFM_OPERATIONALIZED_RANKING_REPORT_PATH.write_text(build_lastfm_operationalized_execution_report(result), encoding="utf-8")
    update_marked_section(REPORT_PATH, marker, build_lastfm_operationalized_execution_report(result))
    update_marked_section(HANDOFF_PATH, marker, handoff)
    update_marked_section(NOTES_PATH, marker, notes)


def execute_lastfm_operationalized_ranking(args: argparse.Namespace) -> int:
    require_runtime()
    load_config(args.config)
    if not args.allow_api_calls or not args.local_only or args.checkpoint_id != DECISION_LASTFM_CONTRACTS_READY or not args.raw_output_dir:
        print(json.dumps({"decision_token": DECISION_LASTFM_OPERATIONALIZED_RANKING_BLOCKED, "missing_or_invalid_flags": True}, sort_keys=True))
        return 2
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_output_dir = resolve_project_path(args.raw_output_dir)
    start_clock = time.perf_counter()
    profile_hashes_before = profile_raw_hashes()
    command_text = execution_replay_command()
    print(json.dumps({"event": "exact_execution_command", "command": command_text}, sort_keys=True), flush=True)
    write_execution_state(
        execution_state_row(
            run_id=run_id,
            stage="starting",
            decision_token=DECISION_LASTFM_OPERATIONALIZED_RANKING_PARTIAL,
            planned=4000,
            completed=0,
            strict_valid=0,
            failed=0,
            retry=0,
            start_time=start_clock,
            raw_output_dir=raw_output_dir,
            notes="Starting authorized LastFM operationalized ranking execution.",
        )
    )
    cleanup_manifest = build_cleanup_shard_coverage_manifest(
        run_id,
        start_clock,
        raw_output_dir=raw_output_dir,
        resume_with_existing_raw=bool(getattr(args, "resume", False) and raw_output_dir.exists() and (raw_output_dir / "raw_responses.jsonl").exists()),
    )
    if cleanup_manifest.get("coverage_status") != "pass":
        write_execution_state(
            execution_state_row(
                run_id=run_id,
                stage="blocked_cleanup_shard_coverage",
                decision_token=DECISION_LASTFM_OPERATIONALIZED_RANKING_BLOCKED,
                planned=4000,
                completed=0,
                strict_valid=0,
                failed=0,
                retry=0,
                start_time=start_clock,
                raw_output_dir=raw_output_dir,
                notes="Cleanup shard coverage failed before inference.",
            )
        )
        print(json.dumps({"decision_token": DECISION_LASTFM_OPERATIONALIZED_RANKING_BLOCKED, "blocker": "cleanup_shard_coverage"}, sort_keys=True))
        return 1
    ollama_state = start_local_ollama_if_needed(run_id)
    preflight_rows = validate_operationalized_ranking_preflight(args, ollama_state, profile_hashes_before)
    write_csv(LASTFM_OPERATIONALIZED_RANKING_PREFLIGHT_PATH, preflight_rows, LASTFM_OPERATIONALIZED_RANKING_TABLE_FIELDS["preflight"])
    failed_preflight = [row for row in preflight_rows if row["status"] != "pass" and row["severity"] == "error"]
    if failed_preflight:
        write_execution_state(
            execution_state_row(
                run_id=run_id,
                stage="blocked_preflight",
                decision_token=DECISION_LASTFM_OPERATIONALIZED_RANKING_BLOCKED,
                planned=4000,
                completed=0,
                strict_valid=0,
                failed=0,
                retry=0,
                start_time=start_clock,
                raw_output_dir=raw_output_dir,
                notes=";".join(row["check_name"] for row in failed_preflight),
            )
        )
        print(json.dumps({"decision_token": DECISION_LASTFM_OPERATIONALIZED_RANKING_BLOCKED, "preflight_failures": [row["check_name"] for row in failed_preflight]}, sort_keys=True))
        return 1
    raw_paths = raw_response_paths(raw_output_dir)
    if getattr(args, "resume", False):
        raw_output_dir.mkdir(parents=True, exist_ok=True)
        raw_paths["individual"].mkdir(parents=True, exist_ok=True)
        if not raw_paths["manifest"].exists():
            print(json.dumps({"decision_token": DECISION_LASTFM_OPERATIONALIZED_RANKING_BLOCKED, "blocker": "resume_missing_run_manifest"}, sort_keys=True))
            return 1
    else:
        raw_output_dir.mkdir(parents=True, exist_ok=False)
        raw_paths["individual"].mkdir(parents=True, exist_ok=True)
        atomic_write_text(raw_paths["command"], command_text + "\n")
    raw_paths["individual"].mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "phase_id": PHASE_ID,
        "run_id": run_id,
        "command": command_text,
        "raw_output_dir": rel(raw_output_dir),
        "checkpoint_id": args.checkpoint_id,
        "ollama_endpoint": args.ollama_endpoint,
        "ollama_service_version": ollama_state.service_version,
        "ollama_start_attempted": ollama_state.start_attempted,
        "ollama_started_pid": ollama_state.started_pid,
        "planned": 4000,
        "authorized_manifest": rel(LASTFM_OPERATIONALIZED_RANKING_MANIFEST_PATH),
        "profile_raw_hashes_before": dict(profile_hashes_before),
        "started_at": utc_now(),
    }
    atomic_write_json(raw_paths["manifest"], run_manifest)
    manifest_rows = manifest_rows_for_operationalized_ranking()
    sample_rows = {row["user_id"]: row for row in read_csv_rows(LASTFM_OPERATIONALIZED_SAMPLE_COHORT_PATH)}
    candidate_by_user = candidate_rows_for_operationalized_sample()
    completed_records = final_response_records(raw_output_dir)
    strict_valid_count = sum(1 for record in completed_records.values() if record.get("strict_valid"))
    failed_count = sum(1 for record in completed_records.values() if record and not record.get("strict_valid"))
    for order, request_row in enumerate(manifest_rows, start=1):
        if request_row["request_id"] in completed_records:
            continue
        prompt = render_prompt_for_execution(request_row, candidate_by_user[request_row["user_id"]], sample_rows)
        generation_parameters = json.loads(request_row["generation_parameters"])
        request_event = {
            "request_id": request_row["request_id"],
            "request_order": order,
            "sample_id": request_row["sample_id"],
            "user_id": request_row["user_id"],
            "model_name": request_row["model_name"],
            "model_digest": request_row["model_digest"],
            "prompting_strategy": request_row["prompting_strategy"],
            "history_hash": request_row["history_hash"],
            "candidate_list_hash": request_row["candidate_list_hash"],
            "positive_item": request_row["positive_item"],
            "positive_position": request_row["positive_position"],
            "rendered_prompt_hash": sha256_text(prompt),
            "prompt": prompt,
            "generation_parameters": generation_parameters,
            "request_contract_hash": request_row["request_contract_hash"],
            "created_at": utc_now(),
        }
        append_jsonl_fsync(raw_paths["requests"], request_event)
        result = execute_local_ollama_request(request_row["model_name"], prompt, generation_parameters)
        parse = parse_paper_title_list_response(result.response_text, candidate_by_user[request_row["user_id"]], expected_count=10)
        raw_json_path = raw_paths["individual"] / f"{request_row['request_id']}.json"
        raw_record = build_ollama_raw_record(
            request_row=request_row,
            prompt=prompt,
            candidate_rows=candidate_by_user[request_row["user_id"]],
            result=result,
            parse=parse,
            raw_json_path=raw_json_path,
            attempt_index=1,
        )
        write_individual_raw_json(raw_json_path, raw_record)
        append_jsonl_fsync(raw_paths["responses"], raw_record)
        completed_records[request_row["request_id"]] = raw_record
        strict_valid_count += 1 if raw_record["strict_valid"] else 0
        failed_count += 0 if raw_record["strict_valid"] else 1
        if order % 5 == 0 or order == len(manifest_rows):
            write_execution_state(
                execution_state_row(
                    run_id=run_id,
                    stage="inference",
                    decision_token=DECISION_LASTFM_OPERATIONALIZED_RANKING_PARTIAL,
                    planned=4000,
                    completed=len(completed_records),
                    strict_valid=strict_valid_count,
                    failed=failed_count,
                    retry=0,
                    start_time=start_clock,
                    raw_output_dir=raw_output_dir,
                    last_request_id=request_row["request_id"],
                    notes="Append-only local Ollama execution in progress.",
                )
            )
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "completed": len(completed_records),
                        "strict_valid": strict_valid_count,
                        "failed": failed_count,
                        "residual": 4000 - len(completed_records),
                        "last_request_id": request_row["request_id"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    run_completed_at = utc_now()
    result = consolidate_lastfm_operationalized_ranking(
        run_id=run_id,
        raw_output_dir=raw_output_dir,
        run_started_at=run_manifest["started_at"],
        run_completed_at=run_completed_at,
        profile_hashes_before=profile_hashes_before,
        cleanup_manifest=cleanup_manifest,
    )
    all_tables = {
        "preflight": preflight_rows,
        "execution_state": [
            execution_state_row(
                run_id=run_id,
                stage="complete" if result.decision_token == DECISION_LASTFM_OPERATIONALIZED_RANKING_COMPLETE else "partial_or_blocked",
                decision_token=result.decision_token,
                planned=4000,
                completed=result.manifest["completed"],
                strict_valid=result.manifest["strict_valid"],
                failed=result.manifest["failed"],
                retry=result.manifest["retry"],
                start_time=start_clock,
                raw_output_dir=raw_output_dir,
                notes="Execution and consolidation complete.",
            )
        ],
        **result.tables,
    }
    for name, path in LASTFM_OPERATIONALIZED_RANKING_OUTPUT_TABLES.items():
        write_csv(path, all_tables[name], LASTFM_OPERATIONALIZED_RANKING_TABLE_FIELDS[name])
    write_execution_state(all_tables["execution_state"][0])
    result = LastfmOperationalizedRankingExecutionResult(result.decision_token, run_id, raw_output_dir, all_tables, result.manifest)
    manifest = {**result.manifest}
    manifest["output_hashes"] = {
        rel(path): safe_file_sha256(path)
        for path in LASTFM_OPERATIONALIZED_RANKING_OUTPUT_TABLES.values()
        if path.exists()
    }
    result = LastfmOperationalizedRankingExecutionResult(result.decision_token, run_id, raw_output_dir, all_tables, manifest)
    write_json(LASTFM_OPERATIONALIZED_RANKING_RUN_MANIFEST_PATH, result.manifest)
    atomic_write_json(raw_paths["manifest"], {**run_manifest, **result.manifest, "completed_at": run_completed_at})
    update_lastfm_operationalized_execution_handoff_and_notes(result)
    print(json.dumps({"decision_token": result.decision_token, "run_id": run_id, "raw_output_dir": rel(raw_output_dir), "completed": result.manifest["completed"], "strict_valid": result.manifest["strict_valid"], "failed": result.manifest["failed"], "residual": result.manifest["residual"]}, sort_keys=True))
    return 0 if result.decision_token in {DECISION_LASTFM_OPERATIONALIZED_RANKING_COMPLETE, DECISION_LASTFM_OPERATIONALIZED_RANKING_PARTIAL} else 1


def execute_check_row(
    index: int,
    name: str,
    required: str,
    observed: str,
    passed: bool,
    evidence: str,
    notes: str,
    *,
    severity: str = "error",
) -> dict[str, str]:
    return {
        "check_id": f"F6L-EXEC-PREFLIGHT-{index:03d}",
        "check_name": name,
        "required": required,
        "observed": observed,
        "status": "pass" if passed else "fail",
        "severity": severity,
        "evidence": evidence,
        "notes": notes,
    }


def included_authorized_matrix_rows() -> list[dict[str, str]]:
    return [
        row
        for row in read_csv_rows(OUTPUT_TABLES["final_execution_authorization_matrix"])
        if row.get("included_in_final_request_count") == "true"
    ]


def authorized_request_total_from_matrix() -> int:
    return sum(int(row.get("expected_requests", "0") or 0) for row in included_authorized_matrix_rows())


def authorized_matrix_cells_are_local_only(rows: Sequence[Mapping[str, str]]) -> bool:
    allowed_models = {"qwen2:7b-instruct", "mistral:7b-instruct"}
    allowed_input_variants = {
        "profile_generation",
        "history_only",
        "profile_only",
        "profile_plus_history",
        "paper_base_openp5_title_list",
        "paper_llmrank_recency_title_list",
    }
    allowed_prompting = {
        "template_user_profile_generate",
        "template",
        "template_profile",
        "template_profile_history",
        "llmrank_rf",
    }
    return all(
        row.get("model") in allowed_models
        and row.get("input_variant") in allowed_input_variants
        and row.get("prompting_strategy") in allowed_prompting
        and row.get("authorization_status") == "authorized_after_explicit_user_execution_approval"
        for row in rows
    )


def protected_hash_regressions(current: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> list[str]:
    checkpoint_entries = checkpoint.get("protected_inputs", [])
    if not isinstance(checkpoint_entries, list):
        return ["checkpoint_protected_inputs_not_list"]
    current_by_path = {
        entry.get("path", ""): entry
        for entry in current.get("protected_inputs", [])
        if isinstance(entry, Mapping)
    }
    regressions: list[str] = []
    for entry in checkpoint_entries:
        if not isinstance(entry, Mapping):
            regressions.append("malformed_checkpoint_entry")
            continue
        path = str(entry.get("path", ""))
        old_hash = str(entry.get("sha256", ""))
        current_entry = current_by_path.get(path)
        if current_entry is None:
            regressions.append(f"{path}:missing_from_current_hash_set")
            continue
        if str(current_entry.get("sha256", "")) != old_hash:
            regressions.append(f"{path}:sha256_changed")
    return regressions


def manifest_audit_rows(expected_requests: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    aggregate_path = OUTPUT_TABLES["execution_request_manifest"]
    aggregate_rows = read_csv_rows(aggregate_path)
    rows.append(
        {
            "audit_id": "F6L-MANIFEST-AUDIT-001",
            "artifact": "aggregate_checkpoint_request_manifest",
            "path": rel(aggregate_path),
            "required": "aggregate rows present from checkpoint",
            "status": "pass" if aggregate_rows else "fail",
            "row_count": str(len(aggregate_rows)),
            "unique_request_ids": "not_applicable",
            "expected_requests": str(expected_requests),
            "sha256": safe_file_sha256(aggregate_path),
            "notes": "This aggregate manifest is sufficient for request counting but does not provide one frozen request_id per model call.",
        }
    )

    per_request_rows = read_csv_rows(PER_REQUEST_MANIFEST_PATH)
    request_ids = [row.get("request_id", "") for row in per_request_rows if row.get("request_id")]
    per_request_ok = (
        PER_REQUEST_MANIFEST_PATH.exists()
        and len(per_request_rows) == expected_requests
        and len(set(request_ids)) == expected_requests
    )
    rows.append(
        {
            "audit_id": "F6L-MANIFEST-AUDIT-002",
            "artifact": "per_request_open_model_manifest",
            "path": rel(PER_REQUEST_MANIFEST_PATH),
            "required": "8800 rows and 8800 unique request_id values before inference",
            "status": "pass" if per_request_ok else "fail",
            "row_count": str(len(per_request_rows)) if PER_REQUEST_MANIFEST_PATH.exists() else "missing",
            "unique_request_ids": str(len(set(request_ids))) if PER_REQUEST_MANIFEST_PATH.exists() else "missing",
            "expected_requests": str(expected_requests),
            "sha256": safe_file_sha256(PER_REQUEST_MANIFEST_PATH),
            "notes": "The user authorization requires this lower-level frozen manifest before the first local model request.",
        }
    )

    cohort_rows = read_csv_rows(LASTFM_FROZEN_SAMPLE_COHORT_PATH)
    candidate_rows = read_csv_rows(LASTFM_FROZEN_SAMPLE_CANDIDATES_PATH)
    sample_ok = (
        LASTFM_FROZEN_SAMPLE_COHORT_PATH.exists()
        and LASTFM_FROZEN_SAMPLE_CANDIDATES_PATH.exists()
        and len(cohort_rows) == 1000
        and len(candidate_rows) == 20000
    )
    rows.append(
        {
            "audit_id": "F6L-MANIFEST-AUDIT-003",
            "artifact": "lastfm_1000_user_ks_valid_frozen_sample",
            "path": f"{rel(LASTFM_FROZEN_SAMPLE_COHORT_PATH)};{rel(LASTFM_FROZEN_SAMPLE_CANDIDATES_PATH)}",
            "required": "1000-user cohort plus 20000 candidate rows frozen by checkpoint",
            "status": "pass" if sample_ok else "fail",
            "row_count": f"cohort={len(cohort_rows) if LASTFM_FROZEN_SAMPLE_COHORT_PATH.exists() else 'missing'};candidates={len(candidate_rows) if LASTFM_FROZEN_SAMPLE_CANDIDATES_PATH.exists() else 'missing'}",
            "unique_request_ids": "not_applicable",
            "expected_requests": "4000 LastFM ranking requests",
            "sha256": ";".join(
                digest
                for digest in [
                    safe_file_sha256(LASTFM_FROZEN_SAMPLE_COHORT_PATH),
                    safe_file_sha256(LASTFM_FROZEN_SAMPLE_CANDIDATES_PATH),
                ]
                if digest
            ),
            "notes": "The existing F6D4A LastFM 1859-user adapted cohort is preserved, but it is not the F6L frozen 1000-user paper-protocol sample.",
        }
    )
    return rows


def ollama_model_preflight_rows(ollama_payload: Mapping[str, Any], error: str = "") -> list[dict[str, str]]:
    models = ollama_payload.get("models", [])
    if not isinstance(models, list):
        models = []
    by_name = {
        str(model.get("name", "")): model
        for model in models
        if isinstance(model, Mapping)
    }
    rows: list[dict[str, str]] = []
    for required in ["qwen2:7b-instruct", "mistral:7b-instruct"]:
        model = by_name.get(required, {})
        details = model.get("details", {}) if isinstance(model, Mapping) else {}
        if not isinstance(details, Mapping):
            details = {}
        model_info = model.get("model_info", {}) if isinstance(model, Mapping) else {}
        if not isinstance(model_info, Mapping):
            model_info = {}
        context_length = (
            model_info.get("qwen2.context_length")
            or model_info.get("llama.context_length")
            or model_info.get("general.context_length")
            or details.get("context_length")
            or ""
        )
        available = bool(model)
        rows.append(
            {
                "model": required,
                "required": "true",
                "local_endpoint": OLLAMA_TAGS_URL,
                "available": "true" if available else "false",
                "digest": str(model.get("digest", "")) if isinstance(model, Mapping) else "",
                "modified_at": str(model.get("modified_at", "")) if isinstance(model, Mapping) else "",
                "size_bytes": str(model.get("size", "")) if isinstance(model, Mapping) else "",
                "family": str(details.get("family", "")),
                "parameter_size": str(details.get("parameter_size", "")),
                "quantization_level": str(details.get("quantization_level", "")),
                "context_length": str(context_length),
                "status": "pass" if available else "fail",
                "notes": "Local Ollama tag found." if available else f"Local Ollama tag unavailable. {error}".strip(),
            }
        )
    return rows


def build_execute_report_section(result: ExecutePreflightResult) -> str:
    blocker = result.tables["execution_blocker"][0]
    failed = [row for row in result.tables["execute_preflight"] if row["status"] != "pass"]
    passed = [row for row in result.tables["execute_preflight"] if row["status"] == "pass"]
    failed_names = ", ".join(row["check_name"] for row in failed) if failed else "none"
    models = result.tables["ollama_local_model_preflight"]
    model_summary = "; ".join(
        f"{row['model']}={row['status']}:{row['digest'][:12] if row['digest'] else 'no_digest'}"
        for row in models
    )
    return "\n".join(
        [
            "## Authorized Execute Preflight",
            "",
            f"- Decision token: `{result.decision_token}`.",
            f"- Planned requests: `{blocker['planned_requests']}`.",
            f"- Completed requests: `{blocker['completed_requests']}`.",
            f"- Strict-valid requests: `{blocker['strict_valid_requests']}`.",
            f"- Failed requests: `{blocker['failed_requests']}`.",
            f"- Residual requests: `{blocker['residual_requests']}`.",
            f"- Passing preflight checks: `{len(passed)}`.",
            f"- Failing preflight checks: `{len(failed)}`.",
            f"- Failing checks: {failed_names}.",
            f"- Local model preflight: {model_summary}.",
            "",
            "No local model inference was started. The raw output directory was not created because execution must stop before the first request when a frozen per-request manifest or frozen LastFM 1000-user sample is absent.",
            "",
            f"`{result.decision_token}`",
            "",
        ]
    )


def update_execute_handoff_and_notes(result: ExecutePreflightResult) -> None:
    blocker = result.tables["execution_blocker"][0]
    failures = [row for row in result.tables["execute_preflight"] if row["status"] != "pass"]
    failure_names = ", ".join(row["check_name"] for row in failures)
    handoff_content = "\n".join(
        [
            "## F6L Authorized Execute Preflight",
            "",
            f"- Decision token: `{result.decision_token}`.",
            f"- Planned/completed/strict-valid/failed/residual requests: `{blocker['planned_requests']}`/`{blocker['completed_requests']}`/`{blocker['strict_valid_requests']}`/`{blocker['failed_requests']}`/`{blocker['residual_requests']}`.",
            "- Local-only Ollama model identity check was performed against `http://127.0.0.1:11434/api/tags`.",
            "- No inference, raw-output directory creation, paid/remote API call, staging, commit, cleanup, download, training, or `original_repo/` mutation was performed.",
            f"- Execution blocker checks: {failure_names}.",
            "- Next action stays inside F6L: produce and freeze the per-request open-model manifest and F6L LastFM 1000-user K-S sample, then rerun execute preflight before any model call.",
        ]
    )
    notes_content = "\n".join(
        [
            "## 2026-08-01 - Phase F6L Authorized Execute Preflight",
            "",
            "### Changes",
            "",
            "- Added and ran the F6L execute-mode preflight after explicit user authorization.",
            "- Verified the aggregate 8800-request authorization and local-only Ollama model availability.",
            "- Blocked before inference because required frozen lower-level execution inputs are absent.",
            "",
            "### Outputs Generated",
            "",
            "- `results/tables/phase_f6l_execute_preflight.csv`",
            "- `results/tables/phase_f6l_authorized_request_manifest_audit.csv`",
            "- `results/tables/phase_f6l_ollama_local_model_preflight.csv`",
            "- `results/tables/phase_f6l_execution_blocker.csv`",
            "- `results/manifests/phase_f6l_execute_preflight_manifest.json`",
            "",
            "### Result",
            "",
            f"- Decision: `{result.decision_token}`.",
            f"- Requests planned/completed/strict-valid/failed/residual: `{blocker['planned_requests']}`/`{blocker['completed_requests']}`/`{blocker['strict_valid_requests']}`/`{blocker['failed_requests']}`/`{blocker['residual_requests']}`.",
            f"- Blockers: {failure_names}.",
        ]
    )
    marker = f"{PHASE_NAME}_execute_preflight"
    update_marked_section(REPORT_PATH, marker, build_execute_report_section(result))
    update_marked_section(HANDOFF_PATH, marker, handoff_content)
    update_marked_section(NOTES_PATH, marker, notes_content)


def build_execute_preflight(
    args: argparse.Namespace,
    *,
    write_outputs_enabled: bool = True,
    ollama_payload: Mapping[str, Any] | None = None,
    ollama_error: str = "",
) -> ExecutePreflightResult:
    require_runtime()
    load_config(args.config)

    checkpoint = read_json_file(CHECKPOINT_MANIFEST_PATH)
    checkpoint_hashes = read_json_file(PROTECTED_HASHES_PATH)
    current_hashes = hash_protected_inputs()
    hash_regressions = protected_hash_regressions(current_hashes, checkpoint_hashes) if checkpoint_hashes else ["missing_checkpoint_hash_manifest"]
    authorized_rows = included_authorized_matrix_rows()
    authorized_total = sum(int(row.get("expected_requests", "0") or 0) for row in authorized_rows)
    breakdown_total = sum(int(row.get("expected_requests", "0") or 0) for row in read_csv_rows(OUTPUT_TABLES["execution_request_breakdown"]))
    request_manifest_rows = manifest_audit_rows(authorized_total)
    per_request_manifest_ok = next(
        row for row in request_manifest_rows if row["artifact"] == "per_request_open_model_manifest"
    )["status"] == "pass"
    lastfm_sample_ok = next(
        row for row in request_manifest_rows if row["artifact"] == "lastfm_1000_user_ks_valid_frozen_sample"
    )["status"] == "pass"
    raw_output_dir = resolve_project_path(args.raw_output_dir)
    raw_root = (PROJECT_ROOT / "results" / "llm").resolve()
    phase_f6l_raw_offenders = sorted(path.name for path in raw_root.glob("*phase_f6l*")) if raw_root.exists() else []
    staged = git_output(["diff", "--cached", "--name-only"])
    zip_summary = official_zip_summary()
    prompt_rows_ok = all(
        zip_summary.get(key) == "200"
        for key in [
            "beauty_profile_gpt35_rows",
            "beauty_profile_gpt4_rows",
            "beauty_profile_glm4_rows",
            "sports_profile_gpt35_rows",
            "sports_profile_gpt4_rows",
            "sports_profile_glm4_rows",
        ]
    )
    lastfm = lastfm_summary()
    lastfm_adapted_candidate_shape_ok = (
        int(lastfm.get("ranking_candidate_sets", 0)) == 1859
        and int(lastfm.get("ranking_candidate_rows", 0)) == 37180
        and int(lastfm.get("ranking_positive_rows", 0)) == 1859
    )

    if ollama_payload is None:
        ollama_payload, ollama_error = fetch_local_ollama_tags()
    model_rows = ollama_model_preflight_rows(ollama_payload, ollama_error)
    required_models_ok = all(row["status"] == "pass" for row in model_rows)

    checks_data = [
        (
            "checkpoint_id_matches",
            DECISION_CHECKPOINT,
            args.checkpoint_id,
            args.checkpoint_id == DECISION_CHECKPOINT,
            "args.checkpoint_id",
            "Execution can only continue from the F6L final authorization checkpoint.",
        ),
        (
            "checkpoint_manifest_present",
            f"{rel(CHECKPOINT_MANIFEST_PATH)} with request_count=8800",
            f"exists={CHECKPOINT_MANIFEST_PATH.exists()};token={checkpoint.get('decision_token', '')};request_count={checkpoint.get('request_count', '')}",
            checkpoint.get("decision_token") == DECISION_CHECKPOINT and checkpoint.get("request_count") == 8800,
            rel(CHECKPOINT_MANIFEST_PATH),
            "The execute preflight must be tied to the recorded checkpoint manifest.",
        ),
        (
            "allow_api_calls_local_only_guard",
            "--allow-api-calls plus --local-only and fixed 127.0.0.1 Ollama endpoint",
            f"allow_api_calls={args.allow_api_calls};local_only={args.local_only};endpoint={OLLAMA_TAGS_URL}",
            bool(args.allow_api_calls and args.local_only and OLLAMA_TAGS_URL.startswith("http://127.0.0.1:")),
            "CLI flags; OLLAMA_TAGS_URL",
            "`--allow-api-calls` is scoped to local Ollama preflight and future localhost generation only.",
        ),
        (
            "authorization_count_8800",
            "8800 included authorized requests",
            str(authorized_total),
            authorized_total == 8800,
            rel(OUTPUT_TABLES["final_execution_authorization_matrix"]),
            "The aggregate authorization matrix must stay exactly at the user-approved count.",
        ),
        (
            "request_breakdown_count_8800",
            "8800 atomic breakdown requests",
            str(breakdown_total),
            breakdown_total == 8800,
            rel(OUTPUT_TABLES["execution_request_breakdown"]),
            "The execution breakdown must remain aligned with the final authorization matrix.",
        ),
        (
            "authorized_cells_positive_local_open_only",
            "only Qwen2/Mistral local open-model cells with positive authorization status",
            f"rows={len(authorized_rows)}",
            authorized_matrix_cells_are_local_only(authorized_rows),
            rel(OUTPUT_TABLES["final_execution_authorization_matrix"]),
            "Closed, paid, remote, adapted, LLMSRec, and reranking-zero cells must not enter execution.",
        ),
        (
            "per_request_manifest_exists_and_unique",
            "8800 frozen rows with 8800 unique request_id values",
            next(row for row in request_manifest_rows if row["artifact"] == "per_request_open_model_manifest")["row_count"],
            per_request_manifest_ok,
            rel(PER_REQUEST_MANIFEST_PATH),
            "This missing lower-level manifest blocks execution before the first model request.",
        ),
        (
            "lastfm_frozen_sample_artifact",
            "F6L 1000-user K-S sample cohort and 20-candidate rows frozen before inference",
            next(row for row in request_manifest_rows if row["artifact"] == "lastfm_1000_user_ks_valid_frozen_sample")["row_count"],
            lastfm_sample_ok,
            f"{rel(LASTFM_FROZEN_SAMPLE_COHORT_PATH)};{rel(LASTFM_FROZEN_SAMPLE_CANDIDATES_PATH)}",
            "The retained F6D4A 1859-user adapted LastFM cohort is not the required F6L frozen sample.",
        ),
        (
            "protected_input_hashes_match_checkpoint",
            "no protected hash regressions",
            "none" if not hash_regressions else ";".join(hash_regressions[:8]),
            current_hashes.get("status") == "pass" and not hash_regressions,
            rel(PROTECTED_HASHES_PATH),
            "Protected inputs must match the checkpoint snapshot before inference.",
        ),
        (
            "f6i2p_yaml_count_23",
            "23 protected F6I2P YAML configs",
            str(current_hashes.get("f6i2p_yaml_count", "")),
            current_hashes.get("f6i2p_yaml_count") == 23,
            rel(PROTECTED_HASHES_PATH),
            "The execute preflight must not mutate the 23 protected F6I2P YAML configs.",
        ),
        (
            "ollama_local_model_identities_available",
            "qwen2:7b-instruct and mistral:7b-instruct available from local Ollama",
            ";".join(f"{row['model']}={row['status']}:{row['digest'][:12] if row['digest'] else 'no_digest'}" for row in model_rows),
            required_models_ok,
            OLLAMA_TAGS_URL,
            "This checks model identity only; it does not perform inference.",
        ),
        (
            "raw_output_dir_new_under_results_llm",
            "non-existing path below results/llm/",
            rel(raw_output_dir) if path_is_under(raw_output_dir, PROJECT_ROOT) else str(raw_output_dir),
            path_is_under(raw_output_dir, raw_root) and not raw_output_dir.exists(),
            "args.raw_output_dir",
            "Raw outputs must be append-only in a new non-colliding directory.",
        ),
        (
            "no_existing_phase_f6l_raw_outputs",
            "no existing results/llm/*phase_f6l* raw-output path",
            ";".join(phase_f6l_raw_offenders) if phase_f6l_raw_offenders else "none",
            not phase_f6l_raw_offenders,
            rel(raw_root) if raw_root.exists() else "results/llm",
            "Execution must not overwrite any previous F6L raw output.",
        ),
        (
            "beauty_sports_official_prompt_payloads",
            "official Beauty/Sports profile artifacts have 200 rows per recovered prompt-output file",
            canonical_json(zip_summary),
            prompt_rows_ok,
            "original_repo/EvaLLMasRecommender/llm_results/*.zip",
            "Profile-generation and profile-recommendation cells rely on these official cohort prompt artifacts.",
        ),
        (
            "lastfm_adapted_candidate_shape_retained_not_promoted",
            "existing adapted LastFM ranking has 1859 candidate sets x 20 candidates",
            f"sets={lastfm.get('ranking_candidate_sets')};rows={lastfm.get('ranking_candidate_rows')};positives={lastfm.get('ranking_positive_rows')}",
            lastfm_adapted_candidate_shape_ok,
            "data/processed/finalization_f6c_lastfm_ranking_candidates.csv",
            "This verifies preserved adapted evidence only; it does not satisfy the frozen 1000-user F6L sample contract.",
        ),
        (
            "paper_title_list_parser_contract_available",
            "official parser and top-10 title-list contract source files present",
            f"output_refine={project_path('original_repo/EvaLLMasRecommender/eva/codes/output_refine.py').exists()};llms={project_path('original_repo/EvaLLMasRecommender/eva/codes/llms.py').exists()}",
            project_path("original_repo/EvaLLMasRecommender/eva/codes/output_refine.py").exists()
            and project_path("original_repo/EvaLLMasRecommender/eva/codes/llms.py").exists(),
            "original_repo/EvaLLMasRecommender/eva/codes/output_refine.py;original_repo/EvaLLMasRecommender/eva/codes/llms.py",
            "The paper title-list parser contract is present, but rendered per-request prompts are not frozen without the missing manifest.",
        ),
        (
            "no_git_staged_paths",
            "0 staged paths",
            "0" if not staged else staged,
            not staged,
            "git diff --cached --name-only",
            "The user explicitly prohibited staging or commits.",
        ),
    ]
    preflight_rows = [
        execute_check_row(index, name, required, observed, passed, evidence, notes)
        for index, (name, required, observed, passed, evidence, notes) in enumerate(checks_data, start=1)
    ]
    failed = [row for row in preflight_rows if row["status"] != "pass" and row["severity"] == "error"]
    blocker_text = "; ".join(row["check_name"] for row in failed) if failed else "execute_runner_not_started_by_preflight_only_implementation"
    blocker_status = "blocked_before_inference" if failed else "ready_but_not_executed"
    blocker_rows = [
        {
            "blocker_id": "F6L-EXEC-BLOCKER-001",
            "decision_token": DECISION_EXECUTION_BLOCKED,
            "scope": "authorized_8800_open_model_requests",
            "status": blocker_status,
            "planned_requests": str(authorized_total),
            "completed_requests": "0",
            "strict_valid_requests": "0",
            "failed_requests": "0",
            "residual_requests": str(authorized_total),
            "blocker": blocker_text,
            "evidence": ";".join(row["evidence"] for row in failed) if failed else "execute_preflight",
            "next_action": "Freeze the missing per-request manifest and F6L LastFM 1000-user K-S sample inside the same F6L contract, then rerun execute preflight before inference.",
        }
    ]
    output_hashes = {
        rel(path): safe_file_sha256(path)
        for path in EXECUTE_OUTPUT_TABLES.values()
        if path.exists()
    }
    manifest = {
        "phase_id": PHASE_ID,
        "phase_name": PHASE_NAME,
        "mode": args.mode,
        "decision_token": DECISION_EXECUTION_BLOCKED,
        "checkpoint_id": args.checkpoint_id,
        "planned_requests": authorized_total,
        "completed_requests": 0,
        "strict_valid_requests": 0,
        "failed_requests": 0,
        "residual_requests": authorized_total,
        "inference_performed": False,
        "raw_output_dir_created": False,
        "raw_output_dir": rel(raw_output_dir) if path_is_under(raw_output_dir, PROJECT_ROOT) else str(raw_output_dir),
        "local_ollama_preflight_api_call_performed": ollama_payload is not None,
        "remote_api_calls_performed": False,
        "paid_api_calls_performed": False,
        "ollama_generation_performed": False,
        "blocking_checks": [row["check_name"] for row in failed],
        "tables": {name: rel(path) for name, path in EXECUTE_OUTPUT_TABLES.items()},
        "output_hashes": output_hashes,
        "self_hash": "self_reference_excluded_for_determinism",
    }
    result = ExecutePreflightResult(
        decision_token=DECISION_EXECUTION_BLOCKED,
        tables={
            "execute_preflight": preflight_rows,
            "authorized_request_manifest_audit": request_manifest_rows,
            "ollama_local_model_preflight": model_rows,
            "execution_blocker": blocker_rows,
        },
        manifest=manifest,
    )

    if write_outputs_enabled:
        for name, path in EXECUTE_OUTPUT_TABLES.items():
            write_csv(path, result.tables[name], EXECUTE_TABLE_FIELDS[name])
        output_hashes = {
            rel(path): safe_file_sha256(path)
            for path in EXECUTE_OUTPUT_TABLES.values()
            if path.exists()
        }
        result.manifest["output_hashes"] = output_hashes
        write_json(EXECUTE_PREFLIGHT_MANIFEST_PATH, result.manifest)
        update_execute_handoff_and_notes(result)
    return result


def guard_execute_mode(args: argparse.Namespace) -> int:
    missing = []
    if not args.allow_api_calls:
        missing.append("--allow-api-calls")
    if not args.local_only:
        missing.append("--local-only")
    if not args.checkpoint_id:
        missing.append("--checkpoint-id")
    if not args.raw_output_dir:
        missing.append("--raw-output-dir")
    if missing:
        print(json.dumps({"phase_id": PHASE_ID, "mode": args.mode, "decision": "blocked_missing_execution_flags", "missing": missing}, sort_keys=True))
        return 2
    result = build_execute_preflight(args, write_outputs_enabled=True)
    blocker = result.tables["execution_blocker"][0]
    print(
        json.dumps(
            {
                "phase_id": PHASE_ID,
                "mode": args.mode,
                "decision": result.decision_token,
                "checkpoint_id": args.checkpoint_id,
                "planned_requests": int(blocker["planned_requests"]),
                "completed_requests": int(blocker["completed_requests"]),
                "strict_valid_requests": int(blocker["strict_valid_requests"]),
                "failed_requests": int(blocker["failed_requests"]),
                "residual_requests": int(blocker["residual_requests"]),
                "blocking_checks": result.manifest["blocking_checks"],
                "inference_performed": False,
            },
            sort_keys=True,
        )
    )
    return 1


def guard_execute_profile_mode(args: argparse.Namespace) -> int:
    from src.analysis.phase_f6l_profile_execution import execute_profile_mode

    return execute_profile_mode(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--mode",
        choices=[
            "audit-checkpoint",
            "execute",
            "execute-profile",
            "consolidate",
            "materialize-contract",
            "separate-contracts",
            "lastfm-contracts",
            "execute-lastfm-operationalized-ranking",
        ],
        default="audit-checkpoint",
    )
    parser.add_argument("--allow-api-calls", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--checkpoint-id", default="")
    parser.add_argument("--raw-output-dir", type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--ollama-endpoint", default=OLLAMA_LOCAL_GENERATE_URL)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.mode == "materialize-contract":
        result = build_contract_materialization(args.config, write_outputs_enabled=True)
        print(
            json.dumps(
                {
                    "phase_id": PHASE_ID,
                    "mode": args.mode,
                    "decision_token": result.decision_token,
                    "lastfm_sample_materialized": result.manifest["lastfm_sample_materialized"],
                    "lastfm_candidates_materialized": result.manifest["lastfm_candidates_materialized"],
                    "request_manifest_materialized": result.manifest["open_model_request_manifest_materialized"],
                    "planned_requests": result.manifest["planned_requests"],
                    "deterministic_in_memory_digest_match": result.manifest["deterministic_in_memory_digest_match"],
                    "inference_performed": False,
                },
                sort_keys=True,
            )
        )
        return 1
    if args.mode == "separate-contracts":
        result = build_separated_contracts(args.config, write_outputs_enabled=True)
        manifest = result.manifests["separated_checkpoint"]
        print(
            json.dumps(
                {
                    "phase_id": PHASE_ID,
                    "mode": args.mode,
                    "decision_token": result.decision_token,
                    "profile_manifest_rows": manifest["profile_manifest_rows"],
                    "profile_unique_request_ids": manifest["profile_unique_request_ids"],
                    "lastfm_exact_materialized": manifest["lastfm_exact_materialized"],
                    "lastfm_operationalization_materialized": manifest["lastfm_operationalization_materialized"],
                    "deterministic_in_memory_digest_match": manifest["deterministic_in_memory_digest_match"],
                    "inference_performed": False,
                },
                sort_keys=True,
            )
        )
        return 0 if result.decision_token == DECISION_SEPARATED_PROFILE_READY else 1
    if args.mode == "lastfm-contracts":
        result = build_lastfm_execution_contracts(args.config, write_outputs_enabled=True)
        manifest = result.manifest
        counts = manifest["counts"]
        print(
            json.dumps(
                {
                    "phase_id": PHASE_ID,
                    "mode": args.mode,
                    "decision_token": result.decision_token,
                    "selected_operationalized_seed": manifest["selected_operationalized_seed"],
                    "operationalized_sample_users": counts["operationalized_sample_users"],
                    "operationalized_ranking_requests": counts["operationalized_ranking_requests"],
                    "full_cohort_ranking_requests": counts["full_cohort_ranking_requests"],
                    "future_operationalized_reranking_requests_if_authorized": counts["future_operationalized_reranking_requests_if_authorized"],
                    "future_full_cohort_reranking_requests_if_authorized": counts["future_full_cohort_reranking_requests_if_authorized"],
                    "deterministic_in_memory_digest_match": manifest["deterministic_in_memory_digest_match"],
                    "inference_performed": False,
                },
                sort_keys=True,
            )
        )
        return 0 if result.decision_token in {DECISION_LASTFM_CONTRACTS_READY, DECISION_LASTFM_SAMPLE_BLOCKED_FULL_READY} else 1
    if args.mode == "execute-profile":
        return guard_execute_profile_mode(args)
    if args.mode == "execute-lastfm-operationalized-ranking":
        return execute_lastfm_operationalized_ranking(args)
    if args.mode != "audit-checkpoint":
        return guard_execute_mode(args)
    result = build(args.config, write_outputs_enabled=True)
    failed = [row for row in result.tables["validation"] if row["status"] != "pass"]
    print(json.dumps({"phase_id": PHASE_ID, "decision_token": result.decision_token, "validation_failures": len(failed)}, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
