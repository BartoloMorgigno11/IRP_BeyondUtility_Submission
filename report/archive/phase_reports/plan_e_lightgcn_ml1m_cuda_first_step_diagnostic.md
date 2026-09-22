# Plan E LightGCN ML-1M CUDA First-Step Divergence Diagnostic

## 1. Scope and provenance
- Branch: `plan-d-extensions`.
- Commit: `02a20b8b509b8cf834a8766dd1cf1f751ae204bf Add Plan E3B BM25-like scoring on E2 manifests`.
- Diagnostic ID: `plan_e_lightgcn_ml1m_cuda_first_step_diagnostic`.

## 2. Executive summary
- D2 classification: `cuda_sparse_forward_tensor_nondeterminism`.
- Scale-up decision: `approve_cpu_only_scale_up`.
- First true tensor divergence stage: `layer_1_all_embeddings`.

## 3. Phase 3M and D1 evidence
- Phase 3M status from D1: `completed_with_determinism_warning`; D1 classification: `cuda_sparse_training_nondeterministic`.
- D1 bounded CUDA state hashes are treated as legacy evidence and are not rewritten.

## 4. Protected-input boundaries
- Known Phase 3M hashes and D1 snapshot hashes were verified before and after execution.

## 5. Exact first-batch reconstruction
- Training hash `1871DE0C52382305B2AA54187300B68889B34551635E4991E6368E556B39D870`; raw-edge hash `6881403F6C5A78BDEF06EED0E54E8762CBD18945ED70521CEE5502969BEAEDD7`.
- First-batch user/positive/negative hashes: `07A38F27AF74FB5696A8220F02FC77D39532D9BC721EFD34A19FC7C83624685F`, `7A7878F04542568F3E0C43C82B32741D82ED8E4D151BE2C9738A5339738DF214`, `F47A5004CA04864D8F0A989CB7309499C289145FBA38809F64B1DD4FA64072BF`.

## 6. Canonical initial-state construction
- Legacy initial hash `42A90B864CB8FDD450E103560B47B4C618E327BA8DE40ABE65804379E35D5E65`; canonical initial hash `75E447A88D6B9AC1FB37513A40C46A2845356268BD4DB8C3BD0DAC49A2D1CC9C`.

## 7. Legacy state-hash audit
- Hash-artifact rows detected: `0`.

## 8. Canonical tensor-content hashing
- Tensor hash method: `sha256_semantic_name_dtype_shape_byte_order_cpu_contiguous_bytes`.

## 9. Optimizer-state parameter-name mapping
- Optimizer state is mapped through `model.named_parameters()` and excludes raw optimizer parameter IDs from canonical hashes.

## 10. Phase 3M Adam implementation
- Source object: `class Adam(Optimizer):`.
- Defaults: lr `0.001`, betas `0.9`/`0.999`, eps `1e-08`, weight_decay `0.0`, foreach `None`, fused `None`.
- With foreach/fused left as `None`, the CUDA runtime path is automatic; D2 does not assume it is single-tensor.

## 11. Original-pipeline twin design
- Two replicas loaded the same canonical CPU state, used original Adam defaults, ran forward, backward and one optimizer step.

## 12. Forward-only twin findings
- Status `tensor_divergence`; first divergence `layer_1_all_embeddings`; max absolute difference `1.862645149230957e-09`.

## 13. Forward-backward twin findings
- Status `tensor_divergence`; first divergence `layer_1_all_embeddings`; max absolute difference `1.862645149230957e-09`.

## 14. Shared-gradient default-Adam findings
- Status `exact`; first divergence `none`; max absolute difference `0.0`.

## 15. Shared-gradient foreach-false findings
- Status `exact`; first divergence `none`; max absolute difference `0.0`.

## 16. Shared-gradient foreach-true findings
- Status `exact`; first divergence `none`; max absolute difference `0.0`.

## 17. Graph and input identity
- Batch and graph stages were compared directly in every variant; see stage comparison CSV.

## 18. Initial-parameter identity
- Initial user/item parameter stages are exact when variants completed.

## 19. Sparse propagation layer 1
- `layer_1_all_embeddings` status `different`, mismatches `232893`, max abs diff `1.1641532182693481e-09`.

## 20. Sparse propagation layer 2
- `layer_2_all_embeddings` status `different`, mismatches `415785`, max abs diff `5.820766091346741e-10`.

## 21. Final propagated embeddings
- `final_all_embeddings` status `different`, mismatches `51152`, max abs diff `1.862645149230957e-09`.

## 22. Positive and negative scores
- `positive_scores` status `different`, mismatches `2104`, max abs diff `3.637978807091713e-11`.
- `negative_scores` status `different`, mismatches `1571`, max abs diff `2.9103830456733704e-11`.

## 23. Loss-tensor identity
- `total_loss_tensor` status `exact`, mismatches `0`, max abs diff `0.0`.

## 24. User-embedding gradients
- `user_embedding_gradient` status `different`, mismatches `175216`, max abs diff `3.410605131648481e-13`.

## 25. Item-embedding gradients
- `item_embedding_gradient` status `different`, mismatches `68519`, max abs diff `2.2737367544323206e-13`.

## 26. Post-step parameter identity
- `post_step_user_embedding_parameter` status `different`, mismatches `26546`, max abs diff `6.984919309616089e-09`.
- `post_step_item_embedding_parameter` status `different`, mismatches `11009`, max abs diff `3.725290298461914e-09`.

## 27. Adam step state
- `user_step` status `exact`, mismatches `0`, max abs diff `0.0`.
- `item_step` status `exact`, mismatches `0`, max abs diff `0.0`.

## 28. Adam first-moment state
- `user_exp_avg` status `different`, mismatches `163407`, max abs diff `2.842170943040401e-14`.
- `item_exp_avg` status `different`, mismatches `63214`, max abs diff `2.842170943040401e-14`.

## 29. Adam second-moment state
- `user_exp_avg_sq` status `different`, mismatches `175207`, max abs diff `8.470329472543003e-22`.
- `item_exp_avg_sq` status `different`, mismatches `68518`, max abs diff `5.293955920339377e-22`.

## 30. First genuine divergence
- `layer_1_all_embeddings`.

## 31. Hash artifact assessment
- Legacy hash artifact detected: `False`; optimizer parameter-ID artifact detected: `False`.

## 32. Refinement of the D1 classification
- D1 `cuda_sparse_training_nondeterministic` refined to D2 `cuda_sparse_forward_tensor_nondeterminism`.

## 33. Whether the default Adam path is implicated
- Default Adam exact: `True`.

## 34. Whether foreach=False resolves the divergence
- foreach=False Adam exact: `True`.

## 35. Scientific status of Phase 3M
- Primary outputs remain valid: `True`; D2 did not rescore candidates.

## 36. Scale-up decision
- `approve_cpu_only_scale_up`.

## 37. Required next corrective action
- `Phase 3N-CPU - controlled CPU LightGCN Beauty and Sports execution and same-cohort consolidation`.

## 38. Limitations
- Only first-batch CUDA behavior is isolated; no complete epoch and no corrected run are executed.

## 39. Protected-input immutability
- Protected inputs were verified unchanged; no checkpoint or ranking output was created.

## 40. Commands executed
- `git status --short --branch`
- approved-interpreter pytest commands
- approved-interpreter D2 diagnostic runner

## 41. Recommended single next task
- Phase 3N-CPU - controlled CPU LightGCN Beauty and Sports execution and same-cohort consolidation
