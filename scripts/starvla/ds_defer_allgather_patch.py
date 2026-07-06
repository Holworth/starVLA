# [OPT #8, docs/qwenpi_zero2_h200_final_report.md]
# Defer the ZeRO-2 post-step param allgather of the action-head param group.
#
# DeepSpeed stage-2 `step()` ends with one allgather per param group, all
# stream-synchronous, so the next forward waits for BOTH gathers (~44 ms/step
# on 8x H200) even though the DiT action head's params are not needed until
# ~200 ms into the step. This patch keeps the LARGEST group (the VLM backbone,
# needed immediately by forward) synchronous and launches the remaining
# groups' gathers async; the matching wait() is inserted at the entry of
# LayerwiseFlowmatchingActionHead.forward, so the VLM forward hides the
# gather. Enabled via STARVLA_DEFER_AG=1 (imported by profile_entry).
#
# Safety: if the wait were skipped, the head would read a partially-gathered
# flat buffer and the loss would visibly diverge — verified unchanged in
# bench logs. Group selection is by numel (largest = backbone), not by group
# order, so a param-group reordering cannot silently defer the backbone.
import deepspeed.runtime.zero.stage_1_and_2 as _s12
from deepspeed import comm as dist
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead

_PENDING = []
_LOGGED = {"done": False}


def _deferred_all_gather_dp_groups(groups_flat, partitioned_param_groups, dp_process_group,
                                   start_alignment_factor, allgather_bucket_size):
    numels = [g.numel() for g in groups_flat]
    sync_group = numels.index(max(numels))  # backbone: needed first by forward
    if not _LOGGED["done"]:
        print(f"[ds_defer_allgather_patch] group numels={numels}, sync group={sync_group}, "
              f"deferring the rest to action-head forward", flush=True)
        _LOGGED["done"] = True
    for group_id, (group_flat, partitioned_params) in enumerate(zip(groups_flat, partitioned_param_groups)):
        partition_id = dist.get_rank(group=dp_process_group[group_id])
        dp_world_size = dist.get_world_size(group=dp_process_group[group_id])
        if dp_world_size == 1:
            continue
        defer = group_id != sync_group
        handle = dist.all_gather_into_tensor(group_flat, partitioned_params[partition_id],
                                             group=dp_process_group[group_id], async_op=defer)
        if defer and handle is not None:
            _PENDING.append(handle)


_s12.all_gather_dp_groups = _deferred_all_gather_dp_groups

_orig_head_forward = LayerwiseFlowmatchingActionHead.forward


def _waiting_head_forward(self, *args, **kwargs):
    while _PENDING:
        _PENDING.pop().wait()
    return _orig_head_forward(self, *args, **kwargs)


LayerwiseFlowmatchingActionHead.forward = _waiting_head_forward
print("[ds_defer_allgather_patch] applied", flush=True)
