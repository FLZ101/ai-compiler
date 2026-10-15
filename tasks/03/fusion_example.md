# Task 03 fusion example

This example uses a reduction followed by a pointwise multiply so that
`ir_post_fusion.txt` shows two scheduler nodes fused into one
`FusedSchedulerNode`.

## Files

- `tasks/03/fusion_example.py`
- `tasks/03/fusion_example.md`
- `tasks/03/fused_scheduler_node.md`

## Run

```bash
TORCH_COMPILE_DEBUG=1 TORCH_LOGS="output_code" \
  make run SCRIPT=tasks/03/fusion_example.py
```

Artifacts are written to `tasks/03/fusion_debug/`.

## Before fusion

`ir_pre_fusion.txt` has two separate `SchedulerNode`s:

```text
op0: SchedulerNode(ComputedBuffer)   # add then sum
op1: SchedulerNode(ComputedBuffer)   # multiply by 2
```

`op0` computes `buf0 = sum(x + 1)`. `op1` loads `buf0` and computes `buf0 * 2`.

## After fusion

`ir_post_fusion.txt` wraps both nodes in one fused scheduler node:

```text
op0_op1: FusedSchedulerNode(SchedulerNode,SchedulerNode)
```

The fused node still contains both underlying nodes as `snodes[0]` and
`snodes[1]`, but the scheduler can now treat them as one unit for memory and
launch planning.

## Check

- `ir_pre_fusion.txt` shows two separate `op0` and `op1` nodes.
- `ir_post_fusion.txt` shows `op0_op1: FusedSchedulerNode(...)`.
- `fused_scheduler_node.md` explains the fusion and why `x + 1` is folded into
  the reduction stage rather than becoming its own scheduler node.
