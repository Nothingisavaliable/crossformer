# CrossFormer pretrained policy in robosuite

This integration evaluates `hf://rail-berkeley/crossformer` without fine-tuning
on robosuite `Lift` with `Panda`. It uses language instruction
`pick up the cube`, the `single_arm` head, and Bridge action statistics.

Run the model server in the CrossFormer GPU environment:

```bash
conda activate crossformer-gpu
cd <path-to-crossformer>
python integrations/robosuite/crossformer_server.py
```

Run robosuite in another terminal:

```bash
conda activate robosuite
python -m pip install json-numpy
cd <path-to-crossformer>
python integrations/robosuite/lift_client.py --action-scale 0.2 --render
```

Start with `--action-scale 0.2` to inspect coordinate and rotation conventions.
Raise it to `1.0` only after the rollout moves in the expected directions.
The client defaults to 5 Hz to match BridgeData V2's 0.2-second action period;
using robosuite's common 20 Hz setting changes the reached-pose dynamics.
If a diagnostic rollout shows a consistent axis convention mismatch, it can be
tested explicitly, for example `--translation-signs 1 -1 1`. The default is
the identity mapping; do not treat a task-specific sign correction as a
replacement for calibration or fine-tuning.

The adapter reads the instantiated controller's `input_min`, `input_max`,
`output_min`, and `output_max`; it does not assume robosuite's default scaling.
Bridge reached-proprio Euler deltas are converted to the axis-angle rotation
vector expected by `OSC_POSE` before that scaling is inverted.
Bridge gripper state `1=open, 0=closed` is mapped to the Panda gripper's actual
action-space endpoints.
