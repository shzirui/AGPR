# dart_rollouter

GUI rollout used by this repository. `run.sh` reads `runner.protocol` from `config/config.yaml` (`qwen3` or `uitars`) and starts the matching runner:

* `src/core/trajectory_runner_uitars.py`
* `src/core/trajectory_runner_qwen3.py`

Process rewards are scored during rollout when `process_reward.backend` is `original`. `seekjudge` is the baseline scorer. `disabled` records trajectories without process rewards.

```bash
cd dart_rollouter
bash model_service.sh
bash run.sh
```

Set checkpoint paths, service URLs, and the MySQL password in `config/config.yaml` before launching. See the repository README for the full training pipeline.
