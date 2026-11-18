# Deploy Managers

## DHCoreDeployManager

`DHCoreDeployManager` creates a new run for each requested deployment profile,
using the source run's function, action, and run specification. The source run
is not modified. It waits for each new run's service endpoint and returns the
URLs in profile order:

```python
from task_benchmark.impl.deploy_manager import DHCoreDeployManager

manager = DHCoreDeployManager()
service_urls = manager.deploy(
    id_project="my-project",
    id_run="source-run-id",
    profiles=["small", "large"],
)
```

Install the project's optional DigitalHub dependencies with
`python -m pip install -e ".[digitalhub]"`. The SDK uses the locally configured
DigitalHub environment. The `id_project` value is the project name expected by
the SDK. The source run must reference a function and use a run kind containing
an action, such as `python+serve` or `container+serve`. The source run's
function reference is resolved at its recorded version. All source run
specification fields except its task, function reference, and original profile
are passed to each new run; the selected profile overrides the source profile.

If a created run does not expose a service URL, `deploy()` raises `RuntimeError`
instead of returning an incomplete result. DigitalHub API and SDK errors are
propagated to the caller.

### TVM: `deployTVM`

`deployTVM(yml_file)` reads a TVM deployment plan, loads its `project`, and
creates the configured `tvm` Function from the YAML `function` name and `model`
reference using ONNX format. It runs the build action once with `build-resources`,
compiles each
`targets` entry with its `target_architecture`, matching `tag`, and compile
resources, then serves each `nodes` entry with its profile, model reference,
served name, and run resources. It returns a list of
`(profile, target_architecture, service_url)` tuples in the order of `nodes`.

The compile-resource key may be spelled `compile-resources` or
`complile-resources` (the spelling used in `yml/dh_tvm.yml`). The compile run's
`so_model` reference is used when provided; otherwise the Function is refreshed
and its updated `so_model` is used. Each node's `target_architecture` must have a
matching entry under `targets`.

Install the SDK and TVM runtime plugin with
`python -m pip install -e ".[digitalhub]"`, then:

```python
from task_benchmark.impl.deploy_manager import DHCoreDeployManager

deployments = DHCoreDeployManager().deployTVM("yml/dh_tvm.yml")
```

The method waits for build and compile completion. It starts each service run
without blocking, then polls its refreshed state until it reaches `RUNNING` or
`ERROR`. `timeout_seconds` controls the per-service timeout and defaults to 30
seconds; expiration raises `TimeoutError`. `ERROR` raises `RuntimeError`.
Failures from the SDK/Core are propagated; missing compiled models or service
URLs raise explicit errors.
