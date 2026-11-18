from pathlib import Path

from task_benchmark.impl.deploy_manager import DHCoreDeployManager


def main() -> None:
    config_path = Path(__file__).resolve().with_name("test.yml")
    deployments = DHCoreDeployManager().deployTVM(config_path)
    print(deployments)


if __name__ == "__main__":
    main()
