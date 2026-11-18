from pathlib import Path

from task_benchmark.impl.deploy_manager import DHCoreDeployManager


def main() -> None:
    config_path = Path(__file__).resolve().with_name("dev.yml")
    try:
        deployments = DHCoreDeployManager().deployTVM(config_path)
        print(deployments)
    except Exception as e:
        print(f"{e}")


if __name__ == "__main__":
    main()
