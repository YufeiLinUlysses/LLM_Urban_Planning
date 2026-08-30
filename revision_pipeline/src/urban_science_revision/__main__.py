"""Run the Kedro project as a Python module."""

from kedro.framework.cli.utils import find_run_command


def main() -> None:
    find_run_command(__package__)(args=[])


if __name__ == "__main__":
    main()
