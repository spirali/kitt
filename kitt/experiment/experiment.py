import hashlib
import json
import logging
import pprint
import random
import string
from pathlib import Path
from typing import Any, Dict, Optional

from ..environment import write_environment_yaml
from ..serialization import write_yaml
from ..files import GenericPath, ensure_directory

RUN_RESULT_PATH = "result.json"


class Run:
    """
    Represents a single training run with some parameters, generated artifacts and metrics.
    """

    def __init__(self, name: str, directory: Path, log_directory: Path):
        self.directory = ensure_directory(directory)
        self.data_directory = ensure_directory(self.directory / "data")
        self.name = name
        self.log_dir = Path(log_directory) / self.name
        self.parameters = {}
        self.metrics = {}
        self.artifacts = []

    def sub_run(self, name: str) -> "Run":
        """
        Creates a nested Run, useful e.g. for combining multiple runs together in cross-validation.
        """
        return Run(name, self.directory / name, self.log_directory())

    def __enter__(self):
        logging.info(f"Starting run {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        msg = f"Finished run {self.name}"
        if exc_type is None:
            msg += " successfully"
        else:
            msg += f" with error: {exc_val}"
        logging.info(msg)

        self.store_to_disk()

    def write_tb_summary(self, name: str, markdown_content: str):
        """
        Writes textual summary in to the TensorBoard log directory of this run.
        """
        import tensorflow as tf

        with tf.summary.create_file_writer(str(self.log_directory())).as_default():
            tf.summary.text(name, markdown_content, step=0)

    def write_parameters_to_tb(self):
        """
        Stores input parameters to TensorBoard.
        Call this before starting training.
        """
        params = [
            f"- {key}: {value}" for (key, value) in sorted(self.parameters.items())
        ]
        params_text = "\n".join(params)
        text_md = f"**Training run {self.name}**\n\n{params_text}"
        self.write_tb_summary("Training run", text_md)

    def store_to_disk(self):
        """
        Stores all saved information to disk.
        """
        write_environment_yaml(self.path("environment.yml"))
        with open(self.path("params.yml"), "w") as f:
            write_yaml(self.parameters, f)

        with open(self.path("result.json"), "w") as f:
            f.write(
                json.dumps(
                    {
                        "parameters": self.parameters,
                        "metrics": self.metrics,
                        "artifacts": self.artifacts,
                    },
                    indent=4,
                )
            )

    def path(self, path: GenericPath) -> Path:
        return self.directory / path

    def data_path(self, path: GenericPath) -> Path:
        return self.data_directory / path

    def model_directory(self) -> Path:
        return self.directory / "models"

    def log_directory(self) -> Path:
        return self.log_dir

    def print_parameters(self):
        params = sorted(self.parameters.items(), key=lambda item: item[0])
        print(f"Run {self.name} at {self.directory}")
        print("\n".join(f"{k}: {pprint.pformat(v)}" for (k, v) in params))

    def record_artifact(self, path: GenericPath):
        self.artifacts.append(str(path))

    def record_param(self, key: str, value):
        """
        Log an input parameter of a run (e.g. batch size or learning rate).
        """
        self.parameters[key] = value

    def record_params(self, parameters: Dict[str, Any]):
        self.parameters.update(parameters)

    def record_metric(self, name: str, metric):
        """
        Log a calculated value of a run (i.e. validation accuracy).
        """
        self.metrics[name] = metric

    def record_hyperparameters(
        self, parameters: Dict[str, Any], metrics: Dict[str, Any]
    ):
        import tensorflow as tf
        from tensorboard.plugins.hparams import api as hp

        with tf.summary.create_file_writer(str(self.log_directory())).as_default():
            hp.hparams_config(
                hparams=[hp.HParam(key) for key in parameters.keys()],
                metrics=[hp.Metric(key) for key in metrics.keys()],
            )
            hp.hparams(parameters)
            for (key, value) in metrics.items():
                tf.summary.scalar(key, value, step=1)


def generate_random_hash() -> str:
    name = "".join(random.choices(string.ascii_lowercase + string.digits, k=32))
    return hashlib.md5(name.encode()).hexdigest()[:16]


def generate_name_from_params(parameters: dict) -> str:
    items = sorted(parameters.items(), key=lambda item: item[0])

    def sanitize(value: str) -> str:
        return value.replace("_", "").replace("-", "").replace("/", "")

    def format_value(v) -> str:
        if isinstance(v, bool):
            return "y" if v else "n"
        if isinstance(v, (str, int, float)) or v is None:
            return sanitize(str(v))
        elif isinstance(v, (tuple, list)):
            v = tuple(v)
            return f"[{','.join(sorted(format_value(i) for i in v))}]"
        elif isinstance(v, dict):
            return "dict"
        else:
            return "NA"

    return "_".join(f"{sanitize(k)}={format_value(v)}" for (k, v) in items)


class ExperimentTracker:
    """
    Tracks parameters and results of experiments.
    """

    def __init__(self, name: str, root_dir: GenericPath = "training"):
        self.root_dir = ensure_directory(Path(root_dir) / name)
        self.log_dir = ensure_directory(Path(self.root_dir) / "logs")

    def new_run(self, name: Optional[str] = None) -> Run:
        if name is None:
            name = generate_random_hash()
        return Run(name, self.root_dir / name, self.log_dir)

    def directory(self) -> Path:
        return self.root_dir

    def log_directory(self) -> Path:
        return self.log_dir
