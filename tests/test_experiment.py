import json
from os.path import dirname

from kitt.experiment.experiment import ExperimentTracker, generate_name_from_params


def test_generate_name_from_params_simple():
    assert generate_name_from_params({"foo": 1}) == "foo=1"


def test_generate_name_from_params_sort_keys():
    assert generate_name_from_params({"a": 1, "b": 2}) == "a=1_b=2"


def test_generate_name_from_params_list():
    assert generate_name_from_params({"a": [1, 2, "x"]}) == "a=[1,2,x]"


def test_generate_name_from_params_bool():
    assert generate_name_from_params({"a": True}) == "a=y"
    assert generate_name_from_params({"a": False}) == "a=n"


def test_generate_name_from_params_dict():
    assert generate_name_from_params({"a": {"b": 5}}) == "a=dict"


def test_generate_name_from_params_sanitize_key():
    assert generate_name_from_params({"a_b-c/x": 5}) == "abcx=5"


def test_generate_name_from_params_sanitize_value():
    assert generate_name_from_params({"a": "a_b-c/x"}) == "a=abcx"


def test_generate_name_from_params_unknown_value():
    assert generate_name_from_params({"a": Exception()}) == "a=NA"


def test_run_dump_to_disk():
    tracker = ExperimentTracker("foo")
    with tracker.new_run("bar") as run:
        run.record_artifact("artifact1")
        run.record_param("a", 5)
        run.record_params({"b": 3, "c": 4})
        run.record_metric("accuracy", 0.8)

    with open(run.path("result.json")) as f:
        data = json.loads(f.read())
        assert data == {
            "artifacts": ["artifact1"],
            "parameters": {"a": 5, "b": 3, "c": 4},
            "metrics": {"accuracy": 0.8},
        }


def test_run_log_directory_has_same_parent():
    tracker = ExperimentTracker("foo")
    r1 = tracker.new_run()
    r2 = tracker.new_run()
    assert r1.log_directory() != r2.log_directory()
    assert dirname(r1.log_directory()) == dirname(r2.log_directory())


def test_run_generate_name():
    tracker = ExperimentTracker("foo")
    r1 = tracker.new_run()
    r2 = tracker.new_run()
    assert r1.directory != r2.directory
    assert r1.name != r2.name
