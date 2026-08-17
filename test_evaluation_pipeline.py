import inspect

import evaluation_pipeline as ep


def test_default_mark_scale_is_out_of_100():
    assert inspect.signature(ep.evaluate_answer).parameters["max_marks"].default == 100.0
    assert inspect.signature(ep.run_pipeline).parameters["max_marks"].default == 100.0
