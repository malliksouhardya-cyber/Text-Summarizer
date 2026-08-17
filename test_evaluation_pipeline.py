import inspect

import evaluation_pipeline as ep


def test_default_mark_scale_is_out_of_100():
    assert inspect.signature(ep.evaluate_answer).parameters["max_marks"].default == 100.0
    assert inspect.signature(ep.run_pipeline).parameters["max_marks"].default == 100.0


def test_strong_answer_covering_core_modules_scores_above_85():
    student_text, model_text = ep.read_answers_from_file(ep.INPUT_FILE)
    result = ep.evaluate_answer(student_text, model_text)
    assert result["final_score"] > 85.0
