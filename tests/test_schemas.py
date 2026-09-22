import pytest

from laya_multimodal.schemas import Option, Question, QuestionType


def test_choice_from_mapping_preserves_keys() -> None:
    question = Question.choice("animal", "What animal?", {"dog": "a dog", "cat": "a cat"})
    assert question.type is QuestionType.CHOICE
    assert [option.key for option in question.options] == ["dog", "cat"]


def test_score_from_dict_assigns_ordered_keys() -> None:
    question = Question.from_dict(
        {
            "id": "severity",
            "type": "score",
            "prompt": "How severe?",
            "options": ["none", "minor", "major"],
        }
    )
    assert [option.key for option in question.options] == ["0", "1", "2"]


def test_binary_requires_two_options() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        Question.from_dict(
            {
                "id": "broken",
                "type": "binary",
                "prompt": "Broken?",
                "options": ["yes", "no", "maybe"],
            }
        )


def test_duplicate_option_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        Question(
            id="duplicate",
            prompt="Choose",
            options=(Option("same", "one"), Option("same", "two")),
        )


def test_noul_is_accepted_as_binary_alias() -> None:
    question = Question.from_dict({"id": "present", "type": "noul", "prompt": "Is it present?"})
    assert question.type is QuestionType.BINARY
