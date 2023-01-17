import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "livedoor_news_corpus.py"


def test_load_dataset(dataset_path: str):
    dataset = ds.load_dataset(path=dataset_path)

    assert dataset["train"].num_rows == 7367  # type: ignore
    assert len(set(dataset["train"]["category"])) == 9  # type: ignore
