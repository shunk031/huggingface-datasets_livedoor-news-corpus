import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "livedoor_news_corpus.py"


def test_load_dataset(dataset_path: str):
    dataset = ds.load_dataset(path=dataset_path)

    assert dataset["train"].num_rows == 7367  # type: ignore
    assert len(set(dataset["train"]["category"])) == 9  # type: ignore


@pytest.mark.parametrize(
    argnames="tng_ratio, val_ratio, tst_ratio,",
    argvalues=(
        (80, 10, 10),
        (60, 20, 20),
    ),
)
def test_train_valid_test_split(
    dataset_path: str,
    tng_ratio: float,
    val_ratio: float,
    tst_ratio: float,
    num_overall_dataset: int = 7367,
):
    assert tng_ratio + val_ratio + tst_ratio == 100

    tng_dataset = ds.load_dataset(
        path=dataset_path,
        split=f"train[:{tng_ratio}%]",
    )
    val_dataset = ds.load_dataset(
        path=dataset_path,
        split=f"train[{tng_ratio}%:{tng_ratio + val_ratio}%]",
    )
    tst_dataset = ds.load_dataset(
        path=dataset_path,
        split=f"train[{tng_ratio + val_ratio}%:{tng_ratio + val_ratio + tst_ratio}%]",
    )

    dataset = ds.DatasetDict(
        {"train": tng_dataset, "validation": val_dataset, "test": tst_dataset}
    )

    assert (
        num_overall_dataset
        == dataset["train"].num_rows
        + dataset["validation"].num_rows
        + dataset["test"].num_rows
    )
