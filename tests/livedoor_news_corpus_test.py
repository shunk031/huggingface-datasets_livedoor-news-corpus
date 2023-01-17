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
        (0.8, 0.1, 0.1),
        (0.6, 0.2, 0.2),
    ),
)
def test_train_valid_test_split(
    dataset_path: str,
    tng_ratio: float,
    val_ratio: float,
    tst_ratio: float,
):
    assert tng_ratio + val_ratio + tst_ratio == 1.0
    original_dataset = ds.load_dataset(path=dataset_path, split="train")
    original_dataset = original_dataset.shuffle(seed=42)

    # split train and validation + test
    tng_valtst_dataset = original_dataset.train_test_split(  # type: ignore
        train_size=tng_ratio,
    )
    # then, split validation + test to validation and test
    val_tst_dataset = tng_valtst_dataset["test"].train_test_split(
        test_size=tst_ratio / (val_ratio + tst_ratio)
    )

    dataset = ds.DatasetDict(
        {
            "train": tng_valtst_dataset["train"],
            "validation": val_tst_dataset["train"],
            "test": val_tst_dataset["test"],
        }
    )

    assert (
        len(original_dataset)  # type: ignore
        == dataset["train"].num_rows
        + dataset["validation"].num_rows
        + dataset["test"].num_rows
    )


@pytest.mark.parametrize(
    argnames="tng_ratio, val_ratio, tst_ratio,",
    argvalues=(
        (80, 10, 10),
        (60, 20, 20),
    ),
)
def test_train_valid_test_split_percent_slicing(
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
