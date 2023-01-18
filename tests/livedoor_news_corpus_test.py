import datasets as ds
import pytest


@pytest.fixture
def dataset_path() -> str:
    return "livedoor_news_corpus.py"


def test_load_dataset(dataset_path: str):
    dataset = ds.load_dataset(path=dataset_path, random_state=42, shuffle=True)

    assert (
        dataset["train"].num_rows  # type: ignore
        + dataset["validation"].num_rows  # type: ignore
        + dataset["test"].num_rows  # type: ignore
        == 7367
    )

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
    dataset = ds.load_dataset(
        path=dataset_path,
        train_ratio=tng_ratio,
        val_ratio=val_ratio,
        test_ratio=tst_ratio,
    )

    assert (
        dataset["train"].num_rows  # type: ignore
        + dataset["validation"].num_rows  # type: ignore
        + dataset["test"].num_rows  # type: ignore
        == 7367
    )
