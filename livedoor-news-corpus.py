import logging
import math
import pathlib
import random
from dataclasses import dataclass
from typing import Dict, List

import datasets as ds

logger = logging.getLogger(__name__)

_CITATION = """\
https://www.rondhuit.com/download.html#ldcc
"""

_DESCRIPTION = """\
本コーパスは、NHN Japan株式会社が運営する「livedoor ニュース」のうち、下記のクリエイティブ・コモンズライセンスが適用されるニュース記事を収集し、可能な限りHTMLタグを取り除いて作成したものです。
"""

_HOMEPAGE = "https://www.rondhuit.com/download.html#ldcc"

_LICENSE = """\
各記事ファイルにはクリエイティブ・コモンズライセンス「表示 – 改変禁止」が適用されます。 クレジット表示についてはニュースカテゴリにより異なるため、ダウンロードしたファイルを展開したサブディレクトリにあるそれぞれの LICENSE.txt をご覧ください。 livedoor はNHN Japan株式会社の登録商標です。
"""


_DOWNLOAD_URL = "https://www.rondhuit.com/download/ldcc-20140209.tar.gz"


@dataclass
class LivedoorNewsCorpusConfig(ds.BuilderConfig):
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    shuffle: bool = False
    random_state: int = 0

    def __post_init__(self):
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0


class LivedoorNewsCorpusDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = LivedoorNewsCorpusConfig
    BUILDER_CONFIGS = [
        LivedoorNewsCorpusConfig(
            version=VERSION,
            description="Livedoor ニュースコーパス",
        )
    ]

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "url": ds.Value("string"),
                "date": ds.Value("string"),
                "title": ds.Value("string"),
                "content": ds.Value("string"),
                "category": ds.ClassLabel(
                    names=[
                        "movie-enter",
                        "it-life-hack",
                        "kaden-channel",
                        "topic-news",
                        "livedoor-homme",
                        "peachy",
                        "sports-watch",
                        "dokujo-tsushin",
                        "smax",
                    ]
                ),
            }
        )
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        dataset_root = dl_manager.download_and_extract(_DOWNLOAD_URL)
        dataset_root_dir = pathlib.Path(dataset_root) / "text"

        article_paths = list(dataset_root_dir.glob("*/**/*.txt"))
        article_paths = list(filter(lambda p: p.name != "LICENSE.txt", article_paths))

        config: LivedoorNewsCorpusConfig = self.config

        if config.shuffle:
            random.seed(config.random_state)
            random.shuffle(article_paths)

        num_articles = len(article_paths)
        num_tng = math.ceil(num_articles * config.train_ratio)
        num_val = math.ceil(num_articles * config.val_ratio)
        num_tst = math.ceil(num_articles * config.test_ratio)

        tng_articles = article_paths[:num_tng]
        val_articles = article_paths[num_tng : num_tng + num_val]
        tst_articles = article_paths[num_tng + num_val : num_tng + num_val + num_tst]

        assert len(tng_articles) + len(val_articles) + len(tst_articles) == num_articles

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={"article_paths": tng_articles},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                gen_kwargs={"article_paths": val_articles},
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                gen_kwargs={"article_paths": tst_articles},
            ),
        ]

    def parse_article(self, article_data: List[str]) -> Dict[str, str]:
        article_url = article_data[0]
        article_date = article_data[1]
        article_title = article_data[2]
        article_content = " ".join(article_data[3:])

        example_dict = {
            "url": article_url,
            "date": article_date,
            "title": article_title,
            "content": article_content,
        }
        return example_dict

    def _generate_examples(self, article_paths: List[pathlib.Path]):  # type: ignore[override]
        for i, article_path in enumerate(article_paths):
            article_category = article_path.parent.name
            with open(article_path, "r") as rf:
                article_data = [line.strip() for line in rf]

            example_dict = self.parse_article(article_data=article_data)
            example_dict["category"] = article_category
            yield i, example_dict
