import logging
import math
import pathlib
import random
from dataclasses import asdict, dataclass
from typing import List, Optional, Union

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
class Article(object):
    url: str
    date: str
    title: str
    content: str
    category: str


class LivedoorNewsCorpusConfig(ds.BuilderConfig):
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = False,
        random_state: int = 0,
        name: str = "default",
        version: Optional[Union[ds.utils.Version, str]] = ds.utils.Version("0.0.0"),
        data_dir: Optional[str] = None,
        data_files: Optional[ds.data_files.DataFilesDict] = None,
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=name,
            version=version,
            data_dir=data_dir,
            data_files=data_files,
            description=description,
        )
        assert train_ratio + val_ratio + test_ratio == 1.0

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.shuffle = shuffle
        self.random_state = random_state


class LivedoorNewsCorpusDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")  # type: ignore

    BUILDER_CONFIG_CLASS = LivedoorNewsCorpusConfig  # type: ignore

    BUILDER_CONFIGS = [
        LivedoorNewsCorpusConfig(
            version=VERSION,  # type: ignore
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

        archive = dl_manager.download(_DOWNLOAD_URL)
        tar_archive_iterator = dl_manager.iter_archive(archive)

        articles: List[Article] = []
        for tar_file_path, tar_file_obj in tar_archive_iterator:
            file_path = pathlib.Path(tar_file_path)
            article_category = file_path.parent.name

            if (
                file_path.match("README.txt")
                or file_path.match("CHANGES.txt")
                or file_path.match("LICENSE.txt")
            ):
                continue

            print(file_path)

            article_data = [line.decode().strip() for line in tar_file_obj.readlines()]
            articles.append(self.parse_article(article_data, article_category))

        if self.config.shuffle:  # type: ignore
            random.seed(self.config.random_state)  # type: ignore
            random.shuffle(articles)

        num_articles = len(articles)
        num_tng = math.ceil(num_articles * self.config.train_ratio)  # type: ignore
        num_val = math.ceil(num_articles * self.config.val_ratio)  # type: ignore
        num_tst = math.ceil(num_articles * self.config.test_ratio)  # type: ignore

        tng_articles = articles[:num_tng]
        val_articles = articles[num_tng : num_tng + num_val]
        tst_articles = articles[num_tng + num_val : num_tng + num_val + num_tst]

        assert len(tng_articles) + len(val_articles) + len(tst_articles) == num_articles

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={"articles": tng_articles},
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                gen_kwargs={"articles": val_articles},
            ),
            ds.SplitGenerator(
                name=ds.Split.TEST,  # type: ignore
                gen_kwargs={"articles": tst_articles},
            ),
        ]

    def parse_article(self, article_data: List[str], article_category: str) -> Article:
        return Article(
            url=article_data[0],
            date=article_data[1],
            title=article_data[2],
            category=article_category,
            content=" ".join(article_data[3:]),
        )

    def _generate_examples(self, articles: List[Article]):  # type: ignore[override]
        for i, article in enumerate(articles):
            yield i, asdict(article)
