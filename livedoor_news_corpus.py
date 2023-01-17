import logging
import pathlib
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


class LivedoorNewsCorpusDataset(ds.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            version=ds.Version("1.0.0"),
            description="livedoor ニュースコーパス",
        ),
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
        article_dir_list = [d for d in dataset_root_dir.iterdir() if d.is_dir()]
        assert len(article_dir_list) == 9

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs={"article_dir_list": article_dir_list},
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

    def _generate_examples(self, article_dir_list: List[pathlib.Path]):  # type: ignore[override]

        article_id = 0
        for article_dir in article_dir_list:
            article_category = article_dir.name
            article_paths = [
                f for f in article_dir.iterdir() if article_dir.name in f.name
            ]
            for article_path in article_paths:
                with open(article_path, "r") as rf:
                    article_data = [line.strip() for line in rf]

                example_dict = self.parse_article(article_data=article_data)
                example_dict["category"] = article_category

                yield article_id, example_dict
                article_id += 1
