"""CatalanQA Dataset."""
# Loading script for the CatalanQA dataset.
import json

import os

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
None 
"""

_DESCRIPTION = """\
CatalanQA: an extractive QA dataset from original Catalan Sources: Wikipedia and VilaWeb newswire.

 It is an aggregation and balancing of 2 previous datasets: VilaQUAD and ViquiQUAD, which were described in 

This dataset can be used to build extractive-QA and Language Models.

Splts have been balanced by kind of question, and unlike other datasets like SQUAD, it only contains, per record, one question and one answer for each context, although the contexts can repeat multiple times.

- test.json 	contains 2135 question/answer pairs

- train.json 	contains	 17135 question/answer pairs

- dev.json contains 2157 question/answer pairs

Funded by the Generalitat de Catalunya, Departament de Polítiques Digitals i Administració Pública (AINA),
 and Plan de Impulso de las Tecnologías del Lenguaje (Plan TL).
"""

_HOMEPAGE = ""

#_URL = "https://huggingface.co/datasets/projecte-aina/catalanqa/resolve/main/"
_TRAINING_FILE = "train_en.json"
_DEV_FILE = "dev-v2.0.json"
_TEST_FILE = "dev-v2.0.json"


class CatalanQA(datasets.GeneratorBasedBuilder):
    """CatalanQA Dataset."""

    VERSION = datasets.Version("1.0.1")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": [
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ],
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = "/content/drive/My Drive/APP1/"

        data_files = {
            "train": os.path.join(data_dir, _TRAINING_FILE),
            "dev": os.path.join(data_dir, _DEV_FILE),
            "test": os.path.join(data_dir, _TEST_FILE),
        }

        #urls_to_download = {
            #"train": f"{_URL}{_TRAINING_FILE}",
            #"dev": f"{_URL}{_DEV_FILE}",
            #"test": f"{_URL}{_TEST_FILE}",
        #}
        #downloaded_files = dl_manager.download(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": data_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": data_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": data_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            catalanqa = json.load(f)
            for article in catalanqa["data"]:
                title = article.get("title", "").strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]
                        # answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        # answers = [answer["text"].strip() for answer in qa["answers"]]
                        text = qa["answers"][0]["text"]
                        answer_start = qa["answers"][0]["answer_start"]

                        # Features currently used are "context", "question", and "answers".
                        # Others are extracted here for the ease of future expansions.
                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": [{"text": text, "answer_start": answer_start}],
                        }

