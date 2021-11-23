import argparse
from collections import defaultdict
from typing import List, Dict, Tuple

import torch.cuda
from flask import Flask, request, jsonify
from transformers import MarianTokenizer, MarianMTModel
from nltk import download, sent_tokenize


class TranslationModel:

    def __init__(self, src_lang_name: str, src_lang: str, tgt_lang: str):
        self.src_lang_name = src_lang_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        model_name = f"Helsinki-NLP/opus-mt-{self.src_lang}-{self.tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir="cache")
        model = MarianMTModel.from_pretrained(model_name, cache_dir="cache")
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        return model, tokenizer

    def translate(self, texts: List[str]):
        mapping, sentences = self._split_sentences(texts)
        output_sentences = self._translate(sentences)
        return self._join_sentences(mapping, output_sentences)

    def _split_sentences(self, texts: List[str]):
        mapping: Dict[int, int] = {}
        results: List[str] = []
        sent_idx = 0
        for text_idx, text in enumerate(texts):
            sentences = sent_tokenize(text, language=self.src_lang_name)
            if len(sentences) == 0: sentences.append("_")
            for sent in sentences:
                mapping[sent_idx] = text_idx
                sent_idx += 1
                results.append(sent)
        return mapping, results

    def _join_sentences(self, mapping: Dict[int, int], sentences: List[str]):
        results = defaultdict(list)
        for idx, sent in enumerate(sentences):
            text_idx = mapping.get(idx)
            results[text_idx].append(sent)
        texts = []
        for text_idx in sorted(results.keys()):
            texts.append(" ".join(results[text_idx]).strip())
        return texts

    def _translate(self, texts: List[str]):
        template = lambda text: f"{text}" if self.tgt_lang == "en" else f">>{self.tgt_lang}<< {text}"
        src_texts = [template(text) for text in texts]
        input_tokens = self.tokenizer(src_texts, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            input_tokens = input_tokens.to("cuda")
        output_tokens = self.model.generate(**input_tokens)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in output_tokens]


class TranslationModelGroup:

    def __init__(self):
        self.models: Dict[Tuple[str, str], TranslationModel] = {}

    def load(self, src_lang_name: str, src_lang: str, tgt_lang: str):
        model = TranslationModel(src_lang_name, src_lang, tgt_lang)
        self.models[(src_lang, tgt_lang)] = model

    def translate(self, texts: List[str], src_lang: str, tgt_lang: str):
        model = self.models[(src_lang, tgt_lang)]
        return model.translate(texts)


def create_app():
    app = Flask(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    model = TranslationModelGroup()
    model.load("polish", "pl", "en")
    model.load("german", "de", "en")
    model.load("french", "fr", "en")
    model.load("spanish", "es", "en")

    @app.route("/translate", methods=["POST"])
    def translate():
        data: any = request.json
        inputs = data.get("inputs")
        src_lang = data.get("src_lang")
        tgt_lang = data.get("tgt_lang")
        inputs = [inputs] if isinstance(inputs, str) else inputs
        outputs = model.translate(inputs, src_lang, tgt_lang) if src_lang != tgt_lang else inputs
        return jsonify({"inputs": inputs, "output": outputs, "src_lang": src_lang, "tgt_lang": tgt_lang})

    return app, args


if __name__ == '__main__':
    download("punkt")
    app, args = create_app()
    app.run(host=args.host, port=args.port, threaded=False)
