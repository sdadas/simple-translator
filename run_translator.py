import argparse
import os
from typing import List, Dict, Tuple, Union
import torch.cuda
from dataclasses import dataclass
from flask import Flask, request, jsonify
from sqlitedict import SqliteDict
from transformers import MarianTokenizer, MarianMTModel
from nltk import download, sent_tokenize
from nltk.data import path as nltk_path

class TranslationModel:

    def __init__(self, src_lang_name: str, src_lang: str, tgt_lang: str, cache_dir: str="cache"):
        self.src_lang_name = src_lang_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.cache_dir = cache_dir
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        model_name = f"Helsinki-NLP/opus-mt-{self.src_lang}-{self.tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        model = MarianMTModel.from_pretrained(model_name, cache_dir=self.cache_dir)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        return model, tokenizer

    def translate(self, texts: List[str]):
        template = lambda text: f"{text}" if self.tgt_lang == "en" else f">>{self.tgt_lang}<< {text}"
        src_texts = [template(text) for text in texts]
        input_tokens = self.tokenizer(src_texts, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            input_tokens = input_tokens.to("cuda")
        output_tokens = self.model.generate(**input_tokens)
        return [self.tokenizer.decode(t, skip_special_tokens=True) for t in output_tokens]


class TranslationModelGroup:

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.models: Dict[Tuple[str, str], TranslationModel] = {}
        self.load("polish", "pl", "en")
        self.load("german", "de", "en")
        self.load("french", "fr", "en")
        self.load("spanish", "es", "en")

    def predict(self, batch: List[str], src_lang: str) -> List[str]:
        return self.translate(batch, src_lang=src_lang, tgt_lang="en")

    def load(self, src_lang_name: str, src_lang: str, tgt_lang: str):
        model = TranslationModel(src_lang_name, src_lang, tgt_lang, cache_dir=self.cache_dir)
        self.models[(src_lang, tgt_lang)] = model

    def translate(self, texts: List[str], src_lang: str, tgt_lang: str) -> List[str]:
        model = self.models[(src_lang, tgt_lang)]
        return model.translate(texts)


@dataclass
class InputSentence:
    sent_idx: int
    text_idx: int
    target: Union[int, str]


class InputProcessor:

    def __init__(self, cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = SqliteDict(os.path.join(cache_dir, "cache.sqlite"), autocommit=True)

    def split_sentences(self, texts: List[str], src_lang: str):
        langs = {"pl": "polish", "en": "english", "de": "german", "fr": "french", "es": "spanish"}
        mapping: Dict[int, InputSentence] = {}
        results: List[str] = []
        sent_idx = 0
        for text_idx, text in enumerate(texts):
            sentences = sent_tokenize(text, language=langs[src_lang])
            if len(sentences) == 0: sentences.append("_")
            for sent in sentences:
                target = self._get_from_cache(sent, src_lang)
                if target is None:
                    target = len(results)
                    results.append(sent)
                input = InputSentence(sent_idx, text_idx, target)
                mapping[sent_idx] = input
                sent_idx += 1
        return mapping, results

    def _get_from_cache(self, sentence: str, src_lang: str):
        key = f"{src_lang}:{sentence}"
        return self.cache.get(key, None)

    def cache_sentences(self, inputs: List[str], outputs: List[str], src_lang: str):
        for first, second in zip(inputs, outputs):
            key = f"{src_lang}:{first}"
            if self.cache.get(key, None) is None:
                self.cache[key] = second

    def join_sentences(self, mapping: Dict[int, InputSentence], sentences: List[str]):
        text_indices: Dict[int,List] = {val.text_idx:[] for val in mapping.values()}
        for sent_idx in sorted(mapping.keys()):
            input = mapping[sent_idx]
            text_idx = input.text_idx
            target = input.target
            if isinstance(target, int):
                target = sentences[target]
            text_indices[text_idx].append(target)
        texts = []
        for text_idx in sorted(text_indices.keys()):
            texts.append(" ".join(text_indices[text_idx]).strip())
        return texts

    def close(self):
        self.cache.close()


def create_app():
    app = Flask(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-cache", type=str, default="model_cache")
    parser.add_argument("--sentence-cache", type=str, default="sentence_cache")
    args = parser.parse_args()

    os.environ['TRANSFORMERS_CACHE'] = args.model_cache
    os.environ["NLTK_DATA"] = args.model_cache
    nltk_path.append(args.model_cache)
    download("punkt", download_dir=args.model_cache)
    proc = InputProcessor(args.sentence_cache)
    model = TranslationModelGroup(args.model_cache)

    @app.route("/translate/<src_lang>", methods=["POST"])
    def translate(src_lang):
        assert src_lang in ("pl", "de", "fr", "es", "en")
        data: any = request.data.decode("utf-8")
        response = _do_translate([data], src_lang)
        return response["outputs"][0]

    @app.route("/batch", methods=["POST"])
    def batch_translate():
        data: any = request.json
        inputs = data.get("inputs")
        src_lang = data.get("src_lang")
        inputs = [inputs] if isinstance(inputs, str) else inputs
        return jsonify(_do_translate(inputs, src_lang))

    def _do_translate(inputs, src_lang) -> Dict:
        if src_lang != "en":
            mapping, sentences = proc.split_sentences(inputs, src_lang)
            output_sentences = model.predict(sentences, src_lang) if len(sentences) > 0 else []
            proc.cache_sentences(sentences, output_sentences, src_lang)
            outputs = proc.join_sentences(mapping, output_sentences)
        else:
            outputs = inputs
        return {"inputs": inputs, "outputs": outputs, "src_lang": src_lang, "tgt_lang": "en"}

    return app, args, proc


if __name__ == '__main__':
    app, args, proc = create_app()
    try:
        app.run(host=args.host, port=args.port, threaded=False, debug=False)
    finally:
        proc.close()
