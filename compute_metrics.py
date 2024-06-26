#!/usr/bin/env python3
import argparse
import string
import fasttext
import urllib.request
import os
import functools
import itertools
import csv
import collections
from typing import Iterable

lid_url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
dict_path = 'words'  # downloaded from https://gist.githubusercontent.com/wchargin/8927565/raw/d9783627c731268fb2935a731a618aa8e95cf465/words
lid_path = 'lid.176.bin'
ja_tokenizer = None
zh_tokenizer = None

en_words = [line.strip() for line in open(dict_path)]
en_words = {word for word in en_words if word.islower() and len(word) > 3}

if not os.path.exists(lid_path):
    urllib.request.urlretrieve(lid_url, lid_path)
lid_model = fasttext.load_model(lid_path)


def normalize(text: str) -> str:
    text = text.split('\nQ:')[0].strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.replace("—", " ")
    text = text.replace("،", "")
    text = text.replace("#", "")
    return text


@functools.lru_cache(maxsize=2**20)
def tokenize(line: str, lang: str) -> list[str]:
    global ja_tokenizer, zh_tokenizer

    if lang == 'zh':
        if zh_tokenizer is None:
            import jieba
            zh_tokenizer = jieba
            zh_tokenizer.initialize()
        
        return list(zh_tokenizer.cut(line))
    
    elif lang == 'ja':
        if ja_tokenizer is None:
            from fugashi import Tagger
            try:
                ja_tokenizer = Tagger("-O wakati -b 50000")
            except RuntimeError:
                import unidic.download
                unidic.download.download_version()
                ja_tokenizer  = Tagger("-O wakati -b 50000")
        
        return ja_tokenizer.parse(line).split()

    else:
        return line.split()


def langid(line: str) -> str:
    (label,), score = lid_model.predict(line)
    return label.removeprefix('__label__') if score > 0.3 else 'unknown'


def compute_metrics(completions: Iterable[str], lang: str) -> dict[str, float]:
    """
    Compute Line Pass Rate (LPR) and Word Pass Rate (WPR) over the given completions, whose expected language is given.
    """
    with_word_errors = 0
    with_line_errors = 0
    non_skipped = 0
    line_acc = []

    for completion in completions:
        completion = normalize(completion)
        lines = completion.split("\n")
        line_tokens = [tokenize(line, lang) for line in lines]
        # remove lines that are too short
        indices = [i for i, tokens in enumerate(line_tokens) if len(tokens) >= 5]
        lines = [lines[i] for i in indices]
        line_tokens = [line_tokens[i] for i in indices]
        if lines:
            non_skipped += 1
            line_errors = sum(langid(line) != lang for line in lines)
            if line_errors > 0:
                with_line_errors += 1
            elif any(token.strip() in en_words for tokens in line_tokens for token in tokens):
                with_word_errors += 1
            line_acc.append(1 - line_errors / len(lines))

    metrics = {}
    metrics['acc'] = sum(line_acc) / len(line_acc) if line_acc else 1.0
    metrics['lpr'] = 1 - with_line_errors / max(1, non_skipped)
    if lang in ('ar', 'hi', 'ja', 'ko', 'ru', 'zh'):  # WPR is inaccurate for latin-script languages
        metrics['wpr'] = 1 - with_word_errors / max(1, non_skipped - with_line_errors)
    return metrics


def compute_all_metrics(outputs: list[dict]) -> dict[str, dict[str, float]]:
    """
    Takes the outputs from a model and returns all the WPR and LPR metrics (WPR and LPR per dataset and averages
    per language and per source).
    The provided outputs should be dictionaries with 'source', 'language' and 'completion' fields, for instance:
    
    ```
    outputs = [
        {'source': 'okapi', 'language': fr, 'completion': 'Je ne sais pas'},
        {'source': 'okapi', 'language': fr, 'completion': 'What do you mean?'},
    ]

    compute_all_metrics(outputs)
    {
        ('okapi', 'fr'): {'lpr': 0.5},               # scores for French Okapi
        ('okapi', 'all'): {'wpr': 1.0, 'lpr': 0.5},  # averages over the Okapi source
        ('all', 'fr'): {'lpr': 1.0},                 # averages over the French language
        ('all', 'all'): {'wpr': 1.0, 'lpr': 0.5},    # overall average
    }
    ```
    """
    all_metrics = {}
    metrics_per_lang = collections.defaultdict(list)
    metrics_per_source = collections.defaultdict(list)

    group_key = lambda output: (output['source'], output['language'])
    outputs = sorted(outputs, key=group_key)

    # for dataset, completions in outputs.items():
    for (source, lang), outputs in itertools.groupby(outputs, key=group_key):
        completions = [output['completion'] for output in outputs]
        metrics = compute_metrics(completions, lang)
        all_metrics[(source, lang)] = metrics
        metrics_per_lang[lang].append(metrics)
        metrics_per_source[source].append(metrics)

    def average_metrics(metrics: list[dict]):
        averages = {}
        for key in ('acc', 'lpr', 'wpr'):
            values = [
                metrics_[key] for metrics_ in metrics
                if metrics_.get(key) is not None  # WPR can be missing for some languages
            ]
            if values:
                averages[key] = sum(values) / len(values)
        return averages

    average_per_source = {
        (source, 'all'): average_metrics(metrics)
        for source, metrics in metrics_per_source.items()
    }
    all_metrics.update(average_per_source)
    
    average_per_lang = {
        ('all', lang): average_metrics(metrics)
        for lang, metrics in metrics_per_lang.items()
    }
    all_metrics.update(average_per_lang)

    average = average_metrics(list(average_per_lang.values()))
    all_metrics[('all', 'all')] = average

    return all_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute WPR and LPR over the model completions in given CSV file")
    parser.add_argument('csv_file', help="CSV file with the same format as the provided command-r outputs "
                        "(with 'source', 'language' and 'completion' fields)")
    args = parser.parse_args()

    with open(args.csv_file) as csv_file:
        reader = csv.DictReader(csv_file)
        outputs = list(reader)
    
    all_metrics = compute_all_metrics(outputs)

    print('source', 'language', 'lpr', 'wpr', sep='\t')
    for (source, lang), metrics in all_metrics.items():
        lpr = f"{metrics['lpr']:.2%}"
        wpr = f"{metrics['wpr']:.2%}" if 'wpr' in metrics else 'N/A'
        print(source, lang, lpr, wpr, sep='\t')
