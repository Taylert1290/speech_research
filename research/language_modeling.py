from nltk import trigrams
from collections import defaultdict
import pandas as pd
import re


class WordProbabilities(object):

    """
    this class creates trigram co-occurences against the input corpus.
    we're looking for probabilities of 2 input words occuring with the target word
    """

    def __init__(self):
        self.model = defaultdict(lambda: defaultdict(lambda: 0))

    def preprocess(self, text):

        """
        clean up input text

        :param text: str
            messy input text
        :return:
            str
            cleaned output text
        """
        text = text.lower()
        text = re.sub("#", "", text)
        text = re.sub("&amp;", "", text)
        text = re.sub(r"\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*", "", text)
        text = " ".join(text.split())
        return text

    def frequency_counter(self, corpus):
        """
        Count the occurrences of each trigram combination

        :param corpus: list(str)
        """
        for doc in corpus:
            for w1, w2, w3 in trigrams(
                self.preprocess(doc).split(), pad_right=True, pad_left=True
            ):
                self.model[(w1, w2)][w3] += 1

    def probabilities(self):
        """
        Converts the counts into probabilities
        :return:
        """
        for w1_w2 in self.model:
            total_count = float(sum(self.model[w1_w2].values()))
            for w3 in self.model[w1_w2]:
                self.model[w1_w2][w3] /= total_count

    def fit_transform(self, corpus):
        """
        converts the input text into a probabilities dictionary
        :param corpus: list[str]
        :return:
        dict[str,str][[dict[str,float]]
        """
        self.frequency_counter(corpus=corpus)
        self.probabilities()
        return self.model


class TextGeneration(object):
    def __init__(self, generation_model):
        self.generation_model = generation_model

    def generate(self, inputword_1, inputword_2):
        sentence_finished = False
        text = [inputword_1, inputword_2]
        while not sentence_finished:
            accumulator = 0
            token_dict = self.generation_model[tuple(text[-2:])]
            sorted_tuple = sorted(token_dict.items(), key=lambda kv: kv[1])
            keys = [x[0] for x in sorted_tuple]
            if len(keys) > 0:
                word = list(self.generation_model[tuple(text[-2:])].keys())[0]
                accumulator += self.generation_model[tuple(text[-2:])][word]
                text.append(word)
            else:
                text.append(None)
            if text[-2:] == [None, None]:
                sentence_finished = True
        generated_text = " ".join([t for t in text if t])
        return " ".join(generated_text.split())


if __name__ == "__main__":
    df = pd.read_csv("../text_data/Corona_NLP_test.csv")
    text = df["OriginalTweet"]
    text_model = WordProbabilities().fit_transform(corpus=text)
    generator_model = TextGeneration(generation_model=text_model)
    examples = [("today", "the"), ("the", "virus"), ("we", "are"), ("this", "is"),('the','people')]
    with open("generated_text.txt", "w") as f:
        for example in examples:

            generated_text = generator_model.generate(
                inputword_1=example[0], inputword_2=example[1]
            )
            f.write(generated_text)
            f.write("\n")
