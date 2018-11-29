from gensim.models import Word2Vec, KeyedVectors, FastText
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models.word2vec import LineSentence
import logging
from gensim import utils

# for logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

model = None


def tokenize_tr(content, token_min_len=2, token_max_len=50, lower=True):
    """
    Converts all characters to lower case in given content
    :param content:
    :param token_min_len:
    :param token_max_len:
    :param lower:
    :return:
    """
    if lower:
        lower_map = {ord(u'I'): u'ı', ord(u'İ'): u'i'}
        content = content.translate(lower_map)
    return [
        utils.to_unicode(token) for token in utils.tokenize(content, lower=True, errors='ignore')
        if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
    ]


def preprocess(corpus_path, output_path):
    """
    Preprocesses corpus for Word2Vec model
    :param corpus_path:
    :param output_path:
    :return:
    """
    i = 0
    output = open(output_path, 'w', encoding='utf-8')
    wiki = WikiCorpus(corpus_path, lemmatize=False, lower=True, dictionary={}, tokenizer_func=tokenize_tr)
    for text in wiki.get_texts():
        output.write(" ".join(text) + "\n")
        i = i + 1
        if i % 10000 == 0:
            print("Saved " + str(i) + " articles")

    output.close()
    print("Finished Saved " + str(i) + " articles")


def train_model(dataset, output_path, type='fasttext'):
    """
    Train a FastText or Word2Vec model
    :param dataset:
    :param output_path:
    :param type:
    :return:
    """
    if type == 'fasttext':
        model = FastText(LineSentence(dataset), alpha=0.025, size=400, window=5, min_count=5, sg=0, negative=20,
                         workers=multiprocessing.cpu_count())
        model.save(output_path)
    else:
        model = Word2Vec(LineSentence(dataset), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())  # parameter tuning?
        model.wv.save_word2vec_format(output_path, binary=True)

    return model


def populate_test_data(path):
    """
    Create combinations for analogies
    :param path:
    :return:
    """
    with open(path, 'r', encoding='utf-8') as file:
        data = file.readlines()

    data = [x.strip('\n') for x in data]
    tuples = []
    for line in data:
        tuples.append((line.split()[0], line.split()[1]))

    new_data = []
    for pair in tuples:
        for pair2 in tuples:
            if pair[0] != pair2[0]:
                new_data.append(" ".join([pair[0], pair[1], pair2[0], pair2[1]+"\n"]))

    with open('temp.txt', 'w', encoding='utf-8') as file:
        file.writelines(new_data)

    return new_data


def load_model(path, type='fasttext'):
    """
    Load a FastText or Word2Vec model
    :param path:
    :param type:
    :return:
    """
    if type == 'fasttext':
        return FastText.load_fasttext_format(path) if 'bin' in path else FastText.load(path)
    else:
        return KeyedVectors.load_word2vec_format(path, binary=True)


def evaluate_word_analogies(analogies, case_insensitive=False, topn=10):
    """
    Evaluate the accuracy of the model for given analogies
    :param analogies:
    :param case_insensitive:
    :param topn:
    :return:
    """
    logger.info("Evaluating word analogies on %s", analogies)
    sections, section = [], None
    for line_no, line in enumerate(utils.smart_open(analogies)):
        line = utils.to_unicode(line)
        if line.startswith(': '):
            # a new section starts => store the old section
            if section:
                sections.append(section)
                _log_evaluate_word_analogies(section)
            section = {'section': line.lstrip(': ').strip(), 'correct': [], 'incorrect': []}
        else:
            if not section:
                raise ValueError("Missing section header before line #%i in %s" % (line_no, analogies))
            try:
                if case_insensitive:
                    a, b, c, expected = [word.upper() for word in line.split()]
                else:
                    a, b, c, expected = [word for word in line.split()]
            except ValueError:
                logger.info("Skipping invalid line #%i in %s", line_no, analogies)
                continue

            predicted = None
            # find the most likely prediction using 3CosAdd (vector offset) method
            sims = model.wv.most_similar_cosmul(positive=[b, c], negative=[a], topn=topn)
            for element in sims:
                predicted = element[0].upper() if case_insensitive else element[0]
                if predicted == expected:
                    break
            if predicted == expected:
                section['correct'].append((a, b, c, expected))
            else:
                logger.info("%s: expected %s, predicted %s", line.strip(), expected, predicted)
                section['incorrect'].append((a, b, c, expected))
    if section:
        # store the last section, too
        sections.append(section)
        _log_evaluate_word_analogies(section)

    total = {
        'section': 'Total accuracy',
        'correct': sum((s['correct'] for s in sections), []),
        'incorrect': sum((s['incorrect'] for s in sections), []),
    }

    analogies_score = _log_evaluate_word_analogies(total)
    sections.append(total)
    # Return the overall score and the full lists of correct and incorrect analogies
    return analogies_score, sections


def _log_evaluate_word_analogies(section):
    correct, incorrect = len(section['correct']), len(section['incorrect'])
    if correct + incorrect > 0:
        score = correct / (correct + incorrect)
        logger.info("%s: %.1f%% (%i/%i)", section['section'], 100.0 * score, correct, correct + incorrect)
        return score


if __name__ == '__main__':
    # train model
    # model = train_model('dataset.txt', 'model-20ns-cbow', 'fasttext')
    
    # load model
    # model = load_model('model-20ns-cbow', 'fasttext')
    results_syntax = evaluate_word_analogies('test-tr-syntax.txt', topn=2)
    results_semantic = evaluate_word_analogies('test-tr-semantic.txt', topn=2)
