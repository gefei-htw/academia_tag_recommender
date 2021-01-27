from joblib import dump, load
import os.path
import time
from pathlib import Path
from academia_tag_recommender.stopwords import stopwordlist
from academia_tag_recommender.definitions import MODELS_PATH

data_folder = Path(MODELS_PATH + '/document_representation')


def top_words(features, vectorizer):
    sum_words = features.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vectorizer.vocabulary_.items()]
    words_freq = list(sorted(words_freq, key=lambda x: x[1], reverse=True))
    return words_freq[:5]


def print_results(result):
    print('{:<15}{:<15}{:<15}{:<15}{:<10}{:<10}{:<10.5}{:<90}'.format(
        result['v'], result['p'], result['t'], result['stopwords'], result['n_grams'], str(result['shape']), result['time'], str(result['top_words'])))


def save_computation_result(result):
    results_path = data_folder / 'results.joblib'
    results = load(results_path)
    index = -1
    for idx, result_ in enumerate(results):
        if result_['v'] == result['v'] and result_['p'] == result['p'] and result_['t'] == result['t'] and result_['stopwords'] == result['stopwords'] and result_['n_grams'] == result['n_grams']:
            index = idx
    if index > -1:
        results[index] = result
    else:
        results.append(result)
    dump(results, results_path)


def fit_vectorizer(vectorizer, data, name):
    begin = time.time()
    features = vectorizer.fit_transform(data)
    time.sleep(1)
    end = time.time()
    vectorizer_path = data_folder / ('vectorizer/' + name + '.joblib')
    feature_path = data_folder / ('features/' + name + '.joblib')
    dump(vectorizer, vectorizer_path)
    dump(features, feature_path)
    time_passed = end - begin
    return [features, time_passed]


def handle_result(vectorizer, preprocessor, tokenizer, stop_word_option, n_gram_option, time, feature_model, vectorizer_model):
    result = {'v': vectorizer.__name__,
              'p': preprocessor.__name__ if preprocessor else 'none',
              't': tokenizer.__name__ if tokenizer else 'none',
              'stopwords': stop_word_option if stop_word_option else 'none',
              'n_grams': str(n_gram_option),
              'shape': feature_model.shape,
              'time': time,
              'top_words': top_words(feature_model, vectorizer_model)}
    print_results(result)
    save_computation_result(result)


def get_vect_feat_with_params(data, vectorizer, tokenizer, preprocessor, stop_word_option, n_gram_option, rerun=False):
    name = 'v={}&p={}&t={}&stopwords={}&ngrams={}'.format(vectorizer.__name__, preprocessor.__name__ if preprocessor else 'none',
                                                          tokenizer.__name__ if tokenizer else 'none', stop_word_option if stop_word_option else 'none', str(n_gram_option))
    vectorizer_path = data_folder / ('vectorizer/{}.joblib'.format(name))
    feature_path = data_folder / ('features/{}.joblib'.format(name))
    if not rerun and os.path.isfile(vectorizer_path) and os.path.isfile(feature_path):
        vectorizer_model = load(vectorizer_path)
        feature_model = load(feature_path)
        print('Received saved model')
    else:
        print('Fitting new model')
        if stop_word_option and preprocessor and tokenizer:
            stop_words_ = tokenizer()(preprocessor()(' '.join(stopwordlist)))
        else:
            stop_words_ = None
        vectorizer_model = vectorizer(
            min_df=2, tokenizer=tokenizer(), preprocessor=preprocessor(), stop_words=stop_words_, ngram_range=n_gram_option)
        feature_model, time = fit_vectorizer(vectorizer_model, data, name)
        handle_result(vectorizer, preprocessor, tokenizer, stop_word_option,
                      n_gram_option, time, feature_model, vectorizer_model)
        print('New model was fitted')
    return [vectorizer_model, feature_model]
