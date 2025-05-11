import re
from datetime import datetime
from os import path

import pandas as pd
import torch
from scipy.stats import spearmanr

from src.RPAM import ModelWrapperBias, inference_model
from src.file_paths import OUTPUT_EXPERIMENTS
from src.helper_functions import f1_balanced_set, f1_balanced_set_from_labels
from src.templates import TEMPLATES_PAPER
from src.experiment2 import valence_by_models_templates, get_ground_truth_valence, file_path_valence, \
    get_text_ground_truth
from src.terms_lexica import get_bellezza_terms

MAPPING_SST2 = {'positive': 1, 'none': -1, 'negative': 0}
MAPPING_fin_phrase = {'positive': 2, 'neutral': 1, 'none': 1, 'negative': 0}
COL_VALENCE = 'valence'


def get_word_ratings(output, words):
    def find_first_number(text, keyword):
        text = text.lower()
        # Regex pattern to find the first number between 1 and 5
        pattern = r'[1-5]'
        match = re.search(pattern, text)
        if match:
            str_nr = match.group(0)  # Returns the first captured group, which is the number
            # convert to integer if possible
            try:
                nr = int(str_nr)
                if 1 <= nr <= 5:
                    return nr
            except ValueError:
                pass
        return None

    ratings = []
    for word in words:
        rating = find_first_number(output, word)
        ratings.append(rating)
    return ratings


def rate_valence_downstream_mistralinstruct(word, model_wrapper, seed):
    # nr_words = len(words)
    prompt = f'''The purpose is to determine whether one has positive or negative 
    feelings about different words. Words can evoke various emotions. You are asked to rate one word 
    based on how pleasant or unpleasant they make you feel.

    Rate the word according to this 5-point scale:

    - If the word has a very pleasant meaning for you, rate it as 5.
    - If the word has a somewhat pleasant meaning, rate it as 4.
    - If the word has no pleasant or unpleasant meaning, rate it as 3.
    - If the word has a somewhat unpleasant meaning, rate it as 2.
    - If the word has a very unpleasant meaning, rate it as 1.

    Try to use all 5 points on the rating scale.  
    The rating of the word "{word}" is: '''

    prompt_olmo = f'''The purpose is to determine whether one has positive or negative 
    feelings about different words. Words can evoke various emotions. You are asked to rate one word 
    based on how pleasant or unpleasant they make you feel.

    Rate the word according to this 5-point scale:

    - If the word has a very pleasant meaning for you, rate it as 5.
    - If the word has a somewhat pleasant meaning, rate it as 4.
    - If the word has no pleasant or unpleasant meaning, rate it as 3.
    - If the word has a somewhat unpleasant meaning, rate it as 2.
    - If the word has a very unpleasant meaning, rate it as 1.

    Try to use all 5 points on the rating scale.  
    The rating of the word "happy" is: 5
    The rating of the word "death" is: 1 
    The rating of the word "{word}" is:'''
    output = inference_model(prompt, model_wrapper, max_new_tokens=50, seed=seed)
    ratings = get_word_ratings(output, [word])
    return ratings[0], output[:30]


def valence_downstream_mistral_instruct(model_wrapper, lexicon_target, lex_name=''):
    def chunk_list(input_list, chunk_size):
        """Yield successive chunks of size `chunk_size` from `input_list`."""
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]

    # Specifying the chunk size
    chunk_size = 100
    model_name = model_wrapper.model_name
    columns = [COL_VALENCE, 'output']
    fp = path.join(OUTPUT_EXPERIMENTS, f'valence_downstream_{lex_name}_{model_name}.csv')
    if path.exists(fp):
        df = pd.read_csv(fp, index_col=0)
    else:
        df = pd.DataFrame(None, columns=columns, index=lexicon_target)

    for i in range(10):
        df_ind = df.dropna(subset=[COL_VALENCE])
        lexicon_target_rest = pd.Index(lexicon_target).difference(df_ind.index)
        len_rest = len(lexicon_target_rest)
        print('words left', len_rest)
        if len_rest == 0:
            break

        # Iterating over the chunks
        for words in chunk_list(lexicon_target_rest, chunk_size):
            valence_associations, rating_texts_all = [], []
            for word in words:
                valence_ratings, rating_texts = rate_valence_downstream_mistralinstruct(word, model_wrapper, seed=1)
                valence_associations.extend(valence_ratings)
                rating_texts_all.extend(rating_texts)

            df_new = pd.DataFrame(zip(valence_associations, rating_texts_all), columns=columns, index=words)
            df_new = df_new.dropna()
            df.update(df_new)
            df.to_csv(fp)
    return df


def run_downstream_mistral_instruct(lexica=['Bellezza']):
    model_names = ['mistral-7b-instruct']
    bellezza_terms, bellezza_valence = get_bellezza_terms()
    target_words = [pd.Series(bellezza_terms)]

    for lex_name, lexicon_target in zip(lexica, target_words):
        dfs = []
        for model_name in model_names:
            df = valence_downstream(model_name, lexicon_target, lex_name)
            # rename column from COL_VALENCE to model_name + template
            df.rename(columns={COL_VALENCE: f'{COL_VALENCE}_{model_name}',
                               'output': f'output_{model_name}'}, inplace=True)
            dfs.append(df)
    eval_downstream_valence_mistral_instruct(lexica)


def eval_downstream_valence_mistral_instruct(lexica=['Bellezza']):
    model_names = ['mistral-7b-instruct']
    bellezza_terms, bellezza_valence = get_bellezza_terms()
    target_words = [bellezza_terms.to_list()]  # change to series
    ground_truths = [bellezza_valence]
    template_keys = TEMPLATES_PAPER
    nr_targets, t_key = 1, 9
    dfs = []
    # for each lexicon one df
    for lex_name, lexicon_target, ground_truth in zip(lexica, target_words, ground_truths):
        for model_name in model_names:
            fp = path.join(OUTPUT_EXPERIMENTS, f'valence_downstream_{lex_name}_{model_name}.csv')
            df = pd.read_csv(fp, index_col=0)
            df = df.loc[lexicon_target]
            df = df.dropna(subset=[COL_VALENCE])
            df = df[df[COL_VALENCE] != -1]
            valence_name = f'{COL_VALENCE}'  # _{model_name}
            for t_key in template_keys:
                ground_truth, internal_valence = get_ground_truth_valence(model_name, lex_name, nr_targets=nr_targets,
                                                                          t_key=t_key)
                for setting, ser in zip(['internal', 'humans'], [internal_valence, ground_truth]):
                    ground_truth_name = ser.name
                    # join internal valence and ground truth
                    df = df.join(ser)
                    # print('pearson', pearsonr(df[ground_truth_name], df[valence_name]))
                    valence_corr, pval = spearmanr(df[ground_truth_name], df[valence_name])
                    print(f'downstream corr. with {setting}', f'template_{t_key}', lex_name, model_name, valence_corr,
                          df.shape)
                    d = {'lexicon': lex_name, 'model_name': model_name,
                         'spearman_corr': valence_corr, 'pval': pval, 'template': t_key,
                         'setting': setting, 'date': datetime.now().date()}
                    dfs.append(d)
                # data frame from list of dicts
    df = pd.DataFrame(dfs)
    fp = path.join(OUTPUT_EXPERIMENTS, f'downstream_bellezza_aggregated_{"_".join(model_names)}.csv')
    df.to_csv(fp)


def rate_gpt2_sentiment(input_text, model_wrapper, seed):
    rating = model_wrapper.sentiment_finetuned(input_text, seed=seed)
    return rating, None


def map_func_sst2(x):
    return MAPPING_SST2.get(x, None)


def filter_fn(output):
    if "positive" in output:
        label = "positive"
    elif "negative" in output:
        label = "negative"
    elif "neutral" in output:
        label = "neutral"
    else:
        label = "none"
    return label


def rate_valence_downstream_mistral(input_text, model_wrapper, seed):
    prompt = generate_test_prompt(input_text)

    output = inference_model(prompt, model_wrapper, max_new_tokens=1, seed=seed)
    label = filter_fn(output)
    return map_func_sst2(label), output[:10]


def rate_valence_downstream(text, model_wrapper, seed=1):
    if model_wrapper.model_name == 'gpt2-sentiment':
        return rate_gpt2_sentiment(text, model_wrapper, seed)
    elif model_wrapper.model_name == 'mistral-7b':
        return rate_valence_downstream_mistral(text, model_wrapper, seed)
    elif model_wrapper.model_name == 'mistral-7b-instruct':
        return rate_valence_downstream_mistralinstruct(text, model_wrapper, seed)


def generate_test_prompt(input_text):
    # strip is crucial for the prompt to work!
    return f"""
            Analyze the sentiment of the text enclosed in square brackets,
            determine if it is positive, or negative, and return the answer as
            the corresponding sentiment label "positive" or "negative"

            [{input_text}] =

            """.strip()


def run_downstream_gpt2_sentiment():
    model_name = 'gpt2-sentiment'
    lexicon_name = 'sst2'
    template_keys = TEMPLATES_PAPER
    main(model_name, lexicon_name, template_keys)


def run_downstream_mistral_sentiment():
    model_name = 'mistral-7b'
    lexicon_name = 'sst2'
    template_keys = TEMPLATES_PAPER
    main(model_name, lexicon_name, template_keys)


def main(model_name='gpt2-sentiment', lexicon_name='sst2', template_keys=TEMPLATES_PAPER):
    fp = file_path_valence(model_name, lexicon_name)
    fp_ds = path.join(OUTPUT_EXPERIMENTS, f'valence_downstream_{lexicon_name}_{model_name}.csv')
    if path.exists(fp) is False:
        with torch.no_grad():
            df = valence_by_models_templates(template_keys, model_name, lexicon_name)
            df.to_csv(fp)
    lexicon_target, ground_truth = get_text_ground_truth(lexicon_name)

    valence_downstream(model_name, lexicon_target, lexicon_name)
    eval_downstream_valence([model_name], lexica=[lexicon_name])


def eval_downstream_valence(model_names, lexica=['sst2']):
    lexicon_name = lexica[0]
    lexicon_target, ground_truth = get_text_ground_truth(lexicon_name)
    template_keys = TEMPLATES_PAPER
    target_words = [lexicon_target]  # change to series
    ground_truths = [ground_truth]

    ds = []
    # for each lexicon one df
    for lex_name, lexicon_target, ground_truth in zip(lexica, target_words, ground_truths):
        for model_name in model_names:
            fp = path.join(OUTPUT_EXPERIMENTS, f'valence_downstream_{lex_name}_{model_name}.csv')
            df = pd.read_csv(fp, index_col=0)
            df = df.loc[lexicon_target.index]
            # df['label_ds'] = df['output'].apply(filter_fn)
            # df[COL_VALENCE] = df['label_ds'].apply(map_func_sst2)
            df = df.dropna(subset=[COL_VALENCE])
            df = df[df[COL_VALENCE] != -1]
            print('df shape', df.shape)
            valence_name = f'{COL_VALENCE}'  # _{model_name}
            for t_key in template_keys:
                ground_truth, internal_valence = get_ground_truth_valence(model_name, lex_name, nr_targets=1,
                                                                          t_key=t_key)
                for setting, ser in zip(['internal', 'humans'], [internal_valence, ground_truth]):
                    ground_truth_name = ser.name
                    # join internal valence and ground truth
                    df = df.join(ser)
                    valence_corr, pval = spearmanr(df[ground_truth_name], df[valence_name])
                    print(f'sentiment downstream corr. with {setting}, template{t_key}', lex_name, model_name,
                          valence_corr, df.shape)
                    if setting == 'internal':
                        f1 = f1_balanced_set(df[ground_truth_name], df[valence_name])
                    else:  # already converted to labels
                        f1 = f1_balanced_set_from_labels(df[ground_truth_name], df[valence_name])
                    print('F1_score: {:.3f}'.format(f1))

                    d = {'lexicon': lex_name, 'model_name': model_name,
                         'f1': f1, 'template': t_key, 'setting': setting, 'date': datetime.now().date()}
                    ds.append(d)
    # data frame from list of dicts
    df = pd.DataFrame(ds)
    fp = path.join(OUTPUT_EXPERIMENTS, f'downstream_sst_aggregated_{"_".join(model_names)}.csv')
    df.to_csv(fp)


def valence_downstream(model_name, lexicon_target: pd.Series, lex_name: str = ''):
    def chunk_series(input_series, chunk_size):
        """Yield successive chunks of size `chunk_size` from `input_list`."""
        for i in range(0, len(input_series), chunk_size):
            yield input_series.iloc[i:i + chunk_size]

    # Specifying the chunk size
    chunk_size = 100
    columns = ['text', COL_VALENCE, 'output']
    df = pd.DataFrame(None, columns=columns, index=lexicon_target.index)

    fp = path.join(OUTPUT_EXPERIMENTS, f'valence_downstream_{lex_name}_{model_name}.csv')
    if path.exists(fp):
        df_upd = pd.read_csv(fp, index_col=0)
        df.update(df_upd)
        df.to_csv(fp)
        print('df shape updated', df.shape)

    df_ind = df.dropna(subset=[COL_VALENCE])
    lexicon_target_rest_idx = lexicon_target.index.difference(df_ind.index)
    len_rest = len(lexicon_target_rest_idx)
    print('words left', len_rest)
    if len_rest == 0:
        return df
    if model_name == 'gpt2-sentiment':
        model_cls_name = 'gpt2-classifier'
    elif model_name == 'mistral-7b':
        model_cls_name = 'mistral-classifier'
    elif model_name == 'mistral-7b-instruct':
        model_cls_name = 'mistral'
    model_wrapper = ModelWrapperBias(model_name, model_cls_name=model_cls_name)  # same model classifier
    lexicon_target_rest = lexicon_target.loc[lexicon_target_rest_idx]
    # Iterating over the chunks
    for texts in chunk_series(lexicon_target_rest, chunk_size):
        valence_associations, rating_texts_all = [], []
        for text in texts.values:
            valence_rating, rating_text = rate_valence_downstream(text, model_wrapper, seed=1)
            valence_associations.append(valence_rating)
            rating_texts_all.append(rating_text)

        df_new = pd.DataFrame(zip(texts.values, valence_associations, rating_texts_all), columns=columns,
                              index=texts.index)
        df_new = df_new.dropna(subset=[COL_VALENCE])
        df.update(df_new)
        df.to_csv(fp)
    return df


if __name__ == "__main__":
    run_downstream_mistral_instruct()

    run_downstream_mistral_sentiment()
    run_downstream_gpt2_sentiment()
