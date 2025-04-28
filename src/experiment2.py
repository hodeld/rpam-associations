import gc
import os
from datetime import datetime
from os import path

import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from src.file_paths import OUTPUT_EXPERIMENTS, p_ws353_csv
from src.helper_functions import f1_balanced_set
from src.settings import DO_MULTIPLY
from src.templates import get_template_from_key, TEMPLATES_PAPER, get_template_dict, LEXICON_TEMPLATE_5TARGETS
from src.terms_lexica import get_bellezza_terms, get_anew_terms, get_sst_dataset, get_fin_phr_ds
from src.terms import pleasant as pleasant_multi, unpleasant as unpleasant_multi
from src.terms_biases import get_multi
from src.RPAM import ModelWrapperBias

unpleasant_single = ['vomit', 'kill', 'assault', 'abuse', 'grief', 'prison', 'accident', 'poison', 'ugly', 'tragedy',
                     'hatred', 'crash', 'divorce', 'jail', 'rotten', 'poverty', 'stink', 'cancer', 'murder', 'sickness',
                     'agony', 'disaster', 'death']
pleasant_single = ['honor', 'friend', 'diploma', 'honest', 'health', 'sunrise', 'family', 'miracle', 'cheer', 'freedom',
                   'rainbow', 'lucky', 'paradise', 'vacation', 'love', 'happy', 'gift', 'diamond', 'peace', 'laughter',
                   'loyal', 'heaven', 'gentle']
pleasant_single_t5 = ['honor', 'friend', 'diploma', 'honest', 'health', 'sunrise', 'family', 'miracle', 'cheer',
                      'freedom', 'rainbow', 'lucky', 'paradise', 'vacation', 'love', 'happy', 'gift', 'diamond',
                      'peace', 'laughter']  # multiple: ['caress']
unpleasant_single_t5 = ['divorce', 'prison', 'poverty', 'cancer', 'crash', 'grief', 'abuse', 'hatred', 'assault',
                        'ugly', 'poison', 'jail', 'accident', 'tragedy', 'stink', 'murder', 'sickness', 'kill',
                        'disaster', 'death']  # multiple: ['filth', 'pollute', 'rotten', 'vomit', 'agony']

LEXICA = ['Bellezza', 'sst2']  # 'ANEW'
RUN_NR = 0


def get_pleasant_singly(model_name):
    if model_name == 'gpt2':
        pleasant, unpleasant = pleasant_single, unpleasant_single
    else:
        pleasant, unpleasant = pleasant_single_t5, unpleasant_single_t5  # get_valence_terms(model_name)
    return pleasant, unpleasant


def get_pleasant_multiply(model_name):
    pleasant, unpleasant = pleasant_multi, unpleasant_multi
    return pleasant, unpleasant


def get_text_ground_truth(lexicon_name):
    lexicon_name = lexicon_name.lower()
    if lexicon_name == 'bellezza':
        return get_bellezza_terms()
    elif lexicon_name == 'anew':
        return get_anew_terms()
    elif lexicon_name == 'sst2':
        return get_sst_dataset()
    elif lexicon_name == 'fin_phrase':
        return get_fin_phr_ds()


def valence_by_models_templates(template_keys, model_name, lexicon_n, lexicon_template=None,
                                singly_tokenized=False, nr_targets=1):
    lexicon_target, ground_truth_val = get_text_ground_truth(lexicon_n)
    if singly_tokenized:
        pleasant, unpleasant = get_pleasant_singly(model_name)
    else:
        pleasant, unpleasant = get_pleasant_multiply(model_name)

    model_wrapper = ModelWrapperBias(model_name)

    dfs = []
    for t_key in template_keys:
        template = get_template_from_key(t_key)
        correlations = []
        col_name = f'{t_key}'
        cols = [n + '_' + col_name for n in ('valence', 'ground_truth', 'text')]
        valence_associations = []
        if nr_targets == 1:
            targets = lexicon_target
            index = lexicon_target.index
            for w in lexicon_target:
                if lexicon_template:
                    w = lexicon_template.format(w)
                terms = (pleasant, unpleasant, w)
                valence_association = model_wrapper.sc_weat_prob_effect_size(*terms, template=template)[0]
                valence_associations.append(valence_association)
        else:
            targets = []
            index = targets
            ground_truth_base = pd.Series(ground_truth_val)
            ground_truth_val = []
            lexicon_target = pd.Series(lexicon_target)
            len_lex = lexicon_target.size
            lexicon_target = lexicon_target.sample(frac=1, random_state=77)
            for i in range(0, len_lex // nr_targets * nr_targets, nr_targets):
                rows = lexicon_target.iloc[i:i + nr_targets]
                # w1, w2 = rows
                ground_truth_val.append(ground_truth_base.loc[rows.index].mean())
                w = lexicon_template.format(*rows)
                targets.append(w)
                terms = (pleasant, unpleasant, w)
                valence_association = model_wrapper.sc_weat_prob_effect_size(*terms, template=template)[0]
                valence_associations.append(valence_association)

        df = pd.DataFrame(zip(valence_associations, ground_truth_val, targets), columns=cols, index=index)
        dfs.append(df)
        valence_corr, pval = pearsonr(ground_truth_val, valence_associations)
        correlations.append(valence_corr)
        print(template, lexicon_n, valence_corr, pval)
    df = pd.concat(dfs, axis='columns')
    return df


def ws353_analysis(template, model_cls_name='gpt2'):
    def get_pat(id_x):
        try:
            # results = model_wrapper.get_token_probability_distribution(atts, targs, template)
            pat_association = model_wrapper.get_single_probability(atts, targs, template, id_x)
            # pat_association = results[0][id_x].item() # first target, xth attribute
            # del results
            gc.collect()
        except AssertionError:
            return False
        return pat_association

    fp = p_ws353_csv
    ws353 = pd.read_csv(fp, sep=',')
    word_1 = ws353['Word 1'].to_list()
    word_2 = ws353['Word 2'].to_list()
    human_orig = ws353['Human (mean)'].to_list()
    # Caution: multiple frequency of words!!
    model_wrapper = ModelWrapperBias(model_cls_name, model_cls_name=model_cls_name)
    current_tokenizer = model_wrapper.tokenizer
    word_1 = pd.Series(word_1)
    word_2 = pd.Series(word_2)
    human = pd.Series(human_orig)
    assert human.size == word_1.size
    df = pd.DataFrame(zip(human, word_1, word_2), columns=('human', 'word_1', 'word_2'))
    if DO_MULTIPLY is False:  # singly_tokenized:
        multi_terms_1 = get_multi(current_tokenizer, word_1)
        multi_terms_2 = get_multi(current_tokenizer, word_2)
        df = df[~word_1.isin(multi_terms_1)]  # ~is not
        df = df[~word_2.isin(multi_terms_2)]  # ~is not
    print('# word pairs:', len(df.index))
    for k, (ta, at) in enumerate((('word_1', 'word_2'), ('word_2', 'word_1'))):
        pats_w = []
        targets = df[ta]
        attributes = df[at]
        human = df.human
        assert human.size == attributes.size == targets.size
        attributes_unique = attributes.drop_duplicates()
        attributes_unique = attributes_unique.reset_index(drop=True)
        attribute_d = {v: k for k, v in attributes_unique.items()}
        assert max(attribute_d.values()) + 1 == len(attributes_unique)
        for i, (w1, w2) in enumerate(zip(targets, attributes)):
            targs = [w1]
            atts = attributes_unique
            id_at = attribute_d[w2]
            p = get_pat(id_at)
            if p is False:
                print(w1)
            pats_w.append(p)
            if i + 1 % 20 == 0:
                print(w1, k, i)
        df[f'assoc_{k}'] = pats_w
        print(f'round {k}')
    pats = df[['assoc_0', 'assoc_1']].mean(axis=1)
    humans = df.human
    ws, pval = spearmanr(pats, humans)  # 0.67 for both (diffrent template 0.43 for W1, W2: 0.52 for W2, W1)

    return ws, pval


def ws353_by_template(model_name, template_keys):
    main_fp = os.path.join(OUTPUT_EXPERIMENTS, f'{model_name}_ws353_template.csv')
    if os.path.isfile(main_fp):
        return
    res, t_keys = [], []

    def save(fpi=None):
        df = pd.DataFrame(res, columns=['rho', 'p'], index=t_keys)
        if fpi is None:
            fpi = os.path.join(OUTPUT_EXPERIMENTS, f'{model_name}_ws353_template.csv')
        df.to_csv(fpi)

    for t_key in template_keys:
        fp = os.path.join(OUTPUT_EXPERIMENTS, f'{model_name}_{t_key}_ws353_template.csv')
        if os.path.isfile(fp):
            continue
        print(model_name, t_key)
        template = get_template_from_key(t_key)
        ws, pval = ws353_analysis(template, model_cls_name=model_name)
        res.append((ws, pval))
        print(template, ws, pval)
        torch.cuda.empty_cache()
        del ws, pval
        t_keys.append(t_key)
        save(fp)
    save()


def file_path_valence(model_name, lexicon_name, nr_targets=1):
    return path.join(OUTPUT_EXPERIMENTS, f'valence_{lexicon_name}_targets{nr_targets}_{model_name}.csv')


def get_ground_truth_valence(model_name, lexicon_name, nr_targets=1, t_key=9):
    fp = file_path_valence(model_name, lexicon_name, nr_targets)
    df = pd.read_csv(fp, index_col=0)
    col_name = f'{t_key}'
    cols = [n + '_' + col_name for n in ('valence', 'ground_truth')]
    for c in cols:
        # convert to float
        df[c] = df[c].astype(float)
    valence_name, ground_truth_name = cols
    return df[ground_truth_name], df[valence_name]


def validate_pat_sw353(model_names):
    templates = TEMPLATES_PAPER
    for model_name in model_names:
        ws353_by_template(model_name, templates)
        torch.cuda.empty_cache()


def validate_pat_valnorm(model_names):
    template_keys = TEMPLATES_PAPER
    t_d = get_template_dict(template_keys)
    lexicon_template = LEXICON_TEMPLATE_5TARGETS  # , LEXICON_TEMPLATE
    singly_tokenized = False
    lexica = LEXICA

    def validate_model(model_name, lexicon, nr_targets=1):

        # w/out lexicon
        if nr_targets == 1:
            lexicon_template_i = None
        else:
            lexicon_template_i = lexicon_template

        df = valence_by_models_templates(template_keys, model_name, lexicon, nr_targets=nr_targets,
                                         singly_tokenized=singly_tokenized, lexicon_template=lexicon_template_i)

        # df['single_terms'] = single_terms
        df['nr_targets'] = nr_targets
        df['singly_tokenized'] = singly_tokenized
        return df

    nrs_targets_d = {'sst2': [1], 'Bellezza': [1, 5]}
    for lexicon in lexica:
        nrs_targets = nrs_targets_d[lexicon]
        for model_name in model_names:
            for nr_targets in nrs_targets:
                fp = file_path_valence(model_name, lexicon, nr_targets)
                if path.exists(fp):
                    continue
                print('validating', model_name, lexicon, nr_targets)
                with torch.no_grad():
                    df = validate_model(model_name, lexicon=lexicon, nr_targets=nr_targets)
                    df.to_csv(fp)
            torch.cuda.empty_cache()

    eval_intrinsic_valence(model_names, lexica, template_keys=template_keys)


def eval_intrinsic_valence(model_names, lexica=LEXICA, nrs_targets=[1, 5], template_keys=TEMPLATES_PAPER):
    # model in index
    ds = []
    mnames_str = '_'.join(model_names)
    for lexicon in lexica:
        for nr_targets in nrs_targets:
            if lexicon == 'sst2' and nr_targets > 1:
                continue
            for model_name in model_names:
                for t_key in template_keys:
                    ground_truth, valence = get_ground_truth_valence(model_name, lexicon, nr_targets, t_key)

                    if lexicon == 'sst2':
                        pears_valence_corr, pears_pval = None, None
                        spear_valence_corr, pears_pval = None, None
                        f1 = f1_balanced_set(valence, ground_truth)
                        print('f1_score: {:.3f}'.format(f1))
                    else:
                        f1 = None
                        pears_valence_corr, pears_pval = pearsonr(ground_truth, valence)
                        spear_valence_corr, pears_pval = spearmanr(ground_truth, valence)
                        print('pearson', lexicon, model_name, nr_targets, f'templ{t_key}', pears_valence_corr)

                    d = {'lexicon': lexicon, 'nr_targets': nr_targets, 'model_name': model_name,
                         'pears_corr': pears_valence_corr, 'spear_corr': spear_valence_corr,
                         'f1': f1, 'template': t_key, 'date': datetime.now().date()}
                    ds.append(d)
    # data frame from list of dicts
    df = pd.DataFrame(ds)
    fp = path.join(OUTPUT_EXPERIMENTS, f'valence_{"_".join(model_names)}.csv')
    df.to_csv(fp)


if __name__ == '__main__':
    model_names = ['mistral-7b-instruct', 'mistral-7b', 'gpt2', ]

    validate_pat_sw353(model_names)
    validate_pat_valnorm(model_names)  # 1 & 5 terms for Bellezza
    eval_intrinsic_valence(model_names)
