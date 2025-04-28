import pandas as pd
from datasets import load_dataset

from src.file_paths import p_belleza_lexicon, p_anew_lexicon

def get_bellezza_terms():
    bellezza = pd.read_csv(p_belleza_lexicon)  # 'Bellezza_Lexicon.csv'
    bellezza_terms = bellezza['word'] #.to_list()
    bellezza_valence = bellezza['combined_pleasantness'].to_list()
    bellezza_terms.index = bellezza_terms.values
    return bellezza_terms, bellezza_valence

def get_anew_terms():
    anew = pd.read_csv(p_anew_lexicon)  #'ANEW.csv'
    anew_terms = anew['Description'] #.to_list()
    anew_terms.index = anew_terms.values
    anew_valence = anew['Valence Mean'].to_list()
    return anew_terms, anew_valence

def get_sst_dataset():
    dataset = load_dataset("sst2", split="validation") # en-US"
    df = dataset.to_pandas()
    df = df.drop(columns=['idx'])
    #df = df.sample(n=10, random_state=1)
    lexicon_target = df['sentence']  #
    dataset_vals = df['label'].to_list()
    return lexicon_target, dataset_vals

def get_equally_distributed_sst_dataset():
    text, vals = get_sst_dataset()
    df = pd.DataFrame(zip(text, vals), columns=['sentence', 'label'], index=text.index)
    nr_i = df.label.value_counts().min()
    dfs = []
    for label in [0, 1]:
        dfi = df[df.label == label]
        dfi = dfi.sample(n=nr_i, random_state=1)
        dfs.append(dfi)
    df = pd.concat(dfs, axis=0)
    return df

def get_fin_phr_ds(do_equalize=False):
    dataset = load_dataset('financial_phrasebank', 'sentences_50agree', split='train') # https://github.com/samvardhan777/unsloth_Finanace_Sentimental_Analysis
    df = dataset.to_pandas()
    tot_nr = 500
    if do_equalize: # equalize
        nr_i = tot_nr // 3
        dfs = []
        for sentiment in ["positive", "neutral", "negative"]:
            dfi = df[df.sentiment == sentiment]
            dfi = dfi.sample(n=nr_i, random_state=1)
            dfs.append(dfi)
    df = df.sample(n=tot_nr, random_state=1)
    lexicon_target = df['sentence']  #
    dataset_vals = df['label'].to_list()
    return lexicon_target, dataset_vals
