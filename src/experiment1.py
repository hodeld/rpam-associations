from os import path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.file_paths import OUTPUT_EXPERIMENTS
from src.templates import get_template_from_key
from src.terms_biases import get_bias_dic, get_bias_dic_unmodified
from src.RPAM import ModelWrapperBias, RUN_NR


MODEL_NAME_D = {
    'mistral-7b-instruct': 'Mistral-7B-Instruct',
    'mistral-7b': 'Mistral-7B',
    'gpt2': 'GPT-2',
    'gpt2-large': 'GPT-2-Large',
    'gpt-neo': 'GPT-Neo',
    't5-small': 'T5-small',
}

RUN_NR = 'exp1'


def wrap_bias_labels(labels):
    labels = [l.replace('Temporary/Permanent', '\n     Temporary/Permanent') for l in labels]
    labels = [l.replace('Weapons, P/U', 'Weapons,\n     P/U') for l in labels]
    labels = [l.replace('Female names, Career/Family', 'Female names,\n     Career/Family') for l in labels]
    labels = [l.replace('Male/Female terms', '\n     Male/Female terms') for l in labels]
    labels = [l.replace('Old names, P/U', 'Old names,\n     P/U') for l in labels]
    return labels


def get_ceat_d():
    bias_d, _ = get_bias_dic_unmodified()
    new_d = {k: v[1] for k, v in bias_d.items()}
    return new_d


def get_colors_hatches(n=20):
    for_bidden_c = [4, 5]
    # colors = list([plt.cm.tab20(i) for i in range(n) if i not in for_bidden_c])
    colors = list([plt.cm.tab20c(i) for i in [0, 4, 12, 16, 20]])
    # colors = list([plt.cm.tab10(i) for i in range(10) if i not in  [2]])
    # colors = ['#0868ac', '#43a2ca', '#7bccc4', '#bae4bc', '#f0f9e8'] # from https://colorbrewer2.org/
    hatches = ['..', '+', '\\', 'x', '--', ]
    return colors, hatches


def run_weat_model(model_names, bias_d, t_key=9, permutations=None):
    def save(mname=''):
        df['tkey'] = t_key
        df['run'] = RUN_NR
        fname = f'bias_models_{RUN_NR}{mname}.csv'
        fp = path.join(OUTPUT_EXPERIMENTS, fname)
        print(fname)
        df.to_csv(fp)

    # for 3 models
    template = get_template_from_key(t_key)
    dfs = []
    for model_name in model_names:
        df = multiple_weat_template(model_name, bias_d, [template], permutations=permutations)
        df.index = [model_name]
        save(mname=model_name)
        dfs.append(df)
    df = pd.concat(dfs, axis=1)  # tdo
    mn = 'all'  # + '_'.join(model_names)
    save(mn)


def plot_models_biases(df):
    plt.rcParams.update({'font.size': 15})
    fig, subplots = plt.subplots(1, 1, figsize=(9, 10))
    nr_models = len(df.index)
    ax = subplots
    colors, hatches = get_colors_hatches(nr_models)
    bias_d = get_ceat_d()
    labels = bias_d.values()
    # linebreaks
    labels = wrap_bias_labels(labels)
    if nr_models == 3:
        shift = 0.3
    elif nr_models == 4:
        shift = 0.21
    k = 0
    for model, model_name in MODEL_NAME_D.items():
        if model not in df.index:
            continue
        biases = df.loc[model][bias_d.keys()]
        set_ax(ax, k, labels, model_name, biases, colors, hatches, do_minor=True, shift=shift)
        ax.set_xlabel("RPAM Test Effect Size [Cohen's d]")
        k += 1
    ax.invert_yaxis()  # labels read top-to-bottom -> only once
    ax.grid('off', axis='y', which='minor')

    plt.legend(loc='upper center', bbox_to_anchor=(.5, 1.08),  # (x, y, width, of ref box
               fontsize='large', ncol=3)
    fig.tight_layout()

    plt.show()


def ceat_bias_pre():
    plt.rcParams.update({'font.size': 16})
    fig, subplots = plt.subplots(1, 1, figsize=(8, 12))

    ax = subplots
    colors, hatches = get_colors_hatches(3)
    bias_d = get_ceat_d()
    return fig, ax, colors, hatches, bias_d


def set_ax(ax, k, labels, label_name, biases, colors, hatches, shift=.3, x_lim=1.8, align='center',
           do_minor=False, k_label=None):
    if k_label is None:
        k_label = k
    y_pos = np.arange(len(labels))
    y_ticks_minor = y_pos - shift / 2
    # edge
    ax.barh(y_pos + k * shift, biases, shift, label=label_name, align=align,
            fill=False, edgecolor=colors[k_label], hatch=hatches[k_label],
            linewidth=1.8)
    ax.set_yticks(y_pos + shift, labels=labels, ha='left')
    if do_minor:
        ax.set_yticks(y_ticks_minor, minor=True)  # , ha='left')
    ax.tick_params(axis='y', direction='in', pad=-20)

    ax.set_xlim(left=-x_lim, right=x_lim)
    ax.axvline(x=0, color='black', )


def run_plot_models_biases(fname='bias_models_all.csv', model_names=None):
    """plot 3 models vs. biases"""
    fp = path.join(OUTPUT_EXPERIMENTS, fname)
    df = pd.read_csv(fp, index_col=0)
    if model_names:
        df = df.loc[model_names]
    plot_models_biases(df)


def multiple_weat_template(model_name, bias_d, templates, fname=None, permutations=None):
    model_wrapper = ModelWrapperBias(model_name)
    rows = len(templates)
    fig, subplots = plt.subplots(rows, 1, figsize=(8, 12))
    axs = subplots
    if rows == 1:
        axs = [axs]
    dfs = []
    for k, template in enumerate(templates):
        ax = axs[k]
        labels, biases, b_keys, p_values, b_keys_p = [], [], [], [], []
        for bias_key, (terms, name) in bias_d.items():
            labels.append(name)
            b_keys.append(bias_key)
            if permutations:
                p_vale, effect_size, std = model_wrapper.permutation_value(*terms, template, permutations=permutations)
                p_values.append(effect_size)
                b_keys_p.append(bias_key)
            else:
                effect_size, std = model_wrapper.weat_prob_effect_size(*terms, template)
            print(name, effect_size, std, end='\n')
            biases.append(effect_size)
        dfs.append(pd.DataFrame([biases + p_values, ], columns=b_keys + b_keys_p, index=[template]))
        y_pos = np.arange(len(labels))
        hbars = ax.barh(y_pos, biases, align='center', fill=False, hatch='..')
        ax.set_yticks(y_pos, labels=labels)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('WEAT Effect Size')
        ax.set_title(f'{template}')

        # Label with given captions, custom padding and annotate options
        # ax.bar_label(hbars, labels=['±%.2f' % e for e in error], padding=8, color='b', fontsize=14)
        x_lim = 1.8
        ax.set_xlim(left=-x_lim, right=x_lim)
        ax.axvline(x=0, color='black', )

    df = pd.concat(dfs, axis=0)
    print("Max values of Effect Size are at following columns :")
    print(df.idxmax())  # before adding model name
    df['model'] = model_name
    if RUN_NR:
        df['run'] = RUN_NR
    fig.suptitle(f'Biases {model_name}')
    fig.tight_layout()

    plt.show()
    return df


if __name__ == "__main__":

    model_names = ['mistral-7b-instruct', 'mistral-7b', 'gpt2', ]  # ['gpt2', 'gpt2-large', 'gpt-neo',  't5-small',] #

    bias_d = get_bias_dic(model_names)
    # RPAM Test Bias Measurement
    fname = f'bias_models_{RUN_NR}all.csv'
    if path.exists(path.join(OUTPUT_EXPERIMENTS, fname)) is False:
        run_weat_model(model_names, bias_d)
    run_plot_models_biases(fname=fname, model_names=model_names)
