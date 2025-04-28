from os.path import dirname, abspath, join

DATA_PATH = join(dirname(dirname(abspath(__file__))), 'data')
OUTPUT_PATH = join(DATA_PATH, 'output')
OUTPUT_EXPERIMENTS = join(OUTPUT_PATH, 'experiments')

misc_data = 'misc_data'
lexica = 'English Lexica'

p_belleza_lexicon = join(DATA_PATH, lexica,  'Bellezza_Lexicon.csv')
p_anew_lexicon = join(DATA_PATH, lexica, 'ANEW.csv')

p_ws353_csv = join(DATA_PATH, lexica, 'ws353', 'combined.csv')


def get_valence_token_name(model_name):
    return f'{model_name}_tokens_val_tox.json'