import random

from src.helper_functions import flatten
from src.models import get_model_tokenizer, get_kwargs_tokenizer
from src.settings import DO_MULTIPLY
from src.terms import pleasant, unpleasant, flower, insect, ea_name, aa_name, ea_name_2, \
    aa_name_2, career, domestic, male_name, female_name, male, female, mathematics, art, science, temporary, permanent, \
    mental, physical, african_american_female, european_american_male, intersectional_aaf, intersectional_eam, \
    intersectional_maf, mexican_american_females, instrument, weapon, pleasant_2, unpleasant_2, young, old, \
    emergent_intersectional_aaf, emergent_intersec_maf, emergent_intersec_eam, \
    emergent_intersectional_eam_2, emergent_intersectional_eam_3


def randomly_select_terms_length(lists, max_len=None):
    """create two lists of same length"""
    targetx, targety = lists
    len_t = min(len(targetx), len(targety))
    if max_len and len_t > max_len:
        len_t = max_len
    random.seed(77)
    random.shuffle(targetx)
    random.shuffle(targety)
    return targetx[:len_t], targety[:len_t]


def remove_token(targetx, targety, forbidden_terms):
    targetx = [t for t in targetx if t not in forbidden_terms]
    targety = [t for t in targety if t not in forbidden_terms]
    return randomly_select_terms_length([targetx, targety])


def remove_token_names(terms, forbidden_terms):
    new_list = []
    for targs in terms:
        new_list.append([t for t in targs if t not in forbidden_terms])
    len_t = min([len(targs) for targs in new_list])

    random.seed(77)
    for targs in new_list:
        random.shuffle(targs)
    short_list = [targs[:len_t] for targs in new_list]
    return short_list


def get_bias_dic_unmodified(biases_sel=None, include_intersectional=False):
    """order of terms: attribute1, attribute2, target1, target2.
    order of names: target1, target2, attribute1, attribute2"""

    # WEAT 1: We use the flower and insect target words along with pleasant and unpleasant attributes found in (5)
    terms_flowers = (pleasant, unpleasant, flower, insect)

    # WEAT 2: We use the musical instruments and weapons target words along with pleasant and unpleasant attributes found in (5).
    terms_weapons = (pleasant, unpleasant, instrument, weapon)
    # attribute1, attribute2, target1, target2,
    # WEAT 3: We use the European American and African American names along with pleasant and unpleasant attributes found in (5)
    racial_bias1 = (pleasant, unpleasant, ea_name, aa_name)

    # WEAT 4: We use the European American and African American names from (7), along with pleasant and unpleasant attributes found in (5)
    racial_bias2 = (pleasant, unpleasant, ea_name_2, aa_name_2)

    # WEAT 5: We use the European American and African American names from (7), along with pleasant and unpleasant attributes found in (9)
    # WEAT 5: different pleasant and unpleasant
    racial_bias3 = (pleasant_2, unpleasant_2, ea_name_2, aa_name_2)

    # WEAT 6: We use the male and female names along with career and family attributes found
    # Gender1
    gender1 = (career, domestic, male_name, female_name)

    # WEAT 7: We use the math and arts target words along with male and female attributes found
    gender2 = (male, female, mathematics, art)

    # WEAT 8: We use the science and arts target words along with male and female attributes
    gender3 = (male, female, science, art)


    # WEAT 9: We use the mental and physical disease target words along with uncontrollability and controllability attributes found in
    disease = (temporary, permanent, mental, physical)

    # WEAT 10: We use young and old people’s names as target words along with pleasant and unpleasant attributes found in (9).
    young_old = (pleasant, unpleasant, young, old)

    # CEAT 11 # in order male, female and "European", "non-European"
    ceatI1 = (intersectional_eam, intersectional_aaf, european_american_male, african_american_female,)

    # CEAT I 2
    # ceatI2 = (emergent_intersectional_aaf, emergent_intersectional_eam_2, african_american_female, european_american_male)
    ceatI2 = (
    emergent_intersectional_eam_2, emergent_intersectional_aaf, european_american_male, african_american_female,)

    # CEAT I3
    ceatI3 = (emergent_intersectional_eam_3, intersectional_maf, european_american_male, mexican_american_females,)

    # CEAT I4
    ceatI4 = (emergent_intersec_eam, emergent_intersec_maf, european_american_male, mexican_american_females,)

    # attribute1, attribute2, target1, target2,
    bias_d = {
        'flowers': (terms_flowers, 'C1 Flowers/Insects, P/U'),  # 1
        'weapons': (terms_weapons, 'C2 Instruments/Weapons, P/U'),  #
        'racial1': (racial_bias1, 'C3 EA/AA names, P/U'),  #
        'racial2': (racial_bias2, 'C4 EA/AA names 2, P/U'),
        'racial3': (racial_bias3, 'C5 EA/AA names, P2/U2 '),
        'gender1': (gender1, 'C6 Male/Female names, Career/Family'),
        'gender2': (gender2, 'C7 Math/Arts, Male/Female terms'),
        'gender3': (gender3, 'C8 Science/Arts, Male/Female terms'),
        'disease': (disease, 'C9 Mental/Physical disease, Temporary/Permanent'),
        'young_old': (young_old, 'C10 Young/Old names, P/U'),
    }
    if include_intersectional:
        int_d = {
            'ceatI1': (ceatI1, 'C11 EM/AF names, EM/AF intersectional'),
            'ceatI2': (ceatI2, 'C12 EM/AF names, EM intersectional/AF emergent'),
            'ceatI3': (ceatI3, 'C13 EM/MF names, EM/MF intersectional'),
            'ceatI4': (ceatI4, 'C14 EM/MF names, EM intersectional/MF emergent'),
        }
        bias_d.update(int_d)
    if biases_sel:
        bias_d = {k: v for k, v in bias_d.items() if k in biases_sel}

    weat_biases = [v[0] for v in bias_d.values()]
    return bias_d, weat_biases


def pat_target_names():
    bias_d = get_intersectional_list()
    name_d = {k: v[1] for k, v in bias_d.items()}
    return name_d


def get_multi(tokenizer, word_set):
    kwargs = get_kwargs_tokenizer(tokenizer)
    multi_li = []
    for w in word_set:
        tokens = tokenizer.tokenize(w, **kwargs)
        if len(tokens) > 1:
            multi_li.append(w)
    return multi_li


def get_bias_dic(model_name='gpt2', exclude_multiple_target=False, biases_sel=None):
    bias_d, weat_biases = get_bias_dic_unmodified(biases_sel)
    if DO_MULTIPLY and exclude_multiple_target is False:
        return bias_d

    _, tokenizer = get_model_tokenizer(model_name, load_model=False)
    set_attribute_terms = set()
    set_target_terms = set()
    multi_terms_list = []
    for li in weat_biases:
        for li_t in li[:2]:
            set_attribute_terms.update([t for t in li_t])
        for li_t in li[2:]:
            set_target_terms.update([t for t in li_t])

    for set_terms in (set_attribute_terms, set_target_terms):
        multi_terms = get_multi(tokenizer, set_terms)
        multi_terms_list.append(multi_terms)

    print('multi attribute and targets, resp:', len(multi_terms_list[0]), len(multi_terms_list[1]))
    new_bias_d = {}
    # removes on both, attributs and targets
    for key, (li, name) in bias_d.items():
        attribut_li = list(li[:2])
        target_li = list(li[2:])
        new_att_li, new_targ_li = [], []
        for k, (ali, newali) in enumerate([(attribut_li, new_att_li), (target_li, new_targ_li)]):
            multi_terms_i = multi_terms_list[k]  # either attribute or target
            # if key in LIST_BIAS_NAMES:
            # targs = ali[0]
            ali_mod = remove_token(*ali, multi_terms_i)
            newali.extend(ali_mod)
        newli = [attribut_li, target_li]
        if exclude_multiple_target:
            newli[1] = new_targ_li
        if DO_MULTIPLY is False:
            newli[0] = new_att_li
        new_bias_d[key] = newli, name

    return new_bias_d


def terms_task2():
    # from email
    target1 = ['woman', 'mother']
    target2 = ['man', 'father']
    attribute2 = ['health', 'happy']
    attribute1 = ['pollute', 'tragedy']
    all_terms = target1 + target2 + attribute1 + attribute2
    return target1, target2, attribute1, attribute2


def get_intersectional_list():
    l_tp = '{} and {}'
    # idea: add one category (class) and constituent identy: poor (in comparison to norm)
    meta_d = {
        'racial1_p1': (('poor', 'poor'), 'P1 EA names & poor/AA names & poor'),
        'gender1_p2': (('poor', 'poor'), 'P2 Male names & poor/Female names & poor'),
        'ceatI1_p3': (('poor', 'poor'), 'P3 EM names & poor/AF names & poor'),
    }

    d_bias, _ = get_bias_dic_unmodified()
    suffix = '_pat'
    d_new = {}
    for k, v in meta_d.items():
        kceat = k.split('_')[0]
        targ2s = v[0]
        targs_ceat = d_bias[kceat][0][2:]
        targs_new = []
        for i in range(2):
            if targ2s[i] == '':  # not used
                op = 1 if i == 0 else 1
                targs = targs_ceat[op]
            else:
                targs = [[l_tp.format(targ2s[i], tc), l_tp.format(tc, targ2s[i])] for tc in targs_ceat[i]]
                targs = flatten(targs)
            targs_new.append(targs)
        d_new[kceat] = (targs_ceat, d_bias[kceat][1])  # same as WEAT-WS
        d_new[k] = (targs_new, v[1])  # new
    return d_new


def get_valence_terms(mod_cl_name):
    bias_d_main = get_bias_dic(mod_cl_name)
    flowers, name = bias_d_main['flowers']
    pleasant, unpleasant = flowers[:2]
    return pleasant, unpleasant


def print_bias_t():
    """prints word stimuli for report.
    Attributes as for WEAT bias measurement -> C7, C8:  male, female terms
    """

    d, _ = get_bias_dic_unmodified()
    for k, v in d.items():

        name = v[1]
        print('\subsection{' + name + '}')
        for i in range(2):
            print('\subsubsection{Target ' + str(i + 1) + '}', end='\n')
            print(', '.join(v[0][i + 2]), end='\n')
        for i in range(2):
            print('\subsubsection{Attribute ' + str(i + 1) + '}', end='\n')
            print(', '.join(v[0][i]), end='\n')


def print_bias_pat_t():
    """prints word stimuli for report.
    """
    d = get_intersectional_list()
    for k, v in d.items():
        if not '_p' in k:
            continue
        name = v[1]
        name = name.replace('&', '\&')
        print('\subsection{' + name + '}')
        for i in range(2):
            print('\subsubsection{Target ' + str(i + 1) + '}', end='\n')
            print(', '.join(v[0][i]), end='\n')


def ceat_names_both(groupname, k, name_str, reverse=False):
    if reverse is True:
        target_names_str = name_str.split(', ')[1]
    else:
        target_names_str = name_str.split(', ')[0]

    names_both = [
        ' '.join([groupname, target_names_str.split('/')[0]]),
        ' '.join([groupname, target_names_str.split('/')[1]])
    ]

    for n in ('names', 'terms'):
        if n in target_names_str:
            names_both[0] = names_both[0] + ' ' + n
            if '2' in target_names_str:
                names_both[0] = names_both[0] + ' 2'
    names_both = tuple(names_both)
    return names_both, target_names_str


def remove_multiple_terms(model_name, term_lists):
    _, tokenizer = get_model_tokenizer(model_name, load_model=False)
    multi_i = get_multi(tokenizer, term_lists[0] + term_lists[1])
    atts = remove_token(*term_lists, multi_i)
    return atts
