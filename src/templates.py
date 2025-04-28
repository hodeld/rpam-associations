TEMPLATES_PAPER = [9, 22] # [8, 9, 17, 22]


TEMPLATE_D = {
    9: 'These words are associated: {} and ',
    22: 'This sentence and this word are associated: "This is {}" and "',
}


LEXICON_TEMPLATE = '{} and {}'
LEXICON_TEMPLATE_5TARGETS = '{} and {} and {} and {} and {}'



def get_template_list(keys=None):
    if keys:
        templates = [v for k, v in TEMPLATE_D.items() if k in keys]
    else:
        templates = TEMPLATE_D.values()
    return templates


def get_template_dict(keys=None):
    if keys:
        template_d = {k: v for k, v in TEMPLATE_D.items() if k in keys}
    else:
        template_d = TEMPLATE_D
    return template_d


def get_template_from_key(template_key):
    return TEMPLATE_D[template_key]
