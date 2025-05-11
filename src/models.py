import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from src.api_secrets import HF_API_TOKEN
from huggingface_hub import InferenceClient
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GPT2LMHeadModel, T5ForConditionalGeneration, T5Tokenizer, \
    AutoTokenizer, AutoModelForCausalLM, \
    GPT2ForSequenceClassification


def get_model_tokenizer(model_name='gpt2', model_cls_name=None, load_model=True):
    mod_cls_d = {'gpt-neo': (GPTNeoForCausalLM, GPT2Tokenizer),
                 'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
                 't5': (T5ForConditionalGeneration, T5Tokenizer),
                 'mistral': (AutoModelForCausalLM, AutoTokenizer),
                 # 'mistral': (MistralForCausalLM, LlamaTokenizer),
                 'olmo': (AutoModelForCausalLM, AutoTokenizer),
                 'gpt2-classifier': (GPT2ForSequenceClassification, GPT2Tokenizer),
                 'mistral-classifier': (AutoModelForCausalLM, AutoTokenizer),
                 }
    if model_name in ('EleutherAI/gpt-neo-125M', 'gpt_neo_orig', 'gpt-neo'):
        mod = 'EleutherAI/gpt-neo-125M'
        model_cls_name = 'gpt-neo'
        model_name_tokenizer = 'gpt2'
    elif model_name == 'gpt2':
        mod = model_name
        model_cls_name = model_name
        model_name_tokenizer = 'gpt2'
    elif model_name == 't5-small':
        mod = 'google/t5-v1_1-small'  # 'google/t5-v1_1-xl'
        model_cls_name = 't5'
        model_name_tokenizer = mod
    elif model_name == 'gpt2-large':
        mod = model_name
        model_cls_name = 'gpt2'
        model_name_tokenizer = model_name
    elif model_name == 'mistral-7b':
        mod = 'mistralai/Mistral-7B-v0.1'
        model_cls_name = 'mistral'
        model_name_tokenizer = mod
    elif model_name == 'mistral-7b-instruct':
        mod = 'mistralai/Mistral-7B-Instruct-v0.2' #
        model_cls_name = 'mistral'
        model_name_tokenizer = mod
    elif model_name == 'olmo-1b':
        mod = 'allenai/OLMo-1B'
        model_cls_name = 'olmo'
        model_name_tokenizer = mod
    elif model_name == 'gpt2-sentiment':
        mod = 'michelecafagna26/gpt2-medium-finetuned-sst2-sentiment'
        if model_cls_name != 'gpt2-classifier':
            model_cls_name = 'gpt2'
        model_name_tokenizer = mod

    model_cls, tokenizer_cls = mod_cls_d[model_cls_name]
    if load_model:
        print(f'loading model {mod}')
    if model_cls_name == 'mistral' or model_cls_name == 'mistral-classifier':
        if load_model:

            model = AutoModelForCausalLM.from_pretrained(
                mod,
            )
            model.config.use_cache = False
        else:
            model = None
    elif model_cls_name == 'gpt2-classifier':
        if load_model: # for classification
            model = model_cls.from_pretrained(mod)
        else:
            model = None
    else:
        if load_model:
            model = model_cls.from_pretrained(mod, output_hidden_states=True, output_attentions=False)
        else:
            model = None
    tokenizer = tokenizer_cls.from_pretrained(model_name_tokenizer)
    if model is not None:
        print('loaded model', model_name)
    return model, tokenizer


def get_kwargs_tokenizer(tokenizer):
    if isinstance(tokenizer, GPT2Tokenizer):
        kwargs = {'add_prefix_space': True}  # changes from "Hello world" to " Hello world"
    else:
        kwargs = {}
        print('kwargs t5, mistral')
    return kwargs


def get_inference_model(model_name='mistral-7b-instruct'):
    print(f'accessing model {model_name} through huggingface api')
    model_name_dic = {
                        'mistral-7b-instruct': 'mistralai/Mistral-7B-Instruct-v0.2',
                        'mistral-7b': 'mistralai/Mistral-7B-v0.1',
                        'olmo-7b-instruct': 'allenai/OLMo-7B-Instruct',
    }
    # get api model
    client = InferenceClient(model=model_name_dic[model_name], token=HF_API_TOKEN)
    return client


