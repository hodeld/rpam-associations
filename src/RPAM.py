import numpy as np
import pandas as pd
import torch
from transformers import set_seed

from src.helper_functions import std_deviation, create_permutation
from src.models import get_model_tokenizer
from src.settings import DO_MULTIPLY
from scipy.stats import norm

TEMPLATE = 'These words are associated: {} and '
RUN_NR = None


class ModelWrapperBias:
    def __init__(self, model_name, model_cls_name=None, use_cuda=True):
        self._device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        if model_cls_name is None:
            model_cls_name = model_name
        if 'gpt' in model_cls_name:
            model_family = 'gpt'
        elif 'mistral' in model_cls_name:
            model_family = 'mistral'
        elif 'olmo' in model_cls_name:
            model_family = 'olmo'
        else:
            model_family = 't5'
            model_cls_name = model_family
        model, tokenizer = get_model_tokenizer(model_name, model_cls_name)
        self.model_cls_name = model_cls_name
        model = model.to(self._device)  # Not compatible with BitsandBytes! needed in case of cuda on Colab
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token  # needed for padding in query_model_batch
        self.model_family = model_family

    def _query_model_batch_t5(self, input_texts):
        input_texts = [input_text + ' <extra_id_0>' for input_text in input_texts]  # with or without space -> same
        output_texts = ['<extra_id_0>'] * len(input_texts)
        inputs = self.tokenizer.batch_encode_plus(input_texts, padding=True, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_ids = self.tokenizer.batch_encode_plus(output_texts, return_tensors='pt')['input_ids'].to(self._device)
        return self.model(labels=output_ids, **inputs)['logits'][:, 1, :]

    def _query_model_batch_gpt(self, input_texts):
        inputs = self.tokenizer.batch_encode_plus(input_texts, padding=False, return_tensors='pt',
                                                  return_token_type_ids=False)  # return_token_type_ids=False for olmo
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        output_indices = inputs['attention_mask'].sum(dim=1) - 1  # indices of last_word of complete inputs
        output = self.model(**inputs)[
            'logits']  # same as .logits; no unit, Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax) for each input.
        return torch.stack(
            [output[example_idx, last_word_idx, :] for example_idx, last_word_idx in enumerate(output_indices)])

    def _generate_model_gpt(self, prompt, max_new_tokens=50, num_return_sequences=1, seed=1):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        inputs = inputs.to(self._device)
        # Generate text (you can adjust parameters like max_length, num_return_sequences, etc.)
        set_seed(seed)
        output = self.model.generate(inputs, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences, )
        # temperature=0, top_p=0)

        # Decode the output minus the input to readable text
        generated_text = self.tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        # print(generated_text)
        return generated_text

    def _generate_model_mistral_sentiment(self, prompt, max_new_tokens=1, num_return_sequences=1, seed=1):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')

        inputs = inputs.to(self._device)
        set_seed(seed)
        output = self.model.generate(inputs, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=max_new_tokens,
                                     temperature=0.0)
        # Decode the output minus the input to readable text
        generated_text = self.tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True)
        print(generated_text)
        return generated_text

    def _generate_model_t5(self, prompt, max_length=50, num_return_sequences=1):
        raise NotImplementedError

    @torch.no_grad
    def sentiment_finetuned(self, input_text, seed=1):
        if self.model_name != 'gpt2-sentiment':
            raise ValueError('Model is not a sentiment model')

        inputs = self.tokenizer.batch_encode_plus([input_text], padding=False, return_tensors='pt')
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        # first element corresponds to prob for positive sentiment.argmax(axis=1).tolist() softmax(axis=1)[:, 1]
        return self.model(**inputs).logits.argmax(axis=1).tolist()[0]

    @torch.no_grad
    def generate_from_prompt(self, prompt, max_length=50, num_return_sequences=1, seed=1):
        if self.model_cls_name == 'mistral-classifier':
            return self._generate_model_mistral_sentiment(prompt, max_length, num_return_sequences, seed)
        elif self.model_family in ['gpt', 'mistral', 'olmo']:
            return self._generate_model_gpt(prompt, max_length, num_return_sequences, seed)
        else:
            return self._generate_model_t5(prompt, max_length, num_return_sequences)

    def _query_model_gpt(self, input_texts, output_id):
        inputs = self.tokenizer.batch_encode_plus(input_texts, padding=True,
                                                  return_tensors='pt')  # -> tokenizerfrom Huggingface
        inputs = {key: val.to(self._device) for key, val in inputs.items()}
        return self.model(**inputs)['logits'][0].softmax(dim=0)[output_id]

    def _get_kwargs(self):
        # add_prefix_space This allows to treat the leading word just as any other word
        return {'add_prefix_space': True} if self.model_family == 'gpt' else {}

    @torch.no_grad  # since we only use inference we can disable gradient calculation for reducing memory consumption
    def query_model_batch(self, input_texts):
        logits = []  # we need to obtain each input individually to avoid padding
        for input_text in input_texts:
            input_texts_i = [input_text]
            if self.model_family in ['gpt', 'mistral', 'olmo']:
                res = self._query_model_batch_gpt(input_texts_i)
            else:
                res = self._query_model_batch_t5(input_texts_i)
            logits.append(res[0])
        return logits

    def get_output_probabilities(self, templ, targs, atts):
        def remove_start_of_word_token(tk):
            tk = tk.replace('Ġ', '')  # GPT
            return tk.replace('_', '')  # Mistral, OLMo

        input_txts = [templ.format(t) for t in targs]
        output_ids = []
        logits_multi = []
        kwargs = self._get_kwargs()
        logits = self.query_model_batch(input_txts)
        multi_ind = {}
        if isinstance(atts[0], np.int64) or isinstance(atts[0], int):  # if id (df.index
            output_ids = atts
            return input_txts, logits, output_ids, multi_ind, logits_multi

        input_txts_multi = []
        multi_index = 0
        for i, word in enumerate(atts):
            tokens = self.tokenizer.tokenize(word, **kwargs)
            if DO_MULTIPLY is False:
                assert len(tokens) == 1, f"Word {word} consists of multiple tokens: {tokens}"
            assert tokens[
                       0] not in self.tokenizer.all_special_tokens, f"Word {word} corresponds to a special token: {tokens[0]}"
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_id = token_ids[0]
            if DO_MULTIPLY and len(token_ids) > 1:
                # print('multiply tokenized word:', word)
                for k, inp_t in enumerate(input_txts):
                    inp_t_i = inp_t + remove_start_of_word_token(tokens[0])
                    input_txts_multi_idx, token_ids_i = [], []
                    k_d = multi_ind.get(k, {})
                    for token, token_id_i in zip(tokens[1:], token_ids[1:]):
                        input_txts_multi.append(inp_t_i)
                        token_ids_i.append(token_id_i)
                        input_txts_multi_idx.append(multi_index)
                        multi_index += 1
                        inp_t_i = inp_t_i + remove_start_of_word_token(token)
                    k_d[token_id] = input_txts_multi_idx, token_ids_i
                    multi_ind[k] = k_d
            output_ids.append(token_id)
        if len(input_txts_multi) > 0:
            # print('multi input_txts')
            logits_multi = self.query_model_batch(input_txts_multi)

        return input_txts, logits, output_ids, multi_ind, logits_multi

    def _get_token_probability_distribution(self, attributes, targets, template):
        """
        For a batch of targets, returns the probability DISTRIBUTION over possible next tokens
        considering only the given list of attribute choices.
        :param attributes: the allowed attributes (output choices, must correspond to single tokens in the model's vocabulary)
        :param targets: the input texts
        :param template: The context template
        :return: a list of lists, where output[i][j] is a (output, probability) tuple for the ith input and jth output choice.
        """

        input_texts, logits, output_ids, multi_ind, multi_logits = self.get_output_probabilities(template, targets,
                                                                                                 attributes)
        result = []
        # print('calculate probabilities')
        for idx, _ in enumerate(input_texts):
            output_probabilities = logits[idx][output_ids].softmax(dim=0)  #
            if multi_ind.get(idx, None):
                d = multi_ind.pop(idx)
                # print('multi', idx)
                for k, (input_txts_multi_idx, token_ids_i) in iter(d.items()):
                    prob = 1
                    for logits_id, token_id_i in zip(input_txts_multi_idx, token_ids_i):
                        logits_i = multi_logits[logits_id]
                        prob_i = logits_i.softmax(dim=0)[token_id_i].item()
                        prob *= prob_i

                    j = output_ids.index(k)
                    output_probabilities[j] = output_probabilities[j] * prob
            result.append(output_probabilities)
        return result  #

    def get_token_probability_distribution(self, attributes, targets, template):
        return self._get_token_probability_distribution(attributes, targets, template)

    def get_single_probability(self, attributes, targets, template, id_x):
        res = self.get_token_probability_distribution(attributes, targets, template)
        return res[0][id_x].item()

    def get_most_likely_outputs(self, targets, template, output_ids=None):
        dfn = []
        for k, t in enumerate(targets):
            input_txts = [template.format(t)]
            ten = self.query_model_batch(input_txts)
            df = pd.DataFrame(ten[0][output_ids].softmax(dim=0).tolist(), columns=['probability'], index=output_ids)
            dfn.append(df)
        return dfn

    def get_all_tokens_series(self):
        d = self.tokenizer.get_vocab()
        ts = pd.Series(d.keys(), index=d.values())
        ts = ts.str.replace('Ġ', '')  # chr like 'À' still tokenized into 2 tokens
        return ts

    def get_all_tokens(self):
        ts = self.get_all_tokens_series()
        return ts.index, ts.to_list()

    def get_tokens_from_ids(self, token_ids):
        d = self.tokenizer.convert_ids_to_tokens(token_ids)
        ts = pd.Series(d, index=token_ids)
        ts = ts.str.replace('Ġ', '')  # chr like 'À' still tokenized into 2 tokens
        return ts.index, ts.to_list()

    def weat_prob_effect_size(self, attributes_a, attributes_b, targets_x, targets_y,
                              template=TEMPLATE):  # or A, B, X, Y

        attributes_a_b = attributes_a + attributes_b
        len_a = len(attributes_a)

        def get_distrution(targets):
            output_a_b = self.get_token_probability_distribution(attributes_a_b, targets, template)
            dist = []
            for probs_a_b in output_a_b:
                probs_a, probs_b = probs_a_b[:len_a], probs_a_b[len_a:]
                mean_a = np.mean(probs_a.tolist())
                mean_b = np.mean(probs_b.tolist())
                dist.append(mean_a - mean_b)
            return dist

        distribution_x = get_distrution(targets_x)
        distribution_y = get_distrution(targets_y)
        return (np.mean(distribution_x) - np.mean(distribution_y)) / std_deviation(distribution_x + distribution_y), \
               std_deviation(distribution_x + distribution_y)

    def sc_weat_prob_effect_size(self, attributes_a, attributes_b, target, template=TEMPLATE):  # or A, B, X, Y

        attributes_a_b = attributes_a + attributes_b
        len_a = len(attributes_a)

        def get_distrution():
            output_a_b = self.get_token_probability_distribution(attributes_a_b, [target], template)
            probs_a_b = output_a_b[0]
            # for probs_a_b in output_a_b:
            probs_a, probs_b = probs_a_b[:len_a], probs_a_b[len_a:]
            dist_a = probs_a.tolist()
            dist_b = probs_b.tolist()
            return dist_a, dist_b

        distribution_a, distribution_b = get_distrution()
        joint_distribution = distribution_a + distribution_b

        return ((np.mean(distribution_a) - np.mean(distribution_b)) / std_deviation(joint_distribution)), std_deviation(
            joint_distribution)

    def single_targets_prob_effect_size(self, attributes_a, attributes_b, targets, template=TEMPLATE):  # or A, B, X, Y

        attributes_a_b = attributes_a + attributes_b
        len_a = len(attributes_a)

        def get_distrution():
            output_a_b = self.get_token_probability_distribution(attributes_a_b, targets, template)
            dist_a, dist_b = [], []
            dist = []
            for probs_a_b in output_a_b:
                probs_a, probs_b = probs_a_b[:len_a], probs_a_b[len_a:]
                mean_a = np.mean(probs_a.tolist())
                mean_b = np.mean(probs_b.tolist())
                dist.append(mean_a - mean_b)

                dist_a.extend(probs_a.tolist())
                dist_b.extend(probs_b.tolist())
            return dist_a, dist_b

        distribution_a, distribution_b = get_distrution()
        joint_distribution = distribution_a + distribution_b

        return ((np.mean(distribution_a) - np.mean(distribution_b)) / std_deviation(joint_distribution)), std_deviation(
            joint_distribution)

    def permutation_test(self, attributes_a, attributes_b, targets_x, targets_y, test_stat, df_name, permutations):
        "returns: one-sided p-value of permutation test"

        distribution = []
        if len(attributes_a) + len(attributes_b) + len(targets_x) + len(targets_y) == 8:
            permutations = 6
        for nrp in range(permutations):
            if (nrp + 1) % 50 == 0:
                print(f'{nrp} permutations')
                print('mean / test_stat:', np.mean(distribution), test_stat)
            j, k = create_permutation(targets_x, targets_y)
            m, _ = self.weat_prob_effect_size(attributes_a, attributes_b, j, k)
            distribution.append(m)

        dist_mean = np.mean(distribution)
        dist_dev = std_deviation(distribution)  # sample standard deviation
        # one-sided
        if df_name == 'normal':  # normal
            p_value = (1 - norm.cdf(test_stat, dist_mean,
                                    dist_dev))  # norm.cdf returns cumulative distr. fn for val x, with shape parameters μ and σ
        else:  # empirical
            p_value = np.mean(distribution > test_stat)

        return p_value

    def permutation_value(self, attributes_a, attributes_b, targets_x, targets_y, dist='norm', permutations=1000,
                          template=TEMPLATE):
        test_statistic, std = self.weat_prob_effect_size(attributes_a, attributes_b, targets_x, targets_y,
                                                         template=template)
        p_value = self.permutation_test(attributes_a, attributes_b, targets_x, targets_y, test_statistic, dist,
                                        permutations)
        return p_value, test_statistic, std

    def compute_loss(self, input_ids: torch.LongTensor, labels: torch.LongTensor) -> torch.Tensor:
        outputs = self.model(input_ids, labels=labels)
        return outputs.loss

    def output_ids(self, token_list):
        output_choice_ids, multiply_toks = [], []
        kwargs = self._get_kwargs()
        for word in token_list:
            tokens = self.tokenizer.tokenize(word, **kwargs)
            token_id = self.tokenizer.convert_tokens_to_ids(tokens)[0]
            if len(tokens) > 1:
                print(f"Word {word} consists of multiple tokens: {tokens}")
                multiply_toks.append(token_id)
            if tokens[0] in self.tokenizer.all_special_tokens:
                print(f"Word {word} corresponds to a special token: {tokens[0]}")
            output_choice_ids.append(token_id)
        return output_choice_ids, multiply_toks


def inference_model(prompt, model_wrapper, max_new_tokens=50, seed=1):
    return model_wrapper.generate_from_prompt(prompt, max_length=max_new_tokens, seed=seed)
