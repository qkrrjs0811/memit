"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import AttributeSnippets
from util.generate import generate_fast
from util.perplexity import perplexity


# 실험을 위해서 임시로 설정
SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
    do_print=False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    generation_prompts = record["generation_prompts"]

    '''
        subject : Danielle Darrieux
        target_new : {'str': 'English', 'id': 'Q1860'}
        target_true : {'str': 'French', 'id': 'Q150'}
        rewrite_prompts : ['The mother tongue of Danielle Darrieux is']
        ...
        generation_prompts : ["Danielle Darrieux's mother tongue is"]
    '''

    def _print():
        print('\n### compute_rewrite_quality_counterfact()._print() ###\n')
        print(f'subject : {subject}')
        print(f'target_new : {target_new}')
        print(f'target_true : {target_true}')
        print(f'rewrite_prompts : {rewrite_prompts}')
        print(f'paraphrase_prompts : {paraphrase_prompts}')
        print(f'neighborhood_prompts : {neighborhood_prompts}')
        print(f'generation_prompts : {generation_prompts}\n\n')
    if do_print: _print()

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
    ]
    which_correct = [
        [0 for _ in range(len(rewrite_prompts))],
        [0 for _ in range(len(paraphrase_prompts))],
        [1 for _ in range(len(neighborhood_prompts))],
    ]
    # Flatten all the evaluated prefixes into one list.
    probs, targets_correct, logits = test_batch_prediction(
        model,
        tok,
        list(chain(*prob_prompts)),
        list(chain(*which_correct)),
        target_new["str"],
        target_true["str"],
        do_print=do_print
    )
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    ret_corrects = [
        targets_correct[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))
    ]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    } | {
        f"{key}_correct": ret_corrects[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
            ]
        )
    }

    if snips is not None:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        essence_texts = [
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts,
            consistency_texts,
            essence_texts,
            vec,
        )
        ret.update(gen_stats)

    return ret


def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    which_correct: str,
    target_new: str,
    target_true: str,
    print_prefix='',
    do_print=False
):
    """
    which_correct: Which target to consider correct. Either 0 for "new" or 1 for "true".
    """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    model.to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    probs = np.zeros((logits.size(0),), dtype=np.float32)
    targets_correct = []

    
    def _print():
        print(f'\n#################### {print_prefix} test_batch_prediction()._print() ####################\n')
        print(f'prefix_lens : {prefix_lens}')
        print(f'prefixes : {prefixes}')
        print(f'target_new : {target_new}')
        print(f'target_true : {target_true}\n')
        print(f'prompt_tok : {prompt_tok}')
        print(f'a_tok(n) : {tok.decode(a_tok)}({a_tok})')
        print(f'b_tok(t) : {tok.decode(b_tok)}({b_tok})\n')
        print(f'logits size : {logits.size()}, type : {type(logits)}\n')
    if do_print: _print()


    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len

        # Compute suffix probabilities
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]

            '''
                - [i]는 [target_new(english), target_true(french)] 2번 돌고
                    - [j]는 target의 토큰 단위로 돔 (단일 토큰이면 1번만 돔)

                    - logits size : torch.Size([2, 10, 50257])
                        - logits[i, prefix_lens[i // 2] + j - 1, :] -> shape = (1, 1, 50257)
                            - if i==0 : index = (0, 8, :)
                            - if i==1 : index = (1, 8, :)
            '''
            prob_temp = -torch.nn.functional.log_softmax(logits[i, prefix_lens[i // 2] + j - 1, :], dim=0)
            prob = prob_temp[cur_tok].item()
            if do_print: print(f'i=[{i}], j=[{j}] tok : {tok.decode(cur_tok)}({cur_tok}), prob : {prob}')

            # _cur_tok = (a_tok if i % 2 != 0 else b_tok)[j]
            # _prob = prob_temp[_cur_tok].item()
            # print(f'i=[{i}], j=[{j}] _tok : {tok.decode(_cur_tok)}({_cur_tok}), _prob : {_prob}\n')

            probs[i] += prob
        probs[i] /= cur_len
        if do_print: print(f'i=[{i}] prob_avg : {probs[i]}\n')

        # Compute accuracy on new targets
        if (which_correct[i // 2] == 0 and i % 2 == 0) or (
            which_correct[i // 2] == 1 and i % 2 == 1
        ):
            correct = True
            
            for j in range(cur_len):
                cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
                _tok = logits[i, prefix_lens[i // 2] + j - 1, :].argmax().item()
                if do_print: print(f'[{j}] predict : {tok.decode(_tok)}, target : {tok.decode(cur_tok)}')

                if cur_tok != _tok:
                    correct = False
                    break
            if do_print: print()

            targets_correct.append(correct)

    return [
        {"target_new": probs[i].item(), "target_true": probs[i + 1].item()}
        for i in range(0, len(probs), 2)
    ], targets_correct, logits


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()
