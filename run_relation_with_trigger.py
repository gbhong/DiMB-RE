"""
This code is based on the file in SpanBERT repo: https://github.com/facebookresearch/SpanBERT/blob/master/code/run_tacred.py
"""

import argparse
import logging
import os
import random
import time
import json
from tqdm import tqdm
import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from torch.nn import CrossEntropyLoss

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from relation.models import BertForRelation
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from relation.utils import generate_relation_data, decode_sample_id
from shared.const import task_rel_labels, task_ner_labels
from shared.utils import set_seed, make_output_dir

CLS = "[CLS]"
SEP = "[SEP]"

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Set the CuBLAS workspace configuration for deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def get_args():
    parser = argparse.ArgumentParser()

    # general arguments: Task, Directories, Train/Eval, and so on.
    parser.add_argument('--task', type=str, default=None, required=True,
                        help=f"Run one of the task in {list(task_ner_labels.keys())}")
    parser.add_argument('--pipeline_task', type=str, default=None, required=True,
                        help=f"Choose what kind of tasks to run: NER, Triplet, and RE")
    parser.add_argument("--do_train", action='store_true', 
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', 
                        help="Whether to run eval on the dev set.")
    parser.add_argument('--do_predict_dev', action='store_true', 
                        help="Whether to run prediction on dev set")
    parser.add_argument('--do_predict_test', action='store_true', 
                        help="Whether to run prediction on test set")
    parser.add_argument("--eval_with_gold", action="store_true", 
                        help="Whether to evaluate the relation model with gold entities provided.")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")                        
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--use_trigger", action='store_true', 
                        help="Whether to train and eval with triggers.")
    parser.add_argument("--load_saved_model", action='store_true',
                        help="Whether to use saved model already trained")
    
    # directory and file arguments
    parser.add_argument('--output_dir', type=str, default=None, required=True,
                        help="Output directory of the experiment outputs")
    parser.add_argument("--entity_output_dir", type=str, default=None,
                        help="Needs to define if you want to use predicted entities of dev set \
                            from different folder")
    parser.add_argument("--triplet_output_dir", type=str, default=None,
                        help="Needs to define if you want to use predicted entities of dev set \
                            from different folder") 
    parser.add_argument("--entity_output_test_dir", type=str, default=None,
                        help="Needs to define if you want to use predicted entities of test set \
                            from different folder")
    parser.add_argument("--triplet_output_test_dir", type=str, default=None,
                        help="Needs to define if you want to use predicted entities of test set \
                            from different folder") 
    parser.add_argument("--relation_output_dir", type=str, default=None,
                        help="Needs to define if you want to Eval and Test only, not training") 
    parser.add_argument("--entity_predictions_dev", type=str, 
                        default="ent_pred_dev.json", help="The entity prediction file of the dev set")
    parser.add_argument("--entity_predictions_test", type=str, 
                        default="ent_pred_test.json", help="The entity prediction file of the test set")
    parser.add_argument("--triplet_predictions_dev", type=str, 
                        default="trg_pred_dev.json", help="The entity prediction file of the dev set")
    parser.add_argument("--triplet_predictions_test", type=str, 
                        default="trg_pred_test.json", help="The entity prediction file of the test set")
    parser.add_argument("--dev_pred_filename", type=str, default="rel_pred_dev.json", 
                        help="The prediction filename for the relation model")
    parser.add_argument("--test_pred_filename", type=str, default="rel_pred_test.json", 
                        help="The prediction filename for the relation model")

    # data-specific arguments:     
    parser.add_argument("--train_file", default=None, type=str, 
                        help="The path of the training data.")
    parser.add_argument("--train_mode", type=str, default='random_sorted', 
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--dev_file", default=None, type=str, 
                        help="The path of the dev data.")
    parser.add_argument("--test_file", default=None, type=str, 
                        help="The path of the test data.")

    # training-specific arguments:
    parser.add_argument("--eval_per_epoch", default=1.0, type=float,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument('--context_window', type=int, default=100)
    parser.add_argument("--max_seq_length", default=300, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_epoch", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--max_patience', type=int, default=4)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--bertadam", action="store_true", 
                        help="If bertadam, then set correct_bias = False")
    parser.add_argument('--print_loss_step', type=int, default=10, 
                        help="how often logging the loss value during training")
    parser.add_argument("--sampling_proportion", default=0.0, type=float)

    # model arguments:
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument("--do_lower_case", default=True, 
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--add_new_tokens', default=True, 
                        help="Whether to add new tokens as marker tokens instead of using [unusedX] tokens.")

    args = parser.parse_args()
    return args

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sub_idx = sub_idx
        self.obj_idx = obj_idx
    

def add_marker_tokens(tokenizer, ner_labels):
    new_tokens = ['<SUBJ_START>', '<SUBJ_END>', '<OBJ_START>', '<OBJ_END>']
    for label in ner_labels:
        new_tokens.append('<SUBJ_START=%s>'%label)
        new_tokens.append('<SUBJ_END=%s>'%label)
        new_tokens.append('<OBJ_START=%s>'%label)
        new_tokens.append('<OBJ_END=%s>'%label)
    for label in ner_labels:
        new_tokens.append('<SUBJ=%s>'%label)
        new_tokens.append('<OBJ=%s>'%label)

    tokenizer.add_tokens(new_tokens)
    logger.info('# vocab after adding markers: %d'%len(tokenizer))

def random_undersampling(samples, ratio):
    class_0_samples = [sample for sample in samples if sample["relation"] == "no_relation"]
    class_1_samples = [sample for sample in samples if sample["relation"] != "no_relation"]

    logger.info(f"Length of No-Relation: {len(class_0_samples)}")
    logger.info(f"Length of Other Classes: {len(class_1_samples)}")

    # if len(class_0_samples) // len(class_1_samples) >= 10:
    logger.info("Now doing undersampling...")
    num_samples_to_keep = min(len(class_0_samples), int(len(class_1_samples) / ratio))
    undersampled_dataset = random.sample(class_0_samples, num_samples_to_keep) + class_1_samples
    random.shuffle(undersampled_dataset)
    return undersampled_dataset
    # else:
    #     return samples

def convert_examples_to_features(examples, label2id, max_seq_length, tokenizer, special_tokens, unused_tokens=True):
    """
    Loads a data file into a list of `InputBatch`s.
    unused_tokens: whether use [unused1] [unused2] as special tokens
    """

    def get_special_token(w):
        if w not in special_tokens:
            if unused_tokens:
                special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
            else:
                special_tokens[w] = ('<' + w + '>')
        return special_tokens[w]

    num_tokens = 0
    max_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    for (ex_index, example) in enumerate(examples):

        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        TRIGGER_START_NER = get_special_token("TRG_START")
        TRIGGER_END_NER = get_special_token("TRG_END")
        SUBJECT_START_NER = get_special_token("SUBJ_START=%s"%example['subj_type'])
        SUBJECT_END_NER = get_special_token("SUBJ_END=%s"%example['subj_type'])
        OBJECT_START_NER = get_special_token("OBJ_START=%s"%example['obj_type'])
        OBJECT_END_NER = get_special_token("OBJ_END=%s"%example['obj_type'])
        
        tokens = [CLS]
        for i, token in enumerate(example['token']):
            if i == example['subj_start']:
                sub_idx = len(tokens)
                tokens.append(SUBJECT_START_NER)
            if i == example['obj_start']:
                obj_idx = len(tokens)
                tokens.append(OBJECT_START_NER)
            if ('trg_start' in example) and (i == example['trg_start']):
                trg_idx = len(tokens)
                tokens.append(TRIGGER_START_NER)
            # for sub_token in tokenizer.tokenize(token):
            for sub_token in tokenizer.tokenize(token):
                tokens.append(sub_token)
            if i == example['subj_end']:
                tokens.append(SUBJECT_END_NER)
            if i == example['obj_end']:
                tokens.append(OBJECT_END_NER)
            if ('trg_end' in example) and (i == example['trg_end']):
                tokens.append(TRIGGER_END_NER)
        tokens.append(SEP)

        num_tokens += len(tokens)
        max_tokens = max(max_tokens, len(tokens))

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
            if sub_idx >= max_seq_length:
                sub_idx = 0
            if obj_idx >= max_seq_length:
                obj_idx = 0
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = label2id[example['relation']]
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if num_shown_examples < 20:
            if (ex_index < 5) or (label_id > 0):
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example['id']))
                logger.info("tokens: %s" % " ".join(
                        [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s (id = %d)" % (example['relation'], label_id))
                logger.info("sub_idx, obj_idx: %d, %d" % (sub_idx, obj_idx))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                sub_idx=sub_idx,
                obj_idx=obj_idx
            )
        ) 
                       
    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("Max #tokens: %d"%max_tokens)
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                num_fit_examples * 100.0 / len(examples), max_seq_length))
    return features

def save_div(a, b):
    if b != 0:
        return a / b 
    else:
        return 0.0

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_f1(preds, labels, e2e_ngold, label_cnt_dict, id2label):

    assert e2e_ngold == sum(label_cnt_dict.values())

    result_by_class = {}
    for idx, label in id2label.items():
        if idx != 0:
            result_by_class[label] = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 
                                      'gold': 0.0, 'pred': 0.0, 'correct': 0.0}
            result_by_class[label]["gold"] = label_cnt_dict[label]

    n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
            result_by_class[id2label[pred]]["pred"] += 1
            if pred == label:
                n_correct += 1
                result_by_class[id2label[pred]]["correct"] += 1
            
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}, result_by_class
    else:
        for label in result_by_class:
            counts = result_by_class[label]
            counts["precision"] = save_div(counts["correct"], counts["pred"])
            counts["recall"] = save_div(counts["correct"], counts["gold"])
            counts["f1"] = save_div(2*counts["precision"]*counts["recall"], counts["precision"]+counts["recall"])

        prec = save_div(n_correct, n_pred)
        rec = save_div(n_correct, e2e_ngold)
        f1 = save_div(2*prec*rec, prec+rec)
        logger.info('Prec: %.5f, Rec: %.5f, F1: %.5f'%(prec, rec, f1))

        result = {'precision': prec, 'recall': rec, 'f1': f1, \
        'n_correct': n_correct, 'n_pred': n_pred, 'n_gold': e2e_ngold}
        return result, result_by_class


def evaluate(model, device, eval_dataloader, eval_label_ids, id2label, label_cnt_dict, e2e_ngold=None, verbose=True):
    
    model.eval()

    num_labels = len(id2label)
    eval_loss = 0
    nb_eval_steps = 0
    preds = []
    for input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        sub_idx = sub_idx.to(device)
        obj_idx = obj_idx.to(device)
        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None, sub_idx=sub_idx, obj_idx=obj_idx)
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    logits = preds[0]
    preds = np.argmax(preds[0], axis=1)
    result, result_by_class = compute_f1(
        preds, eval_label_ids.numpy(), e2e_ngold=e2e_ngold, label_cnt_dict=label_cnt_dict, id2label=id2label
        )
    result['accuracy'] = simple_accuracy(preds, eval_label_ids.numpy())
    result['eval_loss'] = eval_loss
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return preds, result, logits, result_by_class

def print_pred_json(eval_data, eval_examples, preds, id2label, output_file, use_gold=False):
    rels = dict()
    for ex, pred in zip(eval_examples, preds):
        doc_sent, sub, obj = decode_sample_id(ex['id'])
        if doc_sent not in rels:
            rels[doc_sent] = []
        if pred != 0:
            rels[doc_sent].append([sub[0], sub[1], obj[0], obj[1], id2label[pred], ""])

    js = eval_data.js
    for doc in js:
        doc['predicted_relations'] = []
        for sid in range(len(doc['sentences'])):
            k = '%s@%d'%(doc['doc_key'], sid)
            doc['predicted_relations'].append(rels.get(k, []))

        # TODO: if you want to use Trigger for this task, modify this
        if use_gold:
            doc['predicted_ner'] = doc['ner']
    
    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc) for doc in js))

def save_trained_model(model, tokenizer, step, args):
    """
    Save the model to the output directory
    """
    os.makedirs(args.relation_output_dir, exist_ok=True)
    logger.info('Saving model to %s with Step %d ...'%(args.relation_output_dir, step))

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.relation_output_dir)
    tokenizer.save_vocabulary(args.relation_output_dir)

def main() -> None:
    args = get_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    # Generate specific version of output folder
    if args.relation_output_dir is None or args.load_saved_model:
        if args.load_saved_model:
            args.saved_model_dir = args.relation_output_dir
        args.relation_output_dir = make_output_dir(
            args.output_dir, task='relation', pipeline_task=args.pipeline_task
        )
        os.makedirs(args.relation_output_dir, exist_ok=True)

    # Specify the entity/triplet output folder
    if args.entity_output_dir is None:
        path_components = args.relation_output_dir.split(os.path.sep)
        path_components[-1] = "entity"
        args.entity_output_dir = os.path.sep.join(path_components)
    if args.use_trigger and args.triplet_output_dir is None:
        path_components = args.relation_output_dir.split(os.path.sep)
        path_components[-1] = "triplet"
        args.triplet_output_dir = os.path.sep.join(path_components)

    if not args.use_trigger:  # 2-step without trigger
        if args.eval_with_gold:
            entity_dev_path = args.dev_file
        else:
            entity_dev_path = os.path.join(args.entity_output_dir, args.entity_predictions_dev)
        if args.do_predict_test:
            entity_test_path = os.path.join(args.entity_output_test_dir, args.entity_predictions_test)
    else:  # 3-step with trigger
        if args.eval_with_gold:
            entity_dev_path = args.dev_file
        else:
            entity_dev_path = os.path.join(args.triplet_output_dir, args.triplet_predictions_dev)
        if args.do_predict_test:
            entity_test_path = os.path.join(args.triplet_output_test_dir, args.entity_predictions_test)

    if args.do_train:
        # train set
        train_dataset, train_examples, *_  = generate_relation_data(
            args.train_file, 
            training=True, 
            use_gold=True, 
            use_trigger=args.use_trigger, 
            context_window=args.context_window
        )
        # dev set
        if args.do_eval:
            eval_dataset, eval_examples, eval_nrel, eval_label_dict = generate_relation_data(
                entity_dev_path, 
                training=False, 
                use_gold=args.eval_with_gold, 
                use_trigger=args.use_trigger, 
                context_window=args.context_window
            )
        elif args.do_predict_test:
            eval_dataset, eval_examples, eval_nrel, eval_label_dict = generate_relation_data(
                entity_dev_path, 
                training=False, 
                use_gold=True, 
                use_trigger=args.use_trigger, 
                context_window=args.context_window
            )
            # incorporate train set with dev set
            logger.info("## Now moving Dev data to Train data... ##")
            train_examples.extend(eval_examples)
            logger.info(f"## Length of Train data: {len(train_examples)} ##")
            
        if args.sampling_proportion:
            train_examples = random_undersampling(train_examples, ratio=args.sampling_proportion)

    if not args.do_train and not (args.do_predict_dev or args.do_predict_test):
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.relation_output_dir, f"train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.relation_output_dir, f"predict.log"), 'w'))

    logger.info(f"args: {json.dumps(args.__dict__, indent=2, sort_keys=True)}")
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

    # get label_list
    if os.path.exists(os.path.join(args.relation_output_dir, 'label_list.json')):
        with open(os.path.join(args.relation_output_dir, 'label_list.json'), 'r') as f:
            label_list = json.load(f)
    else:
        label_list = [args.negative_label] + task_rel_labels[args.task]
        with open(os.path.join(args.relation_output_dir, 'label_list.json'), 'w') as f:
            json.dump(label_list, f)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    RelationModel = BertForRelation
    # RelationModel = BertForRelation(multitask=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.add_new_tokens:
        add_marker_tokens(tokenizer, task_ner_labels[args.task])

    if os.path.exists(os.path.join(args.relation_output_dir, 'special_tokens.json')):
        with open(os.path.join(args.relation_output_dir, 'special_tokens.json'), 'r') as f:
            special_tokens = json.load(f)
    else:
        special_tokens = {}

    if args.do_eval:
        eval_features = convert_examples_to_features(
            eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens, unused_tokens=not(args.add_new_tokens)
        )
        logger.info("***** Dev *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in eval_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in eval_features], dtype=torch.long)
        # all_trg_idx = torch.tensor([f.trg_idx for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        eval_label_ids = all_label_ids

    with open(os.path.join(args.relation_output_dir, 'special_tokens.json'), 'w') as f:
        json.dump(special_tokens, f)

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label2id, args.max_seq_length, tokenizer, special_tokens, unused_tokens=not(args.add_new_tokens)
        )
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in train_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in train_features], dtype=torch.long)
        # all_trg_idx = torch.tensor([f.trg_idx for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = len(train_dataloader) * args.num_epoch

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        print("eval step >>> ", eval_step)
        
        lr = args.learning_rate
        model = RelationModel.from_pretrained(
            args.model, 
            cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), 
            num_rel_labels=num_labels,
            use_trigger=args.use_trigger
            )
        if hasattr(model, 'bert'):
            model.bert.resize_token_embeddings(len(tokenizer))
        elif hasattr(model, 'albert'):
            model.albert.resize_token_embeddings(len(tokenizer))
        else:
            raise TypeError("Unknown model class")

        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=not(args.bertadam))
        scheduler = get_linear_schedule_with_warmup(optimizer, int(num_train_optimization_steps * args.warmup_proportion), num_train_optimization_steps)

        start_time = time.time()
        global_step = 0
        tr_loss = 0
        nb_tr_examples = 0
        nb_tr_steps = 0
        eval_step = len(train_batches) // args.eval_per_epoch

        max_patience, current_patience = args.max_patience, 0
        if_exit = False

        for epoch in range(1, args.num_epoch+1):

            if if_exit:
                logger.info("=== Do EARLY STOPPING ===")
                break

            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)

            progress = tqdm.tqdm(total=len(train_batches), ncols=150, desc="Epoch: " + str(epoch))
        
            for step, batch in enumerate(train_batches):

                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx = batch

                loss = model(input_ids, segment_ids, input_mask, label_ids, sub_idx, obj_idx)
                if n_gpu > 1:
                    loss = loss.mean()

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.print_loss_step == 0:
                    logger.info('Epoch=%d, step=%d, loss=%.5f'%(epoch, step, tr_loss / nb_tr_examples))
                    tr_loss = 0
                    nb_tr_examples = 0

                if args.do_eval and global_step % eval_step == 0:
                    preds, result, logits, result_by_class = evaluate(
                        model, device, eval_dataloader, eval_label_ids, id2label=id2label, label_cnt_dict=eval_label_dict, e2e_ngold=eval_nrel
                    )
                    model.train()
                    result['global_step'] = global_step
                    result['epoch'] = epoch
                    result['learning_rate'] = lr
                    result['batch_size'] = args.train_batch_size

                    if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                        best_result = result
                        logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                    (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))                                      
                        save_trained_model(model, tokenizer, global_step, args)
                        current_patience = 0
                    else:
                        if result[args.eval_metric] != 0:
                            current_patience += 1
                            if current_patience >= max_patience:
                                if_exit = True
                progress.update(1)
            progress.close()

    if not args.do_eval and args.do_train:
        logger.info('## Without Validation Set: Saving model to %s... ##'%(args.relation_output_dir))
        save_trained_model(model, tokenizer, global_step, args)

    # logger.info(special_tokens)

    if args.do_predict_dev:
        model_dir = args.relation_output_dir if not args.load_saved_model else args.saved_model_dir
        model = RelationModel.from_pretrained(
            model_dir, num_rel_labels=num_labels, use_trigger=args.use_trigger
        )
        model.to(device)
        preds, result, logits, result_by_class = evaluate(
                            model, device, eval_dataloader, eval_label_ids, id2label=id2label, label_cnt_dict=eval_label_dict, e2e_ngold=eval_nrel
                        )
        with open(os.path.join(args.relation_output_dir, "dev_result_by_class.json"), 'w', encoding='utf-8') as f_out:
            f_out.write(json.dumps(result_by_class, indent=4))

        logger.info('*** Final Dev Set Results ***')
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        rel_pred_file_path = os.path.join(args.relation_output_dir, args.dev_pred_filename)
        print_pred_json(eval_dataset, eval_examples, preds, id2label, rel_pred_file_path, use_gold=args.eval_with_gold)

    if args.do_predict_test:
        test_dataset, test_examples, test_nrel, test_label_dict = generate_relation_data(
            entity_test_path, 
            training=False, 
            use_gold=args.eval_with_gold, 
            use_trigger=args.use_trigger, 
            context_window=args.context_window
        )
        test_features = convert_examples_to_features(
            test_examples, 
            label2id, 
            args.max_seq_length, 
            tokenizer, 
            special_tokens, 
            unused_tokens=not(args.add_new_tokens)
        )

        logger.info("***** Test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_sub_idx = torch.tensor([f.sub_idx for f in test_features], dtype=torch.long)
        all_obj_idx = torch.tensor([f.obj_idx for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_sub_idx, all_obj_idx)
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size)
        test_label_ids = all_label_ids

        # Load the fine-tuned model (TRAIN+DEV)
        model_dir = args.relation_output_dir if not args.load_saved_model else args.saved_model_dir
        model = RelationModel.from_pretrained(
            model_dir, 
            num_rel_labels=num_labels, 
            use_trigger=args.use_trigger
        )
        model.to(device)
        preds, result, logits, result_by_class = evaluate(
            model, 
            device, 
            test_dataloader, 
            test_label_ids, 
            id2label=id2label, 
            label_cnt_dict=test_label_dict, 
            e2e_ngold=test_nrel, 
            verbose=False
        )
        with open(os.path.join(args.relation_output_dir, "test_result_by_class.json"), 'w', encoding='utf-8') as f_out:
            f_out.write(json.dumps(result_by_class, indent=4))

        logger.info('*** Final Test Set Results ***')
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        rel_test_file_path = os.path.join(args.relation_output_dir, args.test_pred_filename)
        print_pred_json(test_dataset, test_examples, preds, id2label, rel_test_file_path, use_gold=args.eval_with_gold)


if __name__ == "__main__":
    main()
