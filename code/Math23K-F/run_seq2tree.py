# coding: utf-8
import os
from src.config import args
os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *
import logging
import random
import numpy as np

torch.cuda.manual_seed(0)  
np.random.seed(0)  
random.seed(0)  

batch_size = 10
hidden_size = 512
n_epochs = 85
learning_rate = 3e-4 
bert_lr = 3e-5 
weight_decay = 1e-5
beam_size = 5
n_layers = 2
embedding_size = 128
dropout = 0.5
model_file = "model_traintest"
if not os.path.isdir(model_file):
    os.makedirs(model_file)
BertTokenizer_path = "bert_model/bert-base-chinese" 
Bert_model_path = "bert-base-chinese"
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", filename='log')

def get_formula_fold(data,pairs):
    new_fold = []
    for item,pair in zip(data, pairs):
        pair = list(pair)
        pair.append(item['formula'])
        pair.append(item['id'])
        pair = tuple(pair)
        new_fold.append(pair)
    return new_fold

def load_formulas(file):
    with open(file,'r') as f:
        formula = f.read()
    formulas = formula.split('\n')
    formulas = [i.strip(' ') for i in formulas]
    formula_ent = set(formula.replace('\n',' ').split(' '))- set("+-*/^()=")-{''}
    return formulas, formula_ent
    
all_formulas, all_formula_ent = load_formulas('data/formula.txt')
## all_formula_exp
# todo: autonomous generating
all_formulas_exp, _ = load_formulas('data/formula_variant.txt')

formula_exp_dict = dict(zip(all_formulas_exp, range(len(all_formulas_exp))))
formula_exp_dict['None'] = len(formula_exp_dict)
formula_ent_dict = dict(zip(all_formula_ent, range(len(all_formula_ent))))

train_data_path = "./data/train23k_processed.json"
test_data_path = "./data/test23k_processed.json"
train_data = load_raw_data1(train_data_path)
test_data = load_raw_data1(test_data_path)
pairs1, generate_nums1, copy_nums1 = transfer_num1(train_data,use_bert_flag=True,model_name=BertTokenizer_path)
pairs2, _ , _ = transfer_num1(test_data,use_bert_flag=True,model_name=BertTokenizer_path)
temp_pairs1 = []
for p in pairs1:
    temp_pairs1.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs1 = temp_pairs1
#pairs：input_seq, out_seq(prefix), nums, num_pos
temp_pairs2 = []
for p in pairs2:
    temp_pairs2.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs2 = temp_pairs2
#pairs：input_seq, out_seq(prefix), nums, num_pos
pairs_trained = get_formula_fold(train_data, pairs1)
pairs_tested = get_formula_fold(test_data, pairs2)
input_lang, output_lang, train_pairs, test_pairs = prepare_data1(pairs_trained, pairs_tested, 5, generate_nums1,copy_nums1, bert_path=BertTokenizer_path, formula_exp_dict=formula_exp_dict, tree=True)

del train_data,test_data,pairs1,pairs2,temp_pairs1,temp_pairs2,pairs_trained,pairs_tested

# Initialize models
encoder = EncoderBert(hidden_size=hidden_size,bert_pretrain_path=Bert_model_path,dropout=dropout)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums1 - 1 - len(generate_nums1),
                        input_size=len(generate_nums1))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums1 - 1 - len(generate_nums1),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
formula_enc = Formula_Encoding(formula_exp_dict=formula_exp_dict, formula_ent_dict=formula_ent_dict, hidden_size=hidden_size, embedding_size=embedding_size, word2index=output_lang.word2index)


encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=bert_lr, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
formula_pretrain_optimizer = torch.optim.Adam(formula_enc.parameters(), lr=2e-4, weight_decay=weight_decay)
formula_enc_optimizer = torch.optim.Adam(formula_enc.parameters(), lr=bert_lr, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=30, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=30, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=30, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=30, gamma=0.5)
formula_pretrain_scheduler = torch.optim.lr_scheduler.StepLR(formula_pretrain_optimizer, step_size=30, gamma=0.5)
formula_enc_scheduler = torch.optim.lr_scheduler.StepLR(formula_enc_optimizer, step_size=30, gamma=0.5)

# formula_enc.load_state_dict(torch.load(model_file + "/formula_enc"))
# encoder.load_state_dict(torch.load(model_file + "/encoder"))
# predict.load_state_dict(torch.load(model_file + "/predict"))
# generate.load_state_dict(torch.load(model_file + "/generate"))
# merge.load_state_dict(torch.load(model_file + "/merge"))


# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()
    formula_enc.cuda()

generate_num_ids = []
for num in generate_nums1:
    generate_num_ids.append(output_lang.word2index[num])
best_val_cor = 0
best_eval_total  = 1

bert_tokenizer = BertTokenizer.from_pretrained(BertTokenizer_path, do_lower_case=True)
best_formula_acc = 0

# formula pretaining
for epoch in range(150):
    formula_pretrain_scheduler.step()
    loss_pretrain = 0
    loss = pretrain_formula(formula_enc, formula_pretrain_optimizer)
    loss_pretrain += loss
    logging.info(f"loss_pretrain: {loss_pretrain}")
    
    formula_acc = eval_formula(formula_enc)
    logging.info(f"formula acc: {formula_acc}")
    if formula_acc > best_formula_acc:
        best_formula_acc = formula_acc
        torch.save(formula_enc.state_dict(), model_file + "/formula_enc")

formula_enc.load_state_dict(torch.load(model_file + "/formula_enc"))
logging.info(f"formula pretaining over")

for epoch in range(n_epochs):
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    formula_enc_scheduler.step()
    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, formula_batches = prepare_train_batch1(train_pairs, batch_size)

    logging.info(f"epoch: {epoch + 1}")
    start = time.time()
    for idx in range(len(input_lengths)):
        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge, formula_enc,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, formula_enc_optimizer, input_lang, output_lang, num_pos_batches[idx], formula_batches[idx])
        loss_total += loss

    logging.info(f"loss: {loss_total / len(input_lengths)}")
    logging.info(f"training time: {time_since(time.time() - start)}")
    # valid
    value_ac = 0
    equation_ac = 0
    eval_total = 0
    start = time.time()

    for test_batch in test_pairs:
        test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids,encoder, predict, generate,
                                    merge, formula_enc, input_lang,output_lang, test_batch[5] , formula_label=test_batch[7], text=test_batch[0], tokenizer=bert_tokenizer, gt_symbol=test_batch[2], beam_size=beam_size)
        val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
        if val_ac:
            value_ac += 1
        if equ_ac:
            equation_ac += 1
        eval_total += 1

    logging.info(f"valid_eq_acc: {float(equation_ac) / eval_total}, valid_ans_acc: {float(value_ac) / eval_total}")
    logging.info(f" time: {time_since(time.time() - start)}")
    if float(value_ac) / eval_total > best_val_cor / best_eval_total:
        best_val_cor = value_ac
        best_eval_total = eval_total
        torch.save(encoder.state_dict(), model_file + "/encoder")
        torch.save(predict.state_dict(), model_file + "/predict")
        torch.save(generate.state_dict(), model_file + "/generate")
        torch.save(merge.state_dict(), model_file + "/merge")
        torch.save(formula_enc.state_dict(), model_file + "/formula_enc")

