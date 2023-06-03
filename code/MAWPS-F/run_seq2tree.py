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

batch_size = 4
hidden_size = 512
n_epochs = 40
learning_rate = 0.0008
bert_lr = 8e-6
weight_decay = 1e-5
beam_size = 5
n_layers = 2
embedding_size = 768
dropout = 0.5
BertTokenizer_path = "bert_model/bert-base-uncased" 
Bert_model_path = "bert-base-uncased" 
best_acc = []
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S", filename='log')

def get_formula_fold(data,pairs):
    new_fold = []
    for item,pair in zip(data, pairs):
        pair = list(pair)
        pair.append(item['formula'])
        pair = tuple(pair)
        new_fold.append(pair)
    return new_fold

def load_formulas(file):
    with open(file,'r') as f:
        formula = f.read()
    formulas = formula.split('\n')
    formulas = [i.strip(' ') for i in formulas]
    formula_ent = set(formula.replace('\n',' ').split(' '))- set("+-*/()=")-{''}
    return formulas, formula_ent
    
all_formulas, all_formula_ent = load_formulas('data/formula.txt')
## all_formula_exp
# todo: autonomous generating
all_formulas_exp, _ = load_formulas('data/formula_variant.txt')

formula_exp_dict = dict(zip(all_formulas_exp, range(len(all_formulas_exp))))
formula_exp_dict['None'] = len(formula_exp_dict)
formula_ent_dict = dict(zip(all_formula_ent, range(len(all_formula_ent))))

for fold in range(5):
    train_data_path = "./data/fold"+str(fold)+"/train.json"
    test_data_path = "./data/fold"+str(fold)+"/dev.json"
    train_data = load_raw_data1(train_data_path)
    test_data = load_raw_data1(test_data_path)
    pairs1, generate_nums1, copy_nums1 = transfer_num1(train_data,use_bert_flag=True,model_name=BertTokenizer_path)
    pairs2, _ , _ = transfer_num1(test_data,use_bert_flag=True,model_name=BertTokenizer_path)
    pairs_trained = get_formula_fold(train_data, pairs1)
    pairs_tested = get_formula_fold(test_data, pairs2)
    input_lang, output_lang, train_pairs, test_pairs = prepare_data1(pairs_trained, pairs_tested, 5, generate_nums1,copy_nums1, bert_path=BertTokenizer_path, formula_exp_dict=formula_exp_dict, tree=True)

    # Initialize models
    embedding = BertEncoder(Bert_model_path,dropout=dropout)
    encoder = EncoderSeq(embedding_size=768 , hidden_size=hidden_size,n_layers=n_layers,dropout=dropout)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums1 - 1 - len(generate_nums1),
                            input_size=len(generate_nums1))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums1 - 1 - len(generate_nums1),
                            embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
    formula_enc = Formula_Encoding(formula_exp_dict=formula_exp_dict, formula_ent_dict=formula_ent_dict, hidden_size=hidden_size, embedding_size=embedding_size, word2index=output_lang.word2index)
    
    embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=bert_lr, weight_decay=weight_decay)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
    formula_pretrain_optimizer = torch.optim.Adam(formula_enc.parameters(), lr=learning_rate, weight_decay=weight_decay)
    formula_enc_optimizer = torch.optim.Adam(formula_enc.parameters(), lr=learning_rate, weight_decay=weight_decay)

    embedding_scheduler = torch.optim.lr_scheduler.StepLR(embedding_optimizer, step_size=10, gamma=0.5)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=10, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=10, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=10, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=10, gamma=0.5)
    formula_pretrain_scheduler = torch.optim.lr_scheduler.StepLR(formula_pretrain_optimizer, step_size=10, gamma=0.5)
    formula_enc_scheduler = torch.optim.lr_scheduler.StepLR(formula_enc_optimizer, step_size=10, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        embedding.cuda()
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
    # formula understanding pretaining
    for epoch in range(100):
        formula_pretrain_scheduler.step()
        loss_pretrain = 0
        loss = pretrain_formula(formula_enc, formula_pretrain_optimizer)
        loss_pretrain += loss
        logging.info(f"loss_pretrain: {loss_pretrain}")
        
        formula_acc = eval_formula(formula_enc)
        logging.info(f"formula acc: {formula_acc}")
        if formula_acc > best_formula_acc:
            best_formula_acc = formula_acc
            torch.save(formula_enc.state_dict(), "model_traintest/formula_enc")
    formula_enc.load_state_dict(torch.load("model_traintest/formula_enc"))
    
    for epoch in range(n_epochs):
        embedding_scheduler.step()
        encoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        formula_enc_scheduler.step()
        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, formula_batches = prepare_train_batch1(train_pairs, batch_size)

        logging.info(f"fold: {str(fold)}, epoch: {epoch + 1}")
        start = time.time()
        for idx in range(len(input_lengths)):
            loss = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], generate_num_ids, embedding,encoder, predict, generate, merge, formula_enc,
                embedding_optimizer,encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, formula_enc_optimizer, input_lang, output_lang, num_pos_batches[idx], formula_batches[idx])
            loss_total += loss

        logging.info(f"loss: {loss_total / len(input_lengths)}")
        logging.info(f"training time: {time_since(time.time() - start)}")
        # valid
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()

        for test_batch in test_pairs:
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids,embedding,encoder, predict, generate,
                                        merge, formula_enc, input_lang,output_lang, test_batch[5] , test_batch[7], text=test_batch[0], tokenizer=bert_tokenizer, gt_symbol=test_batch[2], beam_size=beam_size)
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
            torch.save(encoder.state_dict(), "model_traintest/encoder")
            torch.save(predict.state_dict(), "model_traintest/predict")
            torch.save(generate.state_dict(), "model_traintest/generate")
            torch.save(merge.state_dict(), "model_traintest/merge")
            torch.save(formula_enc.state_dict(), "model_traintest/formula_enc")
    
    best_acc.append((best_val_cor,best_eval_total))     
            
total_value_corr = 0
total_len = 0
folds_scores=[]
for w in range(len(best_acc)):
    folds_scores.append(float(best_acc[w][0])/best_acc[w][1])
    total_value_corr += best_acc[w][0]
    total_len += best_acc[w][1]
fold_acc_score = float(total_value_corr)/total_len
print("fold0-fold4 value accs: ",folds_scores)
print("final Val score: ",fold_acc_score)

