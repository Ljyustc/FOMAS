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

def get_rule_fold(data,pairs):
    new_fold = []
    for item,pair in zip(data, pairs):
        pair = list(pair)
        pair.append(item['rule'])
        pair = tuple(pair)
        new_fold.append(pair)
    return new_fold

def load_rules(file):
    with open(file,'r') as f:
        rule = f.read()
    rules = rule.split('\n')
    rules = [i.strip(' ') for i in rules]
    rule_ent = set(rule.replace('\n',' ').split(' '))- set("+-*/()=")-{''}
    return rules, rule_ent
    
all_rules, all_rule_ent = load_rules('data/mawps_rules.txt')
## all_rule_exp
# todo: autonomous generating
all_rules_exp, _ = load_rules('data/mawps_rules_expand.txt')

rule_exp_dict = dict(zip(all_rules_exp, range(len(all_rules_exp))))
rule_exp_dict['None'] = len(rule_exp_dict)
rule_ent_dict = dict(zip(all_rule_ent, range(len(all_rule_ent))))

for fold in range(5):
    train_data_path = "./data/fold"+str(fold)+"/train.json"
    test_data_path = "./data/fold"+str(fold)+"/dev.json"
    train_data = load_raw_data1(train_data_path)
    test_data = load_raw_data1(test_data_path)
    pairs1, generate_nums1, copy_nums1 = transfer_num1(train_data,use_bert_flag=True,model_name=BertTokenizer_path)
    pairs2, _ , _ = transfer_num1(test_data,use_bert_flag=True,model_name=BertTokenizer_path)
    pairs_trained = get_rule_fold(train_data, pairs1)
    pairs_tested = get_rule_fold(test_data, pairs2)
    input_lang, output_lang, train_pairs, test_pairs = prepare_data1(pairs_trained, pairs_tested, 5, generate_nums1,copy_nums1, bert_path=BertTokenizer_path, rule_exp_dict=rule_exp_dict, tree=True)

    # Initialize models
    embedding = BertEncoder(Bert_model_path,dropout=dropout)
    encoder = EncoderSeq(embedding_size=768 , hidden_size=hidden_size,n_layers=n_layers,dropout=dropout)
    predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums1 - 1 - len(generate_nums1),
                            input_size=len(generate_nums1))
    generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums1 - 1 - len(generate_nums1),
                            embedding_size=embedding_size)
    merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
    rule_enc = Rule_Encoding(rule_exp_dict=rule_exp_dict, rule_ent_dict=rule_ent_dict, hidden_size=hidden_size, embedding_size=embedding_size, word2index=output_lang.word2index)
    
    embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=bert_lr, weight_decay=weight_decay)
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rule_pretrain_optimizer = torch.optim.Adam(rule_enc.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rule_enc_optimizer = torch.optim.Adam(rule_enc.parameters(), lr=learning_rate, weight_decay=weight_decay)

    embedding_scheduler = torch.optim.lr_scheduler.StepLR(embedding_optimizer, step_size=10, gamma=0.5)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=10, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=10, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=10, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=10, gamma=0.5)
    rule_pretrain_scheduler = torch.optim.lr_scheduler.StepLR(rule_pretrain_optimizer, step_size=10, gamma=0.5)
    rule_enc_scheduler = torch.optim.lr_scheduler.StepLR(rule_enc_optimizer, step_size=10, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        embedding.cuda()
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()
        rule_enc.cuda()

    generate_num_ids = []
    for num in generate_nums1:
        generate_num_ids.append(output_lang.word2index[num])
    best_val_cor = 0
    best_eval_total  = 1
    
    bert_tokenizer = BertTokenizer.from_pretrained(BertTokenizer_path, do_lower_case=True)
    best_rule_acc = 0
    # rule understanding pretaining
    for epoch in range(100):
        rule_pretrain_scheduler.step()
        loss_pretrain = 0
        loss = pretrain_rule(rule_enc, rule_pretrain_optimizer)
        loss_pretrain += loss
        logging.info(f"loss_pretrain: {loss_pretrain}")
        
        rule_acc = eval_rule(rule_enc)
        logging.info(f"rule acc: {rule_acc}")
        if rule_acc > best_rule_acc:
            best_rule_acc = rule_acc
            torch.save(rule_enc.state_dict(), "model_traintest/rule_enc")
    rule_enc.load_state_dict(torch.load("model_traintest/rule_enc"))
    
    for epoch in range(n_epochs):
        embedding_scheduler.step()
        encoder_scheduler.step()
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        rule_enc_scheduler.step()
        loss_total = 0
        input_batches, input_lengths, output_batches, output_lengths, nums_batches, num_stack_batches, num_pos_batches, num_size_batches, rule_batches = prepare_train_batch1(train_pairs, batch_size)

        logging.info(f"fold: {str(fold)}, epoch: {epoch + 1}")
        start = time.time()
        for idx in range(len(input_lengths)):
            # logging.info(f"batch: {len(input_batches[idx])}, rule_batch: {len(rule_batches[idx])}")
            loss = train_tree(
                input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
                num_stack_batches[idx], num_size_batches[idx], generate_num_ids, embedding,encoder, predict, generate, merge, rule_enc,
                embedding_optimizer,encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, rule_enc_optimizer, input_lang, output_lang, num_pos_batches[idx], rule_batches[idx])
            loss_total += loss

        logging.info(f"loss: {loss_total / len(input_lengths)}")
        logging.info(f"training time: {time_since(time.time() - start)}")
        # valid
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        
        rule_max_cor, rule_max_wrong, rule_max_none_num, rule_max_valid_num, rule_max_ft, rule_max_fn= 0, 0, 0, 0, 0, 0
        rule_min_cor, rule_min_wrong, rule_min_none_num, rule_min_valid_num, rule_min_ft, rule_min_fn= 0, 0, 0, 0, 0, 0
        
        if epoch == n_epochs-1:
            show = True
        else:
            show = False
        # pairs:input_seq, len(input_seq), out_seq(prefix with index), len(out_seq), nums, num_pos, num_stack
        for test_batch in test_pairs:
            test_res, max_cor, max_wrong, max_none_num, max_valid_num, max_ft, max_fn, min_cor, min_wrong, min_none_num, min_valid_num, min_ft, min_fn = evaluate_tree_1(test_batch[0], test_batch[1], generate_num_ids,embedding,encoder, predict, generate,
                                        merge, rule_enc, input_lang,output_lang, test_batch[5] , test_batch[7], text=test_batch[0], tokenizer=bert_tokenizer, gt_symbol=test_batch[2], beam_size=beam_size, s=show)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            else:
                if epoch == n_epochs-1:
                    text = bert_tokenizer.convert_ids_to_tokens(test_batch[0])
                    gt = [output_lang.index2word[i] for i in test_batch[2]]
                    pre = [output_lang.index2word[i] for i in test_res]
                    print("text:", text)
                    print("true:", gt)
                    print("predict", pre)
            if equ_ac:
                equation_ac += 1
            eval_total += 1
            
            rule_max_cor += max_cor
            rule_min_cor += min_cor
            rule_max_wrong += max_wrong
            rule_min_wrong += min_wrong
            rule_max_none_num += max_none_num
            rule_max_valid_num += max_valid_num
            rule_max_ft += max_ft
            rule_max_fn += max_fn
            rule_min_none_num += min_none_num
            rule_min_valid_num += min_valid_num
            rule_min_ft += min_ft
            rule_min_fn += min_fn
                
        # print(equation_ac, value_ac, eval_total)

        logging.info(f"valid_eq_acc: {float(equation_ac) / eval_total}, valid_ans_acc: {float(value_ac) / eval_total}")
        logging.info(f"max_rule_acc: {rule_max_cor / (rule_max_cor + rule_max_wrong)}, max_none_por: {rule_max_none_num / (rule_max_none_num + rule_max_valid_num)}, max_ft: {rule_max_ft / rule_max_valid_num}, max_fn: {rule_max_fn / rule_max_none_num}")
        logging.info(f"min_rule_acc: {rule_min_cor / (rule_min_cor + rule_min_wrong)}, min_none_por: {rule_min_none_num / (rule_min_none_num + rule_min_valid_num)}, min_ft: {rule_min_ft / rule_min_valid_num}, min_fn: {rule_min_fn / rule_min_none_num}")
        logging.info(f" time: {time_since(time.time() - start)}")
        if float(value_ac) / eval_total > best_val_cor / best_eval_total:
            best_val_cor = value_ac
            best_eval_total = eval_total
            torch.save(encoder.state_dict(), "model_traintest/encoder")
            torch.save(predict.state_dict(), "model_traintest/predict")
            torch.save(generate.state_dict(), "model_traintest/generate")
            torch.save(merge.state_dict(), "model_traintest/merge")
            torch.save(rule_enc.state_dict(), "model_traintest/rule_enc")
    
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

