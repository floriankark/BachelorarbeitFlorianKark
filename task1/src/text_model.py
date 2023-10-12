import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
import pandas as pd
import numpy as np
import json
import wandb
import copy
import os
os.environ["WANDB_MODE"]="offline" # HPC clusters are not connected to the internet
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disabling parallelism to avoid deadlocks
from torch.utils.data import Dataset
from datasets import load_from_disk, Dataset
from preprocess_tweets import preprocess_tweets
import argparse

class TweetImageDataLoader:
    def __init__(self, preprocessing_config, tokenizer, seed, 
                 ocr=False, 
                 description=False, 
                 ocr_or_description=False, 
                 ocr_and_description=False,
                 description_and_ocr=False, 
                 gold=False):
        self.tokenizer = tokenizer
        self.seed = seed
        self.ocr = ocr
        self.description = description
        self.ocr_or_description = ocr_or_description
        self.ocr_and_description = ocr_and_description
        self.description_and_ocr = description_and_ocr
        self.gold = gold
        self.preprocessing_config = preprocessing_config
        self.train, self.dev, self.test = self._read_data()
        self.train_ds, self.dev_ds, self.test_ds = self._create_datasets()

    def _read_data(self):
        columns_to_convert = ['tweet_text', 'ocr_text']
        
        dev = pd.read_csv('/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/dev_with_description_InBlip.csv', index_col=0, dtype={'tweet_text': str, 'ocr_text': str, 'description': str})
        dev[columns_to_convert] = dev[columns_to_convert].astype(str)

        dev['label'] = dev['class_label'].apply(lambda x: 1. if x == "Yes" else 0.)
        if self.ocr:
            dev['text'] = dev['tweet_text'].astype(str) + " " + dev['ocr_text'].astype(str)
        elif self.description:
            dev['text'] = dev['tweet_text'].astype(str) + " " + dev['description'].astype(str)
        elif self.ocr_and_description:
            dev['text'] = dev['tweet_text'].astype(str) + " " + dev['ocr_text'].astype(str) + " " + dev['description'].astype(str)
        elif self.description_and_ocr:
            dev['text'] = dev['tweet_text'].astype(str) + " " + dev['description'].astype(str) + " " + dev['ocr_text'].astype(str)
        elif self.ocr_or_description:
            dev.loc[dev['ocr_text'] == 'nan'] = dev['description'].astype(str)
            dev['text'] = dev['tweet_text'].astype(str) + " " + dev['ocr_text'].astype(str)
        else:
            dev['text'] = dev['tweet_text'].astype(str)
        dev = preprocess_tweets(dev, **self.preprocessing_config)
        dev = dev[['text', 'image_path', 'label']]

        if self.gold:
            test = pd.read_csv('/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/gold_with_description_InBlip.csv', index_col=0, dtype={'tweet_text': str, 'ocr_text': str, 'description': str})
        else:
            test = pd.read_csv('/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/dev_test_with_description_InBlip.csv', index_col=0, dtype={'tweet_text': str, 'ocr_text': str, 'description': str})
        test[columns_to_convert] = test[columns_to_convert].astype(str)

        test['label'] = test['class_label'].apply(lambda x: 1. if x == "Yes" else 0.)
        if self.ocr:
            test['text'] = test['tweet_text'].astype(str) + " " + test['ocr_text'].astype(str)
        elif self.description:
            test['text'] = test['tweet_text'].astype(str) + " " + test['description'].astype(str)
        elif self.ocr_and_description:
            test['text'] = test['tweet_text'].astype(str) + " " + test['ocr_text'].astype(str) + " " + test['description'].astype(str)
        elif self.description_and_ocr:
            test['text'] = test['tweet_text'].astype(str) + " " + test['description'].astype(str) + " " + test['ocr_text'].astype(str)
        elif self.ocr_or_description:
            test.loc[test['ocr_text'] == 'nan'] = test['description'].astype(str)
            test['text'] = test['tweet_text'].astype(str) + " " + test['ocr_text'].astype(str)
        else:
            test['text'] = test['tweet_text'].astype(str)
        test = preprocess_tweets(test, **self.preprocessing_config)
        test = test[['text', 'image_path', 'label']]

        train = pd.read_csv('/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/train_with_description_InBlip.csv', index_col=0, dtype={'tweet_text': str, 'ocr_text': str, 'description': str})
        train[columns_to_convert] = train[columns_to_convert].astype(str)

        train['label'] = train['class_label'].apply(lambda x: 1. if x == "Yes" else 0.)
        if self.ocr:
            train['text'] = train['tweet_text'].astype(str) + " " + train['ocr_text'].astype(str)
        elif self.description:
            train['text'] = train['tweet_text'].astype(str) + " " + train['description'].astype(str)
        elif self.ocr_and_description:
            train['text'] = train['tweet_text'].astype(str) + " " + train['ocr_text'].astype(str) + " " + train['description'].astype(str)
        elif self.description_and_ocr:
            train['text'] = train['tweet_text'].astype(str) + " " + train['description'].astype(str) + " " + train['ocr_text'].astype(str)
        elif self.ocr_or_description:
            train.loc[train['ocr_text'] == 'nan'] = train['description'].astype(str)
            train['text'] = train['tweet_text'].astype(str) + " " + train['ocr_text'].astype(str)
        else:
            train['text'] = train['tweet_text'].astype(str)
        train = preprocess_tweets(train, **self.preprocessing_config)
        train = train[['text', 'image_path', 'label']]

        train = train.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return train, dev, test

    def _create_datasets(self):
        if 1==0: #"dev_dataset" in os.listdir():
            train_dataset = load_from_disk("train_dataset", keep_in_memory=True)
            dev_dataset = load_from_disk("dev_dataset", keep_in_memory=True)
            test_dataset = load_from_disk("test_dataset", keep_in_memory=True)
        else:
            def process_example(examples):
                texts = [t for t in examples['text']]

                text_inputs = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=tokenizer.model_max_length)
                
                examples['input_ids'] = text_inputs['input_ids']
                examples['attention_mask'] = text_inputs['attention_mask']
                return examples
            
            train_dataset = Dataset.from_pandas(self.train)
            train_dataset = train_dataset.map(process_example, batched=True, batch_size=32,
                                            remove_columns=['text', 'image_path'])

            dev_dataset = Dataset.from_pandas(self.dev)
            dev_dataset = dev_dataset.map(process_example, batched=True, batch_size=32,
                                        remove_columns=['text', 'image_path'])

            test_dataset = Dataset.from_pandas(self.test)
            test_dataset = test_dataset.map(process_example, batched=True, batch_size=32,
                                            remove_columns=['text', 'image_path'])

            train_dataset.save_to_disk("train_dataset")
            dev_dataset.save_to_disk("dev_dataset")
            test_dataset.save_to_disk("test_dataset")

        return train_dataset, dev_dataset, test_dataset
    
class TextModel(nn.Module):
    def __init__(self,
                 text_model,
                 loss_fct=torch.nn.CrossEntropyLoss(),
                 ):
        
        super(TextModel, self).__init__()

        self.text_model = text_model

        hidden_size = self.text_model.config.hidden_size
        dropout_prob = self.text_model.config.hidden_dropout_prob

        # Standard classification head
        #self.classifier = nn.Linear(hidden_size, 2)

        # Roberta classification head
        self.classifier = nn.Sequential(nn.Dropout(dropout_prob),
                                        nn.Linear(hidden_size, hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(dropout_prob),
                                        nn.Linear(hidden_size, 2),
                                        )

        # Loss function
        self.loss_fct = loss_fct

    def forward(self, 
                input_ids=None,
                attention_mask=None,
                token_type_ids=None, 
                position_ids=None, 
                head_mask=None,
                labels=None
                ):

        outputs = self.text_model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                position_ids=position_ids,
                                #head_mask=head_mask,
                                )

        text_features = outputs.last_hidden_state

        # Classification head
        text_representation = text_features[:, 0, :]

        # Classification
        logits = self.classifier(text_representation)
        
        loss = self.loss_fct(logits.view(-1, 2), labels.view(-1).long())

        return SequenceClassifierOutput(loss=loss, 
                                        logits=logits, 
                                        )

def softmax(x):
    exp = np.exp(x)
    exp_sum = np.sum(np.exp(x), axis=1, keepdims=True)
    return exp / exp_sum


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def annotate_test_dataframe(pred_output):
    test_df['logits'] = pred_output.predictions[:, 1]
    test_df['pred'] = np.argmax(pred_output.predictions, 1)
    test_df['score'] = softmax(pred_output.predictions)[:, 1]

    annotated_test_data.append(test_df.copy())
    
def compute_fold_metrics(pred_output, should_annotate_test_df=False):
    metrics = {}

    labels = pred_output.label_ids

    predictions = np.argmax(pred_output.predictions, 1)
    acc = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions)
    rec = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    metrics.update({
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1})

    if should_annotate_test_df:
        annotate_test_dataframe(pred_output)

    return metrics

def compute_overall_metrics(data):
    overall_metrics = {}

    def get_pr_table(labels, scores):
        precs, recs, thresholds = precision_recall_curve(labels, scores)
        pr_df = pd.DataFrame({'threshold': thresholds, 'precision': precs[:-1], 'recall': recs[:-1]})
        pr_df = pr_df.sample(n=min(1000, len(pr_df)), random_state=0)
        pr_df = pr_df.sort_values(by='threshold')
        pr_table = wandb.Table(dataframe=pr_df)
        return pr_table

    scores = data['score']
    preds = data['pred'] == 1
    labels = data['label'] == 1

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    aps = average_precision_score(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    overall_metrics.update({'avg_acc': acc,
                            'avg_prec': prec,
                            'avg_rec': rec,
                            'avg_f1': f1,
                            'avg_aps': aps,
                            'avg_roc_auc': roc_auc,
                            })

    # log pr-curve
    pr_table = get_pr_table(labels, scores)
    overall_metrics.update({'pr_table': pr_table})

    return overall_metrics

class EvaluateCB(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        epoch = state.epoch
        metrics = {'epoch': epoch}

        train_pred_output = trainer.predict(train_dataset)
        if train_pred_output is not None:
            train_metrics = compute_fold_metrics(train_pred_output, False)
            train_metrics['loss'] = train_pred_output.metrics['test_loss']
            train_metrics = {f'train_eval/{k}': v for k, v in train_metrics.items()}

            metrics.update(train_metrics)

        dev_pred_output = trainer.predict(dev_dataset)
        if dev_pred_output is not None:
            dev_metrics = compute_fold_metrics(dev_pred_output, False)
            dev_metrics['loss'] = dev_pred_output.metrics['test_loss']
            dev_metrics = {f'dev/{k}': v for k, v in dev_metrics.items()}
            metrics.update(dev_metrics)

        test_pred_output = trainer.predict(test_dataset)
        if test_pred_output is not None:
            test_metrics = compute_fold_metrics(test_pred_output, epoch == epochs)
            test_metrics['loss'] = test_pred_output.metrics['test_loss']
            test_metrics = {f'test/{k}': v for k, v in test_metrics.items()}
            metrics.update(test_metrics)

        wandb.log(metrics)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer_max_len', type=int, default=128,
                        help='The maximum length of input sequences for the tokenizer.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train.')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for training.')
    parser.add_argument('--model', type=str, default="roberta-large",
                        help='Type of model to use.')
    parser.add_argument('--ocr', type=bool, default=False,
                        help='Use OCR data in training.')
    parser.add_argument('--description', type=bool, default=False,
                        help='Use description data in training.') 
    parser.add_argument('--ocr_or_description', type=bool, default=False,
                        help='Use description data where there is no OCR in training.')
    parser.add_argument('--ocr_and_description', type=bool, default=False,
                        help='Use OCR and description data in training.')
    parser.add_argument('--description_and_ocr', type=bool, default=False,
                        help='First description then OCR data concatenation.')
    parser.add_argument('--gold', type=bool, default=False,
                        help='Use gold data in training. Default is the dev test set.')               
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    for seed in [0,1,2]:
        run = wandb.init(project="clef_ocr_or_description")

        #text_model_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/twitter-roberta-large-2022-154m"
        text_model_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/" + args.model
        #text_model_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        #text_model_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        #text_model_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/bert-base-uncased"
        wandb.config['text_model_id_or_path'] = text_model_id_or_path

        #tokenizer_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/twitter-roberta-large-2022-154m"
        tokenizer_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/" + args.model
        #tokenizer_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
        #tokenizer_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        #tokenizer_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/bert-base-uncased"
        wandb.config['tokenizer_id_or_path'] = tokenizer_id_or_path 

        tokenizer_max_len = args.tokenizer_max_len
        wandb.config['tokenizer_max_len'] = tokenizer_max_len

        epochs = args.epochs
        wandb.config['epochs'] = epochs

        wandb.config['training_seed'] = seed

        dataloader_config = {'per_device_train_batch_size': 16,
                             'per_device_eval_batch_size': 64}
        wandb.config.update(dataloader_config)

        preprocessing_config = {'lowercase': True,
                                'normalize': True,
                                'urls': False,
                                'user_handles': '@USER',
                                'emojis': 'demojize'}
        wandb.config.update(preprocessing_config)

        learning_rate = args.learning_rate
        wandb.config['learning_rate'] = learning_rate

        num_labels = 2
        problem_type = "single_label_classification"

        wandb.config['num_labels'] = num_labels
        wandb.config['problem_type'] = problem_type

        freezed_until = ""
        wandb.config['freezed_until'] = freezed_until

        # name of the model
        # e.g. roberta-large_128_ocr_description or roberta-large_128_ocr
        model_type = args.model + "_" + str(args.tokenizer_max_len)
        if args.ocr:
            model_type += "_ocr"
        if args.description:
            model_type += "_description"
        if args.ocr_or_description:
            model_type += "_ocr_or_description"
        if args.ocr_and_description:
            model_type += "_ocr_and_description"
        if args.description_and_ocr:
            model_type += "_description_and_ocr"
        if args.gold:
            model_type += "_gold"
        if not args.gold:
            model_type += "_dev_test"

        wandb.config['model_type'] = model_type

        print('load models, tokenizer and processor')
        tokenizer_config = {'pretrained_model_name_or_path': tokenizer_id_or_path,
                            'max_len': tokenizer_max_len}
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)
        text_model = AutoModel.from_pretrained(text_model_id_or_path)
        model = TextModel(text_model)

        if freezed_until != "":
            assert freezed_until in [n for n, _ in model.named_parameters()]
            req_grad = False
            for n, param in model.named_parameters():
                param.requires_grad = req_grad
                if freezed_until == n:
                    req_grad = True

                print(f'Parameters {n} require grad: {param.requires_grad}')

        wandb.config['n_parameters'] = count_parameters(model)

        print('load data')
        dl = TweetImageDataLoader(
            preprocessing_config, tokenizer, seed, 
            ocr=args.ocr,
            description=args.description, 
            ocr_or_description=args.ocr_or_description, 
            ocr_and_description=args.ocr_and_description, 
            description_and_ocr=args.description_and_ocr, 
            gold=args.gold
            )

        annotated_test_data = []

        train_dataset = dl.train_ds
        dev_dataset = dl.dev_ds
        test_dataset = dl.test_ds
        test_df = dl.test.copy()

        training_args = TrainingArguments(
            output_dir="results",  # output directory
            num_train_epochs=epochs,  # total number of training epochs
            **dataloader_config,
            warmup_ratio=0.1,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            learning_rate=learning_rate,
            logging_dir='./logs',  # directory for storing logs
            logging_strategy='epoch',
            save_strategy='no',
            evaluation_strategy="no",  # evaluate each `logging_steps`
            no_cuda=False,
            report_to='wandb',
            dataloader_num_workers=10,
        )

        trainer = Trainer(
            model=model,  # the instantiated Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            callbacks=[EvaluateCB]
        )

        trainer.train()
        print('***** Finished Training *****\n\n\n')

        #print(f'Save model at .../task1/output/clef/{run.name}_model')
        #torch.save(model.state_dict() ,f'/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/output/clef/{run.name}_model')

        print('Evaluate all folds')
        data = pd.concat(annotated_test_data)
        run_name = model_type + "_" + str(seed)
        data.to_pickle(f"/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/output/clef/{run_name}_preds.pkl")
        metrics = compute_overall_metrics(data)
        wandb.log(metrics)
        wandb.finish()