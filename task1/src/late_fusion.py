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
    ViTModel,
    ViTImageProcessor,
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
import math
import copy
import os
os.environ["WANDB_MODE"]="offline" # HPC clusters are not connected to the internet
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disabling parallelism to avoid deadlocks
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_from_disk, Dataset
from preprocess_tweets import preprocess_tweets

class TweetImageDataLoader:
    def __init__(self, preprocessing_config, processor, tokenizer, seed):
        self.processor = processor
        self.tokenizer = tokenizer
        self.seed = seed
        self.preprocessing_config = preprocessing_config
        self.train, self.dev, self.test = self._read_data()
        self.train_ds, self.dev_ds, self.test_ds = self._create_datasets()

    def _read_data(self):
        with open("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/CT23_1A_checkworthy_multimodal_english_dev.jsonl", "r") as fp:
            dev = [json.loads(a) for a in fp.readlines()]

        dev = pd.DataFrame(dev)
        dev['label'] = dev['class_label'].apply(lambda x: 1. if x == "Yes" else 0.)
        dev['text'] = dev['tweet_text']
        dev = preprocess_tweets(dev, **self.preprocessing_config)
        dev = dev[['text', 'image_path', 'label']]

        with open("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/CT23_1A_checkworthy_multimodal_english_dev_test.jsonl", "r") as fp:
            test = [json.loads(a) for a in fp.readlines()]

        test = pd.DataFrame(test)
        test['label'] = test['class_label'].apply(lambda x: 1. if x == "Yes" else 0.)
        test['text'] = test['tweet_text']
        test = preprocess_tweets(test, **self.preprocessing_config)
        test = test[['text', 'image_path', 'label']]

        with open("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/CT23_1A_checkworthy_multimodal_english_train.jsonl", "r") as fp:
            train = [json.loads(a) for a in fp.readlines()]

        train = pd.DataFrame(train)
        train['label'] = train['class_label'].apply(lambda x: 1. if x == "Yes" else 0.)
        train['text'] = train['tweet_text']
        train = preprocess_tweets(train, **self.preprocessing_config)
        train = train[['text', 'image_path', 'label']]
        
        return train, dev, test

    def _create_datasets(self):
        if 1 == 0: #"dev_dataset" in os.listdir():
            train_dataset = load_from_disk("train_dataset", keep_in_memory=True)
            dev_dataset = load_from_disk("dev_dataset", keep_in_memory=True)
            test_dataset = load_from_disk("test_dataset", keep_in_memory=True)
        else:

            from torchvision.transforms import (CenterCrop, 
                                                Compose, 
                                                Normalize, 
                                                RandomHorizontalFlip,
                                                RandomResizedCrop, 
                                                RandomRotation,
                                                Resize, 
                                                ToTensor)

            image_mean, image_std = self.processor.image_mean, self.processor.image_std
            size = self.processor.size["height"]

            normalize = Normalize(mean=image_mean, std=image_std)
            _train_transforms = Compose(
                    [
                        RandomResizedCrop(size),
                        RandomHorizontalFlip(),
                        RandomRotation(degrees=(0, 180)),
                        ToTensor(),
                        normalize,
                    ]
                )

            _val_transforms = Compose(
                    [
                        Resize((size, size)),
                        CenterCrop(size),
                        ToTensor(),
                        normalize,
                    ]
                )
            
            def process_train(examples):
                texts = [t for t in examples['text']]
                text_inputs = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=tokenizer.model_max_length)

                examples['input_ids'] = text_inputs['input_ids']
                examples['attention_mask'] = text_inputs['attention_mask']
                examples['pixel_values'] = [_train_transforms(Image.open("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/"+ip).convert("RGB")) for ip in examples['image_path']]
                return examples
            
            def process_val(examples):
                texts = [t for t in examples['text']]
                text_inputs = self.tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=tokenizer.model_max_length)

                examples['input_ids'] = text_inputs['input_ids']
                examples['attention_mask'] = text_inputs['attention_mask']
                examples['pixel_values'] = [_val_transforms(Image.open("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/"+ip).convert("RGB")) for ip in examples['image_path']]  
                return examples
            
            train_dataset = Dataset.from_pandas(self.train)
            train_dataset = train_dataset.map(process_train, batched=True, batch_size=32,
                                            remove_columns=['text', 'image_path'])

            dev_dataset = Dataset.from_pandas(self.dev)
            dev_dataset = dev_dataset.map(process_val, batched=True, batch_size=32,
                                        remove_columns=['text', 'image_path'])

            test_dataset = Dataset.from_pandas(self.test)
            test_dataset = test_dataset.map(process_val, batched=True, batch_size=32,
                                            remove_columns=['text', 'image_path'])

            train_dataset.save_to_disk("train_dataset")
            dev_dataset.save_to_disk("dev_dataset")
            test_dataset.save_to_disk("test_dataset")

        return train_dataset, dev_dataset, test_dataset
    
class MultimodalModel(nn.Module):
    def __init__(self,
                 text_model,
                 image_model,
                 loss_fct=torch.nn.CrossEntropyLoss(),
                 ):
        
        super(MultimodalModel, self).__init__()

        self.image_model = image_model
        image_hidden_size = self.image_model.config.hidden_size
        image_dropout_prob = self.image_model.config.hidden_dropout_prob

        self.text_model = text_model
        text_hidden_size = self.text_model.config.hidden_size
        text_dropout_prob = self.text_model.config.hidden_dropout_prob

        # Roberta classification head
        self.image_classifier = nn.Sequential(nn.Dropout(image_dropout_prob),
                                        nn.Linear(image_hidden_size, image_hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(image_dropout_prob),
                                        nn.Linear(image_hidden_size, 2),
                                        )

        self.text_classifier = nn.Sequential(nn.Dropout(text_dropout_prob),
                                        nn.Linear(text_hidden_size, text_hidden_size),
                                        nn.Tanh(),
                                        nn.Dropout(text_dropout_prob),
                                        nn.Linear(text_hidden_size, 2),
                                        )
        # Loss function
        self.loss_fct = loss_fct

    def forward(self, 
                input_ids=None,
                pixel_values=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                attention_mask=None,
                token_type_ids=None, 
                position_ids=None, 
                head_mask=None,
                labels=None
                ):
        
        image_outputs = self.image_model(pixel_values=pixel_values,
                                         head_mask=head_mask,
                                         output_attentions=output_attentions,
                                         output_hidden_states=output_hidden_states,
                                         return_dict=return_dict,
                                         )

        text_outputs = self.text_model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       position_ids=position_ids,
                                       head_mask=head_mask,
                                       )

        # Features
        image_features = image_outputs.last_hidden_state
        text_features = text_outputs.last_hidden_state

        # fixed-sized contextual information representation that summarizes the entire input sequence
        image_representation = image_features[:, 0, :] 
        text_representation = text_features[:, 0, :]  
    
        # Classification
        image_logits = self.image_classifier(image_representation) 
        text_logits = self.text_classifier(text_representation)

        logits = 1/2 * (image_logits + text_logits)
        
        loss = self.loss_fct(logits.view(-1, 2), labels.view(-1).long())

        return SequenceClassifierOutput(loss=loss, 
                                        logits=logits, 
                                        )

class GELU(nn.Module):
    def forward(self, input_):
        output = gelu(input_)
        return output

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

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

if __name__ == '__main__':
    for seed in [0,1,2,3,4]:
        run = wandb.init(project="Test_Clef", name="multimodal_model")

        text_model_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/twitter-roberta-large-2022-154m"
        wandb.config['text_model_id_or_path'] = text_model_id_or_path

        tokenizer_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/twitter-roberta-large-2022-154m"
        wandb.config['tokenizer_id_or_path'] = tokenizer_id_or_path 

        image_model_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/dino-vitb16"
        wandb.config['image_model_id_or_path'] = image_model_id_or_path

        processor_id_or_path = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/dino-vitb16"
        wandb.config['processor_id_or_path'] = processor_id_or_path 

        tokenizer_max_len = 128
        wandb.config['tokenizer_max_len'] = tokenizer_max_len

        epochs = 20
        wandb.config['epochs'] = epochs

        wandb.config['seed'] = seed

        dataloader_config = {'per_device_train_batch_size': 16,
                             'per_device_eval_batch_size': 64} 
        wandb.config.update(dataloader_config)

        preprocessing_config = {'lowercase': True,
                                'normalize': True,
                                'urls': False,
                                'user_handles': '@USER',
                                'emojis': 'demojize'}
        wandb.config.update(preprocessing_config)

        learning_rate = 2e-5
        wandb.config['learning_rate'] = learning_rate

        num_labels = 2
        problem_type = "single_label_classification"

        wandb.config['num_labels'] = num_labels
        wandb.config['problem_type'] = problem_type

        freezed_until = ""
        wandb.config['freezed_until'] = freezed_until

        print('load models, tokenizer and processor')
        text_model = AutoModel.from_pretrained(text_model_id_or_path)
        tokenizer_config = {'pretrained_model_name_or_path': tokenizer_id_or_path,
                            'max_len': tokenizer_max_len}
        tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)

        image_model = ViTModel.from_pretrained(image_model_id_or_path)
        processor = ViTImageProcessor.from_pretrained(processor_id_or_path)

        model = MultimodalModel(text_model, image_model)

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
        dl = TweetImageDataLoader(preprocessing_config, processor, tokenizer, seed) 

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
        data.to_csv(f'/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/output/clef/{run.name}_preds.tsv', index=False, sep='\t')
        metrics = compute_overall_metrics(data)
        wandb.log(metrics)
        wandb.finish()