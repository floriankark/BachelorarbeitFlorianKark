from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import pandas as pd
from tqdm import tqdm

print("Loading model...")
model = InstructBlipForConditionalGeneration.from_pretrained("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/instructblip-vicuna-7b")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

class MyDataset(Dataset):
    def __init__(self, path):
        with open(path, "r") as fp:
            lines = [json.loads(a) for a in fp.readlines()]

        self.ds = pd.DataFrame(lines)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ip = self.ds['image_path'].iloc[i]
        img = Image.open("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/" + ip).convert("RGB")
        inputs = processor(images=img, text="Write a description for the image.", return_tensors="pt")
        return inputs

dev= "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/CT23_1A_checkworthy_multimodal_english_dev.jsonl"
dev_test = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/CT23_1A_checkworthy_multimodal_english_dev_test.jsonl"
train = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/CT23_1A_checkworthy_multimodal_english_train.jsonl"
test = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/CT23_1A_checkworthy_multimodal_english_test.jsonl"
gold = "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/CT23_1A_checkworthy_multimodal_english_test_gold.jsonl"

print("Loading datasets...")
train_ds = MyDataset(train)
test_ds = MyDataset(test)
dev_ds = MyDataset(dev)
dev_test_ds = MyDataset(dev_test)
gold_ds = MyDataset(gold)

def generate(ds):
    batch_size = 4
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    generated_texts = []

    for inputs in tqdm(dl):
        pixel_values = inputs['pixel_values'].resize(batch_size,3,224,224)
        input_ids = inputs['input_ids'].resize(batch_size, inputs['input_ids'].shape[-1])
        attention_mask = inputs['attention_mask'].resize(batch_size, inputs['input_ids'].shape[-1])

        outputs = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=5,
            max_length=200, # because 95% coverage with ~300 text+ocr and +200 is ca. 512 (max limit)
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            temperature=0.01,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
        generated_texts.extend(generated_text)

    return generated_texts

print("Generating descriptions...")
train_texts = generate(train_ds)
test_texts = generate(test_ds)
dev_texts = generate(dev_ds)
dev_test_texts = generate(dev_test_ds)
gold_texts = generate(gold_ds)

print("Saving descriptions...")
train_df = train_ds.ds
train_df['description'] = train_texts
train_df.to_pickle("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/train_description.pkl")

test_df = test_ds.ds
test_df['description'] = test_texts
test_df.to_pickle("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/test_description.pkl")

dev_df = dev_ds.ds
dev_df['description'] = dev_texts
dev_df.to_pickle("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/dev_description.pkl")

dev_test_df = dev_test_ds.ds
dev_test_df['description'] = dev_test_texts
dev_test_df.to_pickle("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/dev_test_description.pkl")

gold_df = gold_ds.ds
gold_df['description'] = gold_texts
gold_df.to_pickle("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/gold_description.pkl")

print("Done!")