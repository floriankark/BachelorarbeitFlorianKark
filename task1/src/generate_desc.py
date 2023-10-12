from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import pandas as pd
from tqdm import tqdm
import os

# Set CUDA_LAUNCH_BLOCKING to 1
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Set TORCH_USE_CUDA_DSA to 1
#os.environ["TORCH_USE_CUDA_DSA"] = "1"

print("Loading model...")
model = InstructBlipForConditionalGeneration.from_pretrained("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/models/instructblip-vicuna-7b")
processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# OverflowError: out of range integral type conversion attempted
# tokenizer.add_special_tokens({"pad_token": "[PAD]"})

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
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
        image = Image.open("/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/" + ip).convert("RGB")
        prompt = "Write a detailed description."
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        return inputs

print("Loading datasets...")
train= "/gpfs/project/flkar101/CheckWorthinessInMultimodalContent/task1/data/CT23_1A_checkworthy_multimodal_english_train.jsonl"
train_ds = MyDataset(train)

# generate text from one single input

"""inputs = train_ds[0]
print(inputs)
print(inputs['pixel_values'].shape)
print(inputs['input_ids'].shape)
print(inputs['attention_mask'].shape)
print(inputs['qformer_input_ids'].shape)
print(inputs['qformer_attention_mask'].shape)

outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1.0,
        )

generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)"""



def generate(ds):
    batch_size = 2
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # print shape of dl
    print("Shape of dl:")
    for inputs in dl:
        print(inputs['pixel_values'].shape)
        print(inputs['input_ids'].shape)
        print(inputs['attention_mask'].shape)
        print(inputs['qformer_input_ids'].shape)
        print(inputs['qformer_attention_mask'].shape)
        break

    generated_texts = []
    #outputs_list = []
    text_counter = 0

    for inputs in tqdm(dl):
        if text_counter >= 4:
            break
        # check if we need to empty the cache
        #print(torch.cuda.memory_allocated())

        inputs['pixel_values'] = inputs['pixel_values'].view(batch_size, 3, 224, 224)
        inputs['input_ids'] = inputs['input_ids'].view(batch_size, -1)
        inputs['attention_mask'] = inputs['attention_mask'].view(batch_size, -1)
        inputs['qformer_input_ids'] = inputs['qformer_input_ids'].view(batch_size, -1)
        inputs['qformer_attention_mask'] = inputs['qformer_attention_mask'].view(batch_size, -1)

        # get all inputs and the shapes
        for k, v in inputs.items():
            print(k, v.shape)

        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1.0,
        )
        #outputs[outputs == 0] = 2 #processor.tokenizer.pad_token
        generated_text = processor.batch_decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True)[0].strip()
        generated_texts.append(generated_text)
        text_counter += len(generated_text)

        #outputs_list.append(outputs)
        #text_counter += len(outputs)

    return generated_texts

print("Generating descriptions...")
train_texts = generate(train_ds)
# print all generated texts
for text in train_texts:
    print(text)
print(len(train_texts))