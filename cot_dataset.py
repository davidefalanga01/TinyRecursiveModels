import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

class GSM8KDataset(IterableDataset):
    def __init__(
        self, 
        tokenizer_name: str = "gpt2", 
        seq_len: int = 512, # GSM8K richiede un po' più di spazio per i passaggi matematici
        batch_size: int = 32,
        split: str = "train"
    ):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        # 1. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id

        # 2. Carichiamo GSM8K (scarica automaticamente da HF)
        self.ds = load_dataset("gsm8k", "main", split=split)
        
        # 3. Prepariamo i dati: "Question: ... Answer: ..."
        self.examples = []

        for ex in self.ds:
            prompt = f"Question: {ex['question']}\nAnswer:"
            completion = f" {ex['answer']}{self.tokenizer.eos_token}"

            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids

            if len(input_ids) <= self.seq_len:
                pad_len = self.seq_len - len(input_ids)
                input_ids += [self.pad_id] * pad_len
                labels += [-100] * pad_len

                self.examples.append((input_ids, labels))
        
        print(f"Dataset GSM8K pronto: {len(self.examples)} esempi validi.")

    def __iter__(self):
        while True:
            # Selezioniamo indici casuali
            ix = np.random.randint(0, len(self.examples), self.batch_size)
            
            batch_x, batch_y = zip(*[self.examples[i] for i in ix])

            batch = {
                "inputs": torch.tensor(batch_x, dtype=torch.long),
                "labels": torch.tensor(batch_y, dtype=torch.long),
                "puzzle_identifiers": torch.zeros(self.batch_size, dtype=torch.long)
            }

            yield "gsm8k_train", batch, self.batch_size
