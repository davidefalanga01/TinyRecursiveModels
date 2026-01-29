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

        # 2. Carichiamo GSM8K (scarica automaticamente da HF)
        self.ds = load_dataset("gsm8k", "main", split=split)
        
        # 3. Prepariamo i dati: "Question: ... Answer: ..."
        self.examples = []
        for ex in self.ds:
            # Uniamo domanda e risposta in un unico blocco di testo
            text = f"Question: {ex['question']}\nAnswer: {ex['answer']}{self.tokenizer.eos_token}"
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= self.seq_len:
                self.examples.append(tokens)
        
        print(f"Dataset GSM8K pronto: {len(self.examples)} esempi validi.")

    def __iter__(self):
        while True:
            # Selezioniamo indici casuali
            ix = np.random.randint(0, len(self.examples), self.batch_size)
            
            batch_x = []
            for i in ix:
                tokens = self.examples[i]
                # Aggiungiamo padding se la frase è corta
                padding_len = self.seq_len - len(tokens)
                padded_tokens = tokens + [self.tokenizer.pad_token_id] * padding_len
                batch_x.append(torch.tensor(padded_tokens))
            
            x_tensor = torch.stack(batch_x) # [Batch, Seq_Len]
            
            # Formato per il tuo modello
            batch = {
                "inputs": x_tensor,
                "puzzle_identifiers": torch.zeros((self.batch_size,), dtype=torch.long)
            }
            
            yield "gsm8k_train", batch, self.batch_size
