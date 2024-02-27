#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

    
    
class Detoxify:

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, class_names: list[str], device: torch.device | str = "cpu"):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.device = device
        self.model.to(self.device)

    @classmethod
    def from_pretrained(cls, path: Path, device: torch.device | str = "cpu"):
        model = AutoModelForSequenceClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        class_names = json.loads(path.read_text())
        return cls(model, tokenizer, class_names, device=device)

    @torch.no_grad()
    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        out = self.model(**inputs)[0]
        scores = torch.sigmoid(out).cpu().detach().numpy()
        results = {}
        for i, cla in enumerate(self.class_names):
            results[cla] = (
                scores[0][i] if isinstance(text, str) else [scores[ex_i][i].tolist() for ex_i in range(len(scores))]
            )
        return results
