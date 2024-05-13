#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   transforms_pipeline.py
@Time    :   2024-05-13 15:00
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def main():
    # load pretaied model tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # encode text
    text = "with the great power comes great "
    indexed_tokens = tokenizer.encode(text)

    # convert indexed tokens in a tensor
    tokens_tensor = torch.tensor([indexed_tokens])
    print(tokens_tensor)

    # load pretraied model
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # set the model in evaluation mode to deactivate the DropOut modules
    model.eval()

    # predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    # get the predicted next sub-word
    predicted_index = torch.argmax(predictions[0, -1, :]).item()
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    print(predicted_text)


if __name__ == '__main__':
    main()
