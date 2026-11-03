"""
@Project :   Study
@File    :   llm_base.py
@Time    :   2026-11-03 10:30
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def pipe_line():
    # Initialize the text generation pipeline
    generator = pipeline('text-generation', model='Qwen/Qwen2.5-0.5B')
    # Define your prompt
    prompt = "Write a short introduction about the US capital city."
    # Generate text
    generated_text = generator(prompt, max_length=50, num_return_sequences=1)
    # Print the generated text
    print(generated_text[0]['generated_text'])


def token_by_token_generation():
    # (1) Load tokenizer and model
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
        ).to("cuda" if torch.cuda.is_available() else "cpu")
    )
    # (2) Define the input prompt
    prompt = """The history of human … … … How might the next wave of
    communication tools shape our relationships, societies, and sense
    of identity?"""
    # (3) Convert (tokenize) prompt to the input format that model understands
    max_new_tokens = 100
    # tokenize the input prompt for the first output token
    # PS: prompt is the initial input sequence for LLM generation
    idx = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    start_time = total_time = time.time()
    times = []
    # (4) Main generation loop - generate tokens one by one
    for _ in range(max_new_tokens):
        # (A) Set the current context for generation
        idx_cond = idx
        with torch.no_grad():
            # (B) Generate predictions (token candidates) for next token
            outputs = model(idx_cond)
            # Get the logits (raw prediction scores) for each token prediction
            logits = outputs.logits
        # (C) Select next token from the predictions generated in step (B)
        logits = logits[:, -1, :]  # Select only the logits for the last token
        # Convert logits to probabilities using softmax
        probas = torch.softmax(logits, dim=-1)
        # Sample the next token from the probability distribution of the predicted tokens from step (B)
        idx_next = torch.multinomial(probas, num_samples=1)
        print("Next Token is:", tokenizer.decode(idx_next[0]))
        time_cost = time.time() - start_time
        times.append(time_cost)
        # (D) Append the new token to the input sequence
        idx = torch.cat((idx, idx_next), dim=1)
        # (E) Check if end-of-sequence token was generated
        if idx_next.item() == tokenizer.eos_token_id:
            print("\n[Generation completed - EOS token reached]")
            break
    # (5) Decode the entire generated sequence
    generated_text = tokenizer.decode(idx[0], skip_special_tokens=True)
    print(generated_text)


def token_by_cache():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_cache=True) \
        .to("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(
        "cuda" if torch.cuda.is_available() else "cpu")

    prompt = """The history of human communication is a story of innovation. From ancient cave paintings and spoken language to the invention of writing systems, humans have constantly developed new methods to express ideas and share knowledge. The printing press revolutionized the spread of information, enabling books to be produced and distributed at an unprecedented scale. Centuries later, the invention of the telegraph, radio, and television further transformed how we connect with one another. But perhaps no advancement has reshaped communication more profoundly than the internet.
    Today, digital platforms allow billions of people to share messages, media, and experiences in real time. Social media, messaging apps, and video conferencing have broken down geographical barriers and created new ways of building communities. At the same time, these technologies raise important questions about privacy, information overload, and the nature of human interaction.
    Looking ahead, emerging technologies such as virtual reality, brain-computer interfaces, and artificial intelligence promise to once again redefine how we communicate. As we reflect on this history and anticipate the future, one question arises:

    How might the next wave of communication tools shape our relationships, societies, and sense of identity?"""

    num_interations = 100
    times_with_cache = []

    first_token_generated = False
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    idx = input_ids
    start_time = total_time = time.time()

    # (1) Define Key/Value Cache for faster generation
    past_key_values = None

    for _ in range(num_interations):
        print("input_ids size: " + str(input_ids.size()))
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            past_key_values=past_key_values,  # (2) Use KV-cache from previous iteration
                            use_cache=True,  # (2) Enable KV caching
                            max_new_tokens=100,
                            min_new_tokens=100)

            logits = outputs.logits
            # (3) Update KV Cache
            past_key_values = outputs.past_key_values
            # torch.cuda.synchronize()

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        generated_token_id = torch.multinomial(probas, num_samples=1)  # Sample instead of argmax

        # (4) Update input_ids with only the new token (using KV-cache)
        input_ids = generated_token_id  # Note: Not concatenating with previous tokens due to KV-cache

        print("Next token:", tokenizer.decode(generated_token_id[0], skip_special_tokens=True))
        idx = torch.cat((idx, generated_token_id), dim=1)

        time_cost = time.time() - start_time
        times_with_cache.append(time_cost)
        if not first_token_generated:
            print(f"Time take for first token: {time_cost:.4f} seconds")
            first_token_generated = True
        else:
            print(f"Time take for next token: {time_cost:.4f} seconds")
        start_time = time.time()

        if generated_token_id.item() == tokenizer.eos_token_id:
            print("\n[Generation completed - EOS token reached]")
            break

    generated_text = tokenizer.decode(idx[0], skip_special_tokens=True)
    print(f"Total time take for next token: {time.time() - total_time:.4f} seconds")
    print(generated_text)


if __name__ == '__main__':
    # pipe_line()
    # token_by_token_generation()
    token_by_cache()
