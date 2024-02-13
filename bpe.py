from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# token_ids = tokenizer("A man in a blue shirt standing in a garden .").input_ids
# tokenizer.convert_ids_to_tokens(token_ids)
