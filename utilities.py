def get_special_tokens(tokenizer):
    cls = [tokenizer.cls_token, tokenizer.cls_token_id]
    sep = [tokenizer.sep_token, tokenizer.sep_token_id]
    pad = [tokenizer.pad_token, tokenizer.pad_token_id]
    bos = [tokenizer.bos_token, tokenizer.bos_token_id]
    print(cls, sep, pad, bos)

def get_decoding(tokenizer, model_inputs):
    return tokenizer.batch_decode(model_inputs.input_ids)
    
