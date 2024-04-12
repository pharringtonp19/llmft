def cls_sep_pad(tokenizer):
    cls = [tokenizer.cls_token, tokenizer.cls_token_id]
    sep = [tokenizer.sep_token, tokenizer.sep_token_id]
    pad = [tokenizer.pad_token, tokenizer.pad_token_id]
    print(cls, sep, pad)
