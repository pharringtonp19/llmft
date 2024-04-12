def get_special_tokens(tokenizer):
    cls = [tokenizer.cls_token, tokenizer.cls_token_id]
    sep = [tokenizer.sep_token, tokenizer.sep_token_id]
    pad = [tokenizer.pad_token, tokenizer.pad_token_id]
    bos = [tokenizer.bos_token, tokenizer.bos_token_id]
    print(cls, sep, pad, bos)

def get_decoding(tokenizer, model_inputs):
    return tokenizer.batch_decode(model_inputs.input_ids)




### ---         Importing PyTorch
import torch
import torch.nn.functional as F
print(f"PyTorch Version: {torch.__version__}")

import torch
print(f"Cude is available: {torch.cuda.is_available()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
### ---

### ---         Memory Check
def Memory():
    print("Current memory usage:")
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
### ---

### ---         Print Markdown
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
### ---


    
