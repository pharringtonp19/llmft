def formatting_prompts_func(example):
    # Create a list to store the formatted texts for each item in the example
    formatted_texts = []

    # Iterate through each example in the batch
    for text, raw_label in zip(example[column], example['raw_label']):
        # Format each example as a prompt-response pair
        formatted_text = f"{text} {raw_label}"
        formatted_texts.append(formatted_text)

    # Return the list of formatted texts
    return formatted_texts

response_template_with_context = "\n\n    Answer:"  # We added context here: "\n". This is enough for this tokenizer
response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False) # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
