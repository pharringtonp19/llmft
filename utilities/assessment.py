def find_first_true_or_false(text):
    # This regular expression pattern looks for 'Yes' or 'No' (case insensitive)
    pattern = r'\b(True|False)\b'
    
    # Search the text for the pattern
    match = re.search(pattern, text, re.IGNORECASE)
    
    # If a match is found, return the matched text, otherwise return None
    return match.group(0) if match else None
