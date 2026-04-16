import nltk

def split_with_nltk(text: str, max_chunk_size: int=512):
    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(text)

    if not sentences:
        return []


    chunks: list[str] = []
    current_chunk = sentences[0]

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence

    chunks.append(current_chunk)
    return chunks


def clean_with_prefixes(answer: str, prefixes: list[str]) -> str:
    for prefix in prefixes:
        # Check if the prefix exists in the answer
        if prefix in answer:
            # split(prefix, 1) splits only on the first occurrence
            # [-1] takes the part AFTER the prefix
            cleaned = answer.split(prefix, 1)[-1]
            
            # Clean up resulting spaces or punctuation (e.g., " : London")
            return cleaned.strip(" :.,")
            
    # If no prefix matches, return the original (normalized)
    return answer
