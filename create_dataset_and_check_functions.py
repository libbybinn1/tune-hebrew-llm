import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import string

# Load translation models
HEB_TO_ENG_MODEL_NAME = 'Helsinki-NLP/opus-mt-tc-big-he-en'
ENG_TO_HEB_MODEL_NAME = 'Helsinki-NLP/opus-mt-en-he'

heb_to_eng_tokenizer = MarianTokenizer.from_pretrained(HEB_TO_ENG_MODEL_NAME)
heb_to_eng_model = MarianMTModel.from_pretrained(HEB_TO_ENG_MODEL_NAME)

eng_to_heb_tokenizer = MarianTokenizer.from_pretrained(ENG_TO_HEB_MODEL_NAME)
eng_to_heb_model = MarianMTModel.from_pretrained(ENG_TO_HEB_MODEL_NAME)

# Utility Functions
def preprocess_text(text):
    """Removes punctuation from text."""
    return text.translate(str.maketrans("", "", string.punctuation))

def read_sentences(input_file):
    """Reads sentences from a file."""
    with open(input_file, "r", encoding="utf-8") as infile:
        return infile.read().strip().splitlines()

def translate(text, tokenizer, model, num_return_sequences=1, max_length=100):
    """Translates text using the specified tokenizer and model."""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated_tokens = model.generate(**inputs, num_return_sequences=num_return_sequences, num_beams=3, max_length=max_length)
        return [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    except Exception as e:
        print(f"Translation error: {e}")
        return [""]

# Check Functions
def check_using_input_and_label(eng_input, eng_label, heb_label):
    """
    Checks if translating an English input and label back to Hebrew matches the Hebrew label.
    """
    eng_sentence = preprocess_text(eng_input) + " " + preprocess_text(eng_label)
    back_to_hebrew = translate(eng_sentence, eng_to_heb_tokenizer, eng_to_heb_model)[0]

    return preprocess_text(back_to_hebrew.split()[-1]) == preprocess_text(heb_label), back_to_hebrew

def check_using_input_and_label_first_tran_label(eng_input, eng_label, heb_label, max_iterations=5):
    """
    Iteratively translates English labels to Hebrew until it matches the Hebrew label or the max_iterations limit is reached.
    """
    label_back_to_hebrew = translate(eng_label, eng_to_heb_tokenizer, eng_to_heb_model)[0]
    iteration = 0

    while len(label_back_to_hebrew.split()) > 1 and iteration < max_iterations:
        eng_label = " ".join(eng_label.split()[:-1])  # Remove last word
        label_back_to_hebrew = translate(eng_label, eng_to_heb_tokenizer, eng_to_heb_model)[0]
        iteration += 1

    return check_using_input_and_label(eng_input, eng_label, heb_label)

# Dataset Processing
def process_hebrew_translation_dataset(input_file, output_csv_file, buffer_size=50):
    """
    Processes a dataset file and creates a CSV with matches and mismatches.
    """
    sentences = read_sentences(input_file)
    buffer = []

    for idx, sentence in enumerate(sentences):
        result = process_sentence(sentence)
        if result:
            buffer.append(result)

        if (idx + 1) % buffer_size == 0 or (idx + 1) == len(sentences):
            write_to_csv(buffer, output_csv_file, idx < buffer_size)
            buffer.clear()


def process_sentence(sentence):
    """Processes a single sentence and validates translations."""
    words = preprocess_text(sentence).split()
    if len(words) < 2:
        return None

    heb_input, heb_label = " ".join(words[:-1]), words[-1]
    eng_input = translate(heb_input, heb_to_eng_tokenizer, heb_to_eng_model, num_return_sequences=1)[0]
    eng_label = translate(heb_label, heb_to_eng_tokenizer, heb_to_eng_model, num_return_sequences=1)[0]

    match_sentence = check_using_input_and_label(eng_input, eng_label, heb_label)[0]

    return {
        'eng_input': eng_input,
        'eng_label': eng_label,
        'heb_input': heb_input,
        'heb_label': heb_label,
        'match': match_sentence
    }

def write_to_csv(buffer, csv_file, is_first_batch):
    """Writes buffered results to a CSV file."""
    mode = 'w' if is_first_batch else 'a'
    header = is_first_batch
    pd.DataFrame(buffer).to_csv(csv_file, mode=mode, index=False, header=header, encoding='utf-8')
