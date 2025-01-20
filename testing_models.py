import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from create_dataset_and_check_functions import check_using_input_and_label

# Constants
MODEL_NAME = "meta-llama/Llama-3.2-1B"
ADJUSTED_MODEL_PATH = "./llama_model_16k"
DATASET_PATH = "testing_4000.csv"
OUTPUT_PATH = "test_results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_ADJUSTED_MODEL = False  # Flag to toggle between base and tuned models
CHECK_FUNCTION = check_using_input_and_label

# Load Tokenizer and Model
def load_model_and_tokenizer(model_name, adjusted_model_path=None, use_adjusted=False):
    """Loads a tokenizer and model, with optional tuning adjustments."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if use_adjusted and adjusted_model_path:
        model = PeftModel.from_pretrained(model, adjusted_model_path)

    model.to(DEVICE)
    model.eval()
    return model, tokenizer


# Process Rows
def process_row(row, model, tokenizer):
    """Processes a single row from the dataset."""
    eng_input = row["eng_input"]
    eng_label = row["eng_label"]
    heb_label = row["heb_label"]

    # Generate output and check match
    model_gen = generate_output(model, tokenizer, eng_input)
    is_match = CHECK_FUNCTION(eng_input, model_gen, heb_label)[0]

    return {
        "eng_input": eng_input,
        "eng_label": eng_label,
        "heb_label": heb_label,
        "model_gen": model_gen,
        "is_match": is_match
    }


def generate_output(model, tokenizer, text):
    """Generates text output from the model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(DEVICE)
    outputs = model.generate(inputs["input_ids"], max_new_tokens=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Main Script
def main():
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, ADJUSTED_MODEL_PATH, USE_ADJUSTED_MODEL)
    df = pd.read_csv(DATASET_PATH)
    results = [process_row(row, model, tokenizer) for _, row in df.iterrows()]

    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
