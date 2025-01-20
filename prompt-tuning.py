import re
import torch
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from datasets import Dataset

# Constants
MODEL_NAME = "meta-llama/Llama-3.2-1B"  # Replace with the exact model name
OUTPUT_MODEL_DIR = "./llama_model_16kk"
DATASET_PATH = "training_data_16k.csv"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# Utility function to clean text by removing non-alphanumeric characters
def clean_text(text: str) -> str:
    """
    Clean the input text by removing non-alphanumeric characters.
    :param text: Input string to be cleaned.
    :return: Cleaned string.
    """
    return re.sub(r'[^A-Za-z0-9\s]', '', text).strip()

# Tokenization function for dataset processing
def tokenize_function(examples):
    """
    Tokenize the dataset's inputs and labels with the provided tokenizer.
    :param examples: Dictionary containing "eng_input" and "eng_label".
    :return: Tokenized dataset with input and label IDs.
    """
    inputs = [clean_text(str(input_)) if input_ else "" for input_ in examples["eng_input"]]
    labels = [clean_text(str(label)) if label_ else "" for label_ in examples["eng_label"]]

    tokenized_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=512)
    tokenized_labels = tokenizer(labels, truncation=True, padding="max_length", max_length=512)

    tokenized_inputs["labels"] = tokenized_labels["input_ids"]
    return tokenized_inputs

# Load and preprocess dataset
df = pd.read_csv(DATASET_PATH)
dataset = Dataset.from_pandas(df)
dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Configure prompt tuning
tuning_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text=(
        "Complete the following sentence and make sure it translates well into Hebrew"
    ),
    num_virtual_tokens=32,  # Optimal starting point
    tokenizer_name_or_path=MODEL_NAME
)

# Define the PEFT model
peft_model = get_peft_model(model, tuning_config)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./llama_word_gen_model",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=0.001,  # Lower learning rate for stability
    num_train_epochs=10,
    save_steps=50,
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    optim="adamw_torch"
)

# Create Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
peft_model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print(f"Model and tokenizer saved to {OUTPUT_MODEL_DIR}")
