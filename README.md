Negative Text Generation with Prompt Tuning on LLAMA 3.2 This project involves fine-tuning the OPT-350M model using prompt tuning techniques to generate negative text completions based on given prompts. The model is fine-tuned using a custom dataset and Parameter-Efficient Fine-Tuning (PEFT) methods, specifically using prompt tuning. The final goal is to create a model that responds negatively to various prompts, like "The service was" or "The product quality was." Model We are using the facebook/opt-350m model from Hugging Face, which is a pre-trained language model suitable for causal language modeling tasks.

Prompt Tuning Prompt tuning is applied to the model using peft. In this project, we use soft prompts (virtual tokens) to influence the model's behavior without fully fine-tuning it. This approach significantly reduces computational costs and the number of parameters to train.

Training Script The main script for training and inference is opt350.py. Here's a breakdown of its key components:

Model and Tokenizer Loading: Loads the OPT-350M model and tokenizer from Hugging Face's pre-trained models.

Text Generation Function: A utility to generate text based on a prompt using the fine-tuned model.

Dataset Loading and Tokenization: The dataset is loaded from a CSV file, and the prompts and completions are tokenized with dynamic padding and truncation.

Prompt Tuning Configuration: The script uses the PromptTuningConfig from peft to set up the prompt tuning. It defines the number of virtual tokens, task type, and initialization method.

Trainer Setup: The Hugging Face Trainer is used for training the model, with customized training arguments like batch size, learning rate, and mixed precision training (FP16) if a GPU is available.

Training: The model is fine-tuned on the custom dataset.

Inference: After training, the model generates text based on a list of test prompts and prints the output.

