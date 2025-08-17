import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if we're using CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Import Model from Huggingface Transformers library
model_name = "microsoft/DialoGPT-medium"


"""
Tokenizer *tokenizes* text into tokens. Since each model has different
specifications in tokenizing, we use DialoGPT's tokenizer specifically.
"""


tokenizer = AutoTokenizer.from_pretrained(model_name)
"""
AutoModelForCausalLM is used to generate text and have conversation based
on query. There are other model types such as AutoModelForQuestionAnswering,
AutoModelForSequenceClassification
"""
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

"""
eos stands for "end of sequence" which means the sentence or the
conversation is finished.
pad means Padding. It's used to match the length of the sentence.
This is specifically used in DialoGPT (and GPT2) since pad_token
is not given in default.
"""
tokenizer.pad_token = tokenizer.eos_token


def generate_response(prompt, max_length=100):
    inputs = tokenizer(
        # Whenver user adds a query(or prompt), eos_token is added
        prompt + tokenizer.eos_token,
        # return tensor type is set to PyTorch
        return_tensors="pt",
        # match the length of sentences using pad_token <PAD>
        padding=True,
        # If the sentence is too long, truncate it
        truncation=True,
    ).to(device)

    # turn off gradient calculation since we are NOT training the model
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            # This differentiates real tokens and paddings
            attention_mask=inputs.attention_mask,
            # max length of generated text (100)
            max_length=max_length,
            # number of responses to generate
            num_return_sequences=2,
            # add randomness
            do_sample=True,
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[:, inputs.input_ids.shape[-1] :][0],
        # e.g. <PAD>, <EOS>
        skip_special_tokens=True,
    )
    return response


if __name__ == "__main__":
    print("DialoGPT Chat Bot - Type 'quit' to exit")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        response = generate_response(user_input)
        print(f"Bot: {response}")
        print()
