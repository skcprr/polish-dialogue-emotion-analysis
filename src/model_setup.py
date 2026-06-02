import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def tokenizer_setup(model_name):
    
    print(f'Downloading tokenizer for: {model_name}')
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return tokenizer



def model_setup(model_name, num_labels):

    print(f'Downloading base model for: {model_name}')

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type='multi_label_classification'
    )

    return model


def test_pipeline(tokenizer, model, sample_text):

    print('Testing pipeline...')    

    inputs = tokenizer(
        sample_text,
        return_tensors = 'pt',
        padding = True,
        truncation = True,
        max_length = 128
    )

    print('Tokenization results:')
    print(f'Original text: {sample_text}')
    print(f'Tokens: {inputs['input_ids'][0][:15].tolist()}')
    print(f'Attention Mask: {inputs['attention_mask'][0][:15].tolist()}')

    # Forward pass
    # Without calculating gradient
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    print('Model results:')
    print(f'Output shape (Batch_size, Num_labels): {logits.shape}')
    print(f'Logits for 8 classes: \n{logits[0].numpy()}')

    probabilities = torch.sigmoid(logits)[0]

    print(f'\nProbability (random before training): \n{probabilities.numpy()}')




def main():
    MODEL_NAME = 'allegro/herbert-base-cased'
    NUM_LABELS = 8
    sample_text = 'Masakra, znowu opóźnienie pociągu! Jestem niesamowicie wściekły...'


    tokenizer = tokenizer_setup(MODEL_NAME)
    model = model_setup(MODEL_NAME, NUM_LABELS)

    test_pipeline(tokenizer, model, sample_text)


if __name__ == '__main__':
    main()