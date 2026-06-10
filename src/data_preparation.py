import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer



def get_prepared_datasets(model_name, max_length):
    
    print("Downloading dataset: clarin-pl/twitteremo...")    
    raw_dataset = load_dataset("clarin-pl/twitteremo")

    full_dataset = raw_dataset['train']


    print('Splitting dataset into train dataset: 80%, validation dt: 10% & test dt: 10%')

    train_dev_split = full_dataset.train_test_split(test_size=0.2, seed=22)
    train_dataset = train_dev_split['train']

    val_test_split = train_dev_split['test'].train_test_split(test_size=0.5, seed=22)
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']   

    label_cols = ['radość', 'smutek', 'zaufanie', 'wstręt', 'strach', 'gniew', 
                  'przeczuwanie', 'zdziwienie', 'pozytywny', 'negatywny', 
                  'neutralny', 'ambiwalentny', 'sarkazm']
    
    print(f'Downloading tokenizer for {model_name}')

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_format(examples):
        tokenized = tokenizer(
            examples['tekst'],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        
        labels_matrix = []

        for i in range(len(examples['tekst'])):
            tweet_labels = [float(examples[col][i]) for col in label_cols]
            labels_matrix.append(tweet_labels)
            
        tokenized["labels"] = labels_matrix
        return tokenized

    train_tokenized = train_dataset.map(tokenize_and_format, batched=True, remove_columns=full_dataset.column_names)
    val_tokenized = val_dataset.map(tokenize_and_format, batched=True, remove_columns=full_dataset.column_names)
    test_tokenized = test_dataset.map(tokenize_and_format, batched=True, remove_columns=full_dataset.column_names)

    print("All datasets are prepared:")
    print(f"Train rows:  {len(train_tokenized)}")
    print(f"Validation rows: {len(val_tokenized)}")
    print(f"Test rows: {len(test_tokenized)}")

    return train_tokenized, val_tokenized, test_tokenized, label_cols


def main():
    MODEL_NAME = "allegro/herbert-base-cased"
    MAX_LENGTH = 128

    train_ds, val_ds, test_ds, labels = get_prepared_datasets(MODEL_NAME, MAX_LENGTH)

if __name__ == '__main__':
    main()