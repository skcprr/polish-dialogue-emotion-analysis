import pandas as pd
from datasets import load_dataset
from pprint import pprint

def load_data():
    dataset = load_dataset("clarin-pl/twitteremo")

    print(dataset)


if __name__ == '__main__':
    load_data()
