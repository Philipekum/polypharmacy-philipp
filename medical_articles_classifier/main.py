import argparse

import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


NUM_LABELS = 4
CATEGORY_NAMES = ['Кардиология', 'Неврология', 'Офтальмология', 'Хирургия']


class MedicalClassifier(nn.Module):
    def __init__(self, num_labels):
        super(MedicalClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def read_args() -> None:
    global FILEPATH, THRESHOLD

    parser = argparse.ArgumentParser()
    
    parser.add_argument('filepath', type=str, help='Путь к файлу')

    parser.add_argument('-t', type=float, help='Порог учета значений', required=False)
    
    args = parser.parse_args()

    global FILEPATH
    global THRESHOLD
    
    FILEPATH = args.filepath
    THRESHOLD = args.t if args.t is not None else 0.2

    if not FILEPATH:
        print('Передайте путь к файлу аргументом')
        exit(1)


def predict(text: str) -> dict:
    model.eval()
    encoding = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
    input_ids, attention_mask = encoding['input_ids'].to(device), encoding['attention_mask'].to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask)
        probabilities = nn.functional.softmax(output, dim=1).cpu().numpy()[0]

    result = {CATEGORY_NAMES[i]: float(probabilities[i]) for i in range(len(CATEGORY_NAMES))}

    return result


def init_model() -> None:
    global device
    global tokenizer
    global model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    model = MedicalClassifier(num_labels=NUM_LABELS)

    model.load_state_dict(torch.load('state_dict_improved.pt', map_location=device))
    model.to(device)
    model.eval()


def read_file() -> str:
    try:
        with open(FILEPATH, 'r', encoding='utf-8') as f:
            text = f.read()

        if len(text) == 0:
            print(f'Файл {FILEPATH} пустой')
            exit(1)

        return text
    
    except FileNotFoundError:
        print(f'Файл {FILEPATH} не найден')
        exit(1)
    
    except Exception:
        print(f'Ошибка при чтении файла {FILEPATH}')
        exit(1)


def format_result(data: dict) -> str:
    filtered_specialties = [name for name, value in data.items() if value > THRESHOLD]

    filtered_specialties = sorted(
        ((name, value) for name, value in data.items() if value > THRESHOLD),
        key=lambda item: item[1],
        reverse=True
    )

    filtered_specialties_names = [f'{name} ({value * 100:.2f}%)' for name, value in filtered_specialties] 

    return ', '.join(filtered_specialties_names)


def main():
    read_args()

    text = read_file()

    init_model()

    data = predict(text)

    result = format_result(data)
  
    print(result)


if __name__ == '__main__':
    main()
