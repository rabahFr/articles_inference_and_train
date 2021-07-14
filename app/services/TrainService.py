import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from app.model.ArticleDataset import ArticleDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from app.services.BertService import load_model_bert


class TrainService:

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, data_path):
        self.data_path = data_path
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

    def fit(self, num_train_epochs, warmup_steps, weight_deccay):
        print("fit...")
        model = load_model_bert()
        data = self.load_data()
        train_data, test_data = self.prepare_data(data, 0.1)
        trainer = self.prepare_trainer(model, num_train_epochs, warmup_steps, weight_deccay, train_data, test_data)
        print("starting the training")
        trainer.train()
        print("start evaluation")
        trainer.evaluate()
        print("saving model locally")
        trainer.save_model("app/resources/bert_classification_train/")
        print("training pipeline done")


    def load_data(self):
        print("Start loading data")
        data_raw = pd.read_csv(self.data_path)

        data_raw['sentence'] = data_raw.sentence.str.lower()

        positive_sentences = data_raw[(data_raw.positive=="x") | (data_raw.slightly_positive=="x")]
        negative_sentences = data_raw[(data_raw.negative=="x") | (data_raw.slightly_negative=="x")]
        data_pos = pd.DataFrame(data = positive_sentences, columns = {"sentence"})
        data_pos['label'] = 1
        data_neg = pd.DataFrame(data = negative_sentences, columns = {"sentence"})
        data_neg['label'] = 0
        print("data loading done")
        return pd.concat([data_pos, data_neg])


    def prepare_data(self, data, test_size):
        print("preparing data for training")
        X_train, X_test, y_train, y_test = train_test_split(data["sentence"], data["label"], test_size=test_size)

        train_encoding = self.tokenizer(list(X_train), padding=True, truncation=True)
        train_labels = list(y_train)

        test_encoding = self.tokenizer(list(X_test), padding=True, truncation=True)
        test_labels = list(y_test)

        print("data loading done")
        return ArticleDataset(train_encoding, train_labels), ArticleDataset(test_encoding, test_labels)

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def prepare_trainer(self, model, num_train_epochs, warmup_steps, weight_decay, train_data, test_data):
        print("preparing the trainer")
        training_args = TrainingArguments(
            output_dir='results_bert/',          # output directory
            num_train_epochs=num_train_epochs,              # total # of training epochs
            per_device_train_batch_size=50,  # batch size per device during training
            per_device_eval_batch_size=50,   # batch size for evaluation
            warmup_steps=warmup_steps,                # number of warmup steps for learning rate scheduler
            weight_decay=weight_decay,               # strength of weight decay
            logging_dir='logs_bert/',            # directory for storing logs
        )

        print("trainer prepared")
        return Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_data,         # training dataset
            eval_dataset=test_data,            # evaluation dataset
            compute_metrics=self.compute_metrics,
        )
