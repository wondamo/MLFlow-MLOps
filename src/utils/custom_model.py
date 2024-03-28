import json
import spacy
import mlflow
import cloudpickle
import tensorflow as tf
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TensorWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        # Load the TensorFlow model
        self.model = tf.keras.models.load_model(context.artifacts["tensorflow_model"])
        # Load the spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        # Load the tokenizer
        tokenizer_path = context.artifacts["tokenizer"]
        with open(tokenizer_path, 'r') as json_file:
            tokenizer_json = json_file.read()
        # Convert the json to a dictionary
        tokenizer_dict = json.loads(tokenizer_json)
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_dict)

    def preprocess(self, string):
        doc = self.nlp(string)
        lemma = [token.lemma_ for token in doc if token.lemma_.isalpha() or token.lemma_ not in STOP_WORDS]
        return ' '.join(lemma)
    
    def predict(self, context, input):
        model_input = pd.Series(input)
        print("Model Input")
        preprocessed_input = model_input.apply(self.preprocess)
        print("Processed Input")
        tokenized_input = self.tokenizer.texts_to_sequences(preprocessed_input)
        print("Sequences")
        sequences = pad_sequences(tokenized_input, padding='post', maxlen=200)

        return self.model.predict(sequences)
    

conda_env = {
    "channels": ["default", "conda-forge"],
    "dependencies": [
        "python=3.12",
        "pip",
        {
            "pip": [
                "spacy==3.7.4",
                "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz",
                f"mlflow=={mlflow.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
                f"tensorflow=={tf.__version__}",
                f"pandas=={pd.__version__}"
            ]
        }
    ],
    "name": "tf_env"
}