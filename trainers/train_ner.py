import spacy
from spacy.training import Example, offsets_to_biluo_tags
import json
from spacy.util import minibatch, compounding

# Load the base model
nlp = spacy.load("en_core_web_sm")

# Define the custom NER component
ner = nlp.get_pipe("ner")

# Add the custom labels
for label in ["SKILL"]:
    ner.add_label(label)

# Load your annotated training data
with open("../data/training_data.json", "r") as f:
    training_data = json.load(f)

# Disable other components during training
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.begin_training()
    for itn in range(30):  # Number of training iterations
        losses = {}
        # Create minibatches of the training data
        batches = minibatch(training_data, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            # Extract texts and annotations from the batch
            texts = [example['text'] for example in batch]
            annotations = [{'entities': example['entities']} for example in batch]

            # Create Example objects for training
            examples = [
                Example.from_dict(nlp.make_doc(text), annotation)
                for text, annotation in zip(texts, annotations)
            ]
            nlp.update(examples, drop=0.5, losses=losses)
        print(f"Iteration {itn}: Losses: {losses}")

# Save the trained model
nlp.to_disk("models/custom_ner_model")
