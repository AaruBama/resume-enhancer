import spacy
from spacy.training import Example
import json
from spacy.util import minibatch, compounding

# Load the base model
nlp = spacy.load("en_core_web_sm")

# Define the custom NER component
ner = nlp.get_pipe("ner")

# Load skills from the array
with open("./data/skills.json", "r") as f:
    skills = json.load(f)

# Add skills as labels to the NER component
for skill in skills:
    ner.add_label(skill)

# Load your annotated training data
with open("data/training_data.json", "r") as f:
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
            texts = [example['text'] for example in batch]
            annotations = [{'entities': example['entities']} for example in batch]
            examples = [
                Example.from_dict(nlp.make_doc(text), annotation)
                for text, annotation in zip(texts, annotations)
            ]
            nlp.update(examples, drop=0.5, losses=losses)
        print(f"Iteration {itn}: Losses: {losses}")

# Save the trained model
nlp.to_disk("models/custom_ner_model")
