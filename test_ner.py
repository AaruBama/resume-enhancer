import spacy

# Load the trained model
nlp = spacy.load("models/custom_ner_model")

# Test the model with new text
test_text = "Rakesh Verma is proficient in Python, Java, and React."
doc = nlp(test_text)

print("Hello")
print(doc.ents)
# Print the recognized entities
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")
