from spacy.training import offsets_to_biluo_tags

import spacy

nlp = spacy.load("en_core_web_sm")

text = "Bob has certifications in AWS, Azure, and Google Cloud."
entities = [
        [26, 29, "SKILL"],
        [31, 36, "SKILL"],
        [42, 54, "SKILL"]
    ]
doc = nlp.make_doc(text)
biluo_tags = offsets_to_biluo_tags(doc, entities)
print(f"BILUO Tags: {biluo_tags}")
doc = nlp.make_doc(text)
for start, end, label in entities:
    span = doc.char_span(start, end, label=label)
    if span is None:
        print(f"Skipping entity [{start}, {end}, '{label}'] due to misalignment.")
    else:
        print(f"Aligned entity: {span.text} with indices [{span.start_char}, {span.end_char}]")
