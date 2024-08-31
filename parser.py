# parser.py
import spacy
import re
from resume_handler import extract_text

# Load the SpaCy model
nlp = spacy.load("en_core_web_sm")

# Predefined list of skills
SKILLS_LIST = [
    "Python", "Java", "JavaScript", "C++", "C#", "React", "Node.js", "SQL",
    "Django", "Flask", "HTML", "CSS", "Ruby", "PHP", "Swift", "Kotlin", "AWS", "Docker"
]


def extract_email_and_phone(lines):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'

    email = None
    phone = None

    for line in lines:
        if not email and re.search(email_pattern, line):
            email = re.search(email_pattern, line).group()
        if not phone and re.search(phone_pattern, line):
            phone = re.search(phone_pattern, line).group()

    return email, phone


def parse_resume(file_path):
    # Extract text from the resume
    resume_text = extract_text(file_path)

    # Normalize the text
    resume_text = resume_text.lower()

    # Process the text with SpaCy
    doc = nlp(resume_text)

    # Initialize data structure to hold extracted information
    parsed_data = {
        "name": None,
        "email": None,
        "phone": None,
        "skills": [],
        "experience": [],
        "education": []
    }

    # Split text into lines for easier processing
    lines = resume_text.split('\n')

    # Extract name (assuming it's the first line)
    parsed_data["name"] = lines[0].strip()

    # Extract email and phone
    parsed_data["email"], parsed_data["phone"] = extract_email_and_phone(lines)

    # Extract skills using NER and predefined list
    for token in doc:
        if token.text in SKILLS_LIST:
            parsed_data["skills"].append(token.text.capitalize())

    # Extract experience and education based on keywords
    experience_keywords = ["experience", "work", "employment"]
    education_keywords = ["education", "degree", "school", "university", "college"]

    current_section = None

    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in experience_keywords):
            current_section = "experience"
        elif any(keyword in line_lower for keyword in education_keywords):
            current_section = "education"
        elif current_section == "experience" and line.strip():
            parsed_data["experience"].append(line.strip())
        elif current_section == "education" and line.strip():
            parsed_data["education"].append(line.strip())

    return parsed_data


# Example usage
if __name__ == "__main__":
    file_path = "resume-dataset/Resume2.pdf"  # Change to your resume path
    parsed_resume = parse_resume(file_path)
    print(parsed_resume)