# app.py
import os

from flask import Flask, request, render_template
from parser import parse_resume

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['resume']
        file_path = f"./uploads/{file.filename}"
        file.save(file_path)  # Save the uploaded file
        parsed_data = parse_resume(file_path)  # Parse the resume
        return render_template('result.html', parsed_data=parsed_data)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host=os.getenv('IP', '0.0.0.0'),
        port=int(os.getenv('PORT', 4444)))
