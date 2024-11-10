from flask import Flask, request, jsonify, send_file
from io import BytesIO
from docx import Document
from fpdf import FPDF
from langchain.llms import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
import os

app = Flask(__name__)


# Initialize the language model and prompt
huggingface_token = os.getenv("HUGGINGFACE_API_TOKEN")
llm = HuggingFaceHub(repo_id="facebook/bart-large-cnn", huggingfacehub_api_token=huggingface_token)
prompt = PromptTemplate(
    input_variables=["text"],
    template="Please provide a concise summary for the following text:\n\n{text}\n\nSummary:"
)
chain = LLMChain(llm=llm, prompt=prompt)

# Helper functions for file generation
def get_txt_file(content):
    return BytesIO(content.encode('utf-8'))

def get_docx_file(content):
    doc = Document()
    doc.add_paragraph(content)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

def get_pdf_file(content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in content.split('\n'):
        pdf.multi_cell(0, 10, line)
    buffer = BytesIO()
    pdf.output(buffer, "S")
    buffer.seek(0)
    return buffer

# Endpoint for summarizing text
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({"error": "No text provided for summarization."}), 400

    summary = chain.run({"text": text})
    return jsonify({"summary": summary})

# Endpoint for downloading the summary in different formats
@app.route('/download', methods=['POST'])
def download():
    data = request.get_json()
    text = data.get('text', '').strip()
    file_format = data.get('format', 'txt').lower()

    if not text:
        return jsonify({"error": "No text provided for summarization."}), 400

    summary = chain.run({"text": text})

    if file_format == 'txt':
        file = get_txt_file(summary)
        file_name = "summary.txt"
        mime_type = "text/plain"
    elif file_format == 'docx':
        file = get_docx_file(summary)
        file_name = "summary.docx"
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif file_format == 'pdf':
        file = get_pdf_file(summary)
        file_name = "summary.pdf"
        mime_type = "application/pdf"
    else:
        return jsonify({"error": "Unsupported file format. Choose from 'txt', 'docx', or 'pdf'."}), 400

    return send_file(file, as_attachment=True, download_name=file_name, mimetype=mime_type)

if __name__ == '__main__':
    app.run(debug=True)
