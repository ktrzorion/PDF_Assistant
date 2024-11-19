## Information Extraction and Retrieval System

# Overview
This project is a information extraction and retrieval system designed to extract relevant information from PDF documents and provide a search interface to retrieve specific data.

Features
Extracts information from PDF documents using regular expressions and natural language processing techniques.
Caches extracted data for faster retrieval.
Provides a search interface to retrieve specific information based on queries.
Supports querying by any relevant topic.

# Requirements
Python 3.8+
OpenAI library for natural language processing
PyPDF2 library for PDF processing
FAISS library for similarity search
json library for data storage
datetime library for timestamping
asyncio library for asynchronous processing

# Installation
Clone the repository: git clone https://github.com/ktrzorion/PDF_Assistant
Install required libraries: pip install -r requirements.txt
Update the folder_path variable in processor.py to point to your PDF directory.

# Usage
Run python processor.py to extract information from PDFs and cache data.
Use the query_info function in processor.py to retrieve specific information.

# Contributing
Contributions are welcome! Please submit a pull request with your changes.

# License
This project is licensed under the MIT License.

# Acknowledgments
OpenAI for natural language processing capabilities
PyPDF2 for PDF processing capabilities
FAISS for similarity search capabilities

# Contact
Priyanshu Katiyar
ktrzorion@gmail.com

# Changelog
v1.0
Initial release
Feel free to modify this README file to suit your project's specific needs.
