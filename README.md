PDF Outline Extractor :- 
This project is an offline PDF outline extraction system. It intelligently detects titles, headings, and the document structure from a set of PDF files. The solution uses font styles, position data, and NLP techniques to produce a clean outline in structured JSON format.

Project Highlights :- 

1. Extracts document title and headings using formatting + NLP
2. Supports English and Japanese documents (auto-detect language)
3. Builds a hierarchical outline (H1, H2, H3) using KMeans clustering on font size
4. Leverages boldness, indentation, and regex patterns for heading detection
5. Fully offline, no internet or cloud required
6. Fast multiprocessing-based PDF processing

Folder Structure :-
CHALLENGE_1A/
│
├── sample_dataset/
│ ├── pdfs/ → Input PDFs
│ └── outputs/ → Output JSONs
│
├── process_pdfs.py → Main script to process PDFs
├── requirements.txt → List of required Python packages
└── README.md → Project documentation (this file)

How to Run :- 
Option 1 – Run Locally with Python
Install dependencies
Ensure Python 3.10+ is installed. Then run:

pip install -r requirements.txt

Add your PDF files
Place all .pdf files in the folder: sample_dataset/pdfs/

python process_pdfs.py

Output .json files will be created in: sample_dataset/outputs/

Output Format :- 
Each output file will be a JSON file with the following structure:

1. title: Extracted title from the first page
2. outline: A list of detected headings with the following info:
3. level: H1, H2, or H3 (based on visual features)
4. text: The heading text
5. page: Page number in the document


System Requirements :- 

1. Python 3.10 or above
2. Internet (only for installing Python packages, not for execution)
3. RAM: ≤1GB (suitable for lightweight machines)
4. No GPU required
5. Works offline once dependencies are installed

Used Libraries :-

1. pdfplumber (for PDF parsing)
2. spaCy (for NLP)
3. langdetect (for language detection)
4. scikit-learn (for KMeans clustering)
5. numpy

Future Enhancements :-

1. Add content summary under each heading
2. Integrate semantic analysis using sentence embeddings
3. Support more languages and larger documents
4. Build a web interface to upload and view outlines

Docker Hub Repository (if applicable) :-

This command follow the make docker image.

docker build -t adobe_doc_analyzer .
docker tag adobe_doc_analyzer 11222750/challenge_1a:latest
docker login
docker push 11222750/challenge_1a:latest

https://hub.docker.com/repositories/11222750


