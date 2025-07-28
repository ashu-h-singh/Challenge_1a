# PDF Outline Extractor

This solution extracts structured outlines from PDF documents, identifying the title and headings (H1, H2, H3) with their respective page numbers.

## Approach

The solution uses a combination of techniques to extract and classify headings:

1. **Text Extraction**: Uses pdfplumber to extract text with formatting information (font size, position, etc.)
2. **Title Detection**: Identifies the document title based on font size, position, and NLP analysis
3. **Heading Detection**: Uses a multi-faceted approach:
   - Font size clustering to identify heading levels (H1, H2, H3)
   - Pattern matching for common heading formats
   - Formatting cues (bold text, capitalization)
   - NLP analysis for heading-like phrases
4. **Hierarchical Organization**: Organizes headings into a structured outline

## Libraries Used

- **pdfplumber**: For PDF text extraction with formatting information
- **spaCy**: For NLP analysis to improve heading detection
- **scikit-learn**: For clustering font sizes to determine heading levels
- **NumPy**: For numerical operations  and (**time , sys, and kmeans)

## Building and Running

### Build the Docker Image

```bash
docker build --platform linux/amd64 -t pdf-outline-extractor:latest .
```

### Run the Container

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none pdf-outline-extractor:latest
```

This will process all PDF files in the `input` directory and generate corresponding JSON files in the `output` directory.

## Performance

- The solution is optimized to process a 50-page PDF in under 10 seconds
- The total model size is under 200MB (primarily the spaCy model)
- The solution works completely offline with no internet access required

## Multilingual Support

The solution provides comprehensive multilingual support:

- **Language Detection**: Automatically detects the document language using the langdetect library
- **Universal Pattern Matching**: Uses language-agnostic patterns that work across different languages
- **Font Analysis**: Relies on font size clustering and formatting cues which are language-independent
- **Extensible Framework**: Supports adding language-specific patterns and NLP models

The solution currently includes the English spaCy model by default, but the architecture allows for easy addition of models for other languages.

## Key Features

1. **Robust Heading Detection**:
   - Uses font size clustering to identify heading levels
   - Pattern matching for common heading formats
   - Formatting cues (bold text, capitalization)
   - NLP analysis for heading-like phrases

2. **Performance Optimization**:
   - Processes PDFs efficiently to meet the 10-second constraint
   - Uses lightweight models to stay under the 200MB limit

3. **Error Handling**:
   - Gracefully handles various PDF formats and structures
   - Provides fallback mechanisms when heading detection is challenging

4. **Enhanced Multilingual Support**:
   - Automatic language detection for optimal processing
   - Language-specific heading patterns where available
   - Universal patterns that work across all languages
   - Font-based analysis that is language-independent