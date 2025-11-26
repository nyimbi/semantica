# Parse Module

The `parse` module provides comprehensive document parsing capabilities for extracting content, metadata, and structure from various file formats.

## Overview

- **Document Parsing**: PDF, DOCX, XLSX, PPTX, and more
- **OCR Processing**: Extract text from images and scanned documents
- **Table Extraction**: Extract tables with structure preservation
- **Metadata Extraction**: Author, title, dates, and custom metadata
- **Structure Analysis**: Headings, sections, lists, and document hierarchy
- **Multi-Format Support**: 50+ file formats with specialized parsers

---

## Algorithms Used

### PDF Parsing
- **Text Extraction**: PDFMiner algorithm for text layout analysis
- **OCR**: Tesseract with preprocessing (deskewing, noise reduction)
- **Table Detection**: Hough transform for line detection + clustering
- **Layout Analysis**: XY-cut algorithm for reading order determination

### OCR Algorithms
- **Preprocessing**: Otsu's thresholding, morphological operations
- **Text Detection**: EAST/CRAFT deep learning models
- **Recognition**: Tesseract LSTM + language models
- **Post-processing**: Spell correction, confidence filtering

### Table Extraction
- **Rule-based**: Line detection + cell clustering
- **ML-based**: Deep learning table structure recognition
- **Hybrid**: Combine rules + ML for robustness

---

## Quick Start

```python
from semantica.parse import DocumentParser

# Initialize parser
parser = DocumentParser()

# Parse with options
docs = parser.parse(
    sources=["documents/"],
    formats=["pdf", "docx", "xlsx"],
    metadata={"source": "company_docs"}
)

# Access parsed data
for doc in docs:
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Author: {doc.metadata.get('author')}")
    print(f"Pages: {doc.metadata.get('pages')}")
    print(f"Text length: {len(doc.text)} characters")
```

---

### PDFParser


**Example Usage:**

```python
from semantica.parse import PDFParser

# Basic PDF parsing
parser = PDFParser()
doc = parser.parse("document.pdf")

# Advanced PDF parsing with OCR
parser = PDFParser(
    ocr_enabled=True,
    ocr_language="eng",
    extract_tables=True,
    extract_images=True,
    dpi=300
)

doc = parser.parse("scanned_document.pdf")

# Access PDF-specific data
print(f"Pages: {doc.metadata['pages']}")
print(f"PDF version: {doc.metadata['pdf_version']}")
print(f"Tables extracted: {len(doc.tables)}")

# Access tables
for i, table in enumerate(doc.tables):
    print(f"Table {i+1}:")
    print(table.to_dataframe())
```

---

### DOCXParser


**Example Usage:**

```python
from semantica.parse import DOCXParser

# Parse Word document
parser = DOCXParser()
doc = parser.parse("document.docx")

# With structure preservation
parser = DOCXParser(
    preserve_structure=True,
    extract_tables=True,
    extract_images=True
)

doc = parser.parse("report.docx")

# Access structure
print(f"Headings: {doc.structure['headings']}")
print(f"Sections: {len(doc.structure['sections'])}")
print(f"Tables: {len(doc.tables)}")
```

---

### ExcelParser


**Example Usage:**

```python
from semantica.parse import ExcelParser

# Parse Excel file
parser = ExcelParser()
doc = parser.parse("data.xlsx")

# Parse specific sheets
parser = ExcelParser(
    sheets=["Sheet1", "Data"],
    header_row=0,
    skip_empty_rows=True
)

doc = parser.parse("workbook.xlsx")

# Access data
for sheet_name, df in doc.dataframes.items():
    print(f"Sheet: {sheet_name}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(df.head())
```

---

### HTMLParser


**Example Usage:**

```python
from semantica.parse import HTMLParser

# Parse HTML
parser = HTMLParser()
doc = parser.parse("page.html")

# Advanced parsing
parser = HTMLParser(
    extract_metadata=True,
    extract_links=True,
    remove_scripts=True,
    remove_styles=True
)

doc = parser.parse("article.html")

# Access extracted data
print(f"Title: {doc.metadata['title']}")
print(f"Links: {len(doc.links)}")
print(f"Main content: {doc.main_content}")
```

---

### JSONParser


**Example Usage:**

```python
from semantica.parse import JSONParser

# Parse JSON
parser = JSONParser()
doc = parser.parse("data.json")

# Flatten nested JSON
parser = JSONParser(
    flatten=True,
    separator=".",
    extract_schema=True
)

doc = parser.parse("nested_data.json")

# Access schema
print(f"Schema: {doc.schema}")
print(f"Flattened keys: {list(doc.flattened_data.keys())}")
```

---

### ImageParser


**Example Usage:**

```python
from semantica.parse import ImageParser

# Parse image with OCR
parser = ImageParser(
    ocr_enabled=True,
    ocr_language="eng",
    detect_orientation=True
)

doc = parser.parse("scanned_page.jpg")

# Access OCR results
print(f"Extracted text: {doc.text}")
print(f"Confidence: {doc.metadata['ocr_confidence']}")
print(f"Orientation: {doc.metadata['orientation']}")
```

---

### CodeParser


**Example Usage:**

```python
from semantica.parse import CodeParser

# Parse source code
parser = CodeParser(
    language="python",
    extract_functions=True,
    extract_classes=True,
    extract_docstrings=True
)

doc = parser.parse("module.py")

# Access code structure
print(f"Functions: {len(doc.functions)}")
print(f"Classes: {len(doc.classes)}")

for func in doc.functions:
    print(f"Function: {func.name}")
    print(f"  Parameters: {func.parameters}")
    print(f"  Docstring: {func.docstring}")
```

---

## Common Patterns

### Pattern 1: Batch Parsing

```python
from semantica.parse import DocumentParser
from pathlib import Path

parser = DocumentParser()

# Get all files
files = list(Path("documents/").rglob("*.*"))

# Parse in batches
batch_size = 10
for i in range(0, len(files), batch_size):
    batch = files[i:i+batch_size]
    docs = parser.parse(batch)
    # Process docs
```

### Pattern 2: Format-Specific Parsing

```python
from semantica.parse import PDFParser, DOCXParser, ExcelParser

# Route to appropriate parser
def parse_document(file_path):
    if file_path.endswith('.pdf'):
        parser = PDFParser(ocr_enabled=True)
    elif file_path.endswith('.docx'):
        parser = DOCXParser(preserve_structure=True)
    elif file_path.endswith('.xlsx'):
        parser = ExcelParser()
    else:
        raise ValueError(f"Unsupported format: {file_path}")
    
    return parser.parse(file_path)
```

### Pattern 3: Error Handling

```python
from semantica.parse import DocumentParser, ParseError

parser = DocumentParser()

successful = []
failed = []

for file_path in file_paths:
    try:
        doc = parser.parse(file_path)
        successful.append(doc)
    except ParseError as e:
        print(f"Failed to parse {file_path}: {e}")
        failed.append((file_path, str(e)))

print(f"Parsed: {len(successful)}, Failed: {len(failed)}")
```

---

## Configuration

```yaml
# config.yaml - Parse Configuration

parse:
  pdf:
    ocr_enabled: true
    ocr_language: eng
    extract_tables: true
    extract_images: true
    dpi: 300
    
  docx:
    preserve_structure: true
    extract_tables: true
    extract_images: true
    
  excel:
    header_row: 0
    skip_empty_rows: true
    infer_types: true
    
  html:
    extract_metadata: true
    extract_links: true
    remove_scripts: true
    remove_styles: true
    
  image:
    ocr_enabled: true
    ocr_language: eng
    detect_orientation: true
    
  code:
    extract_functions: true
    extract_classes: true
    extract_docstrings: true
```

---

## See Also

- [Ingest Module](ingest.md) - Data ingestion
- [Normalize Module](normalize.md) - Data cleaning and normalization
- [Semantic Extract Module](semantic_extract.md) - Entity and relationship extraction
