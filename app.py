import os
import streamlit as st
import tempfile
import logging
import re
from docx2pdf import convert
from pdf2image import convert_from_path
import pdfplumber
from PIL import Image
import fitz  # PyMuPDF
from gtts import gTTS

# Check for PyTorch or TensorFlow
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

if TORCH_AVAILABLE or TF_AVAILABLE:
    from transformers import pipeline, AutoTokenizer
else:
    pipeline = None
    st.error("Error: Neither PyTorch nor TensorFlow is installed. Please install one of them to enable summarization.\n"
             "- To install PyTorch: `pip install torch` (see https://pytorch.org/)\n"
             "- To install TensorFlow: `pip install tensorflow` (see https://www.tensorflow.org/install/)")

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Directory for temporary files
TEMP_DIR = os.path.join(tempfile.gettempdir(), "chapter_summarizer")
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize session state
if 'GENERATED_OUTPUTS' not in st.session_state:
    st.session_state['GENERATED_OUTPUTS'] = {}
if 'section_titles' not in st.session_state:
    st.session_state['section_titles'] = []
if 'selected_section' not in st.session_state:
    st.session_state['selected_section'] = None
if 'file_processed' not in st.session_state:
    st.session_state['file_processed'] = False
if 'current_file' not in st.session_state:
    st.session_state['current_file'] = None
if 'show_full' not in st.session_state:
    st.session_state['show_full'] = False

def convert_docx_to_pdf(docx_path, pdf_path):
    """Convert DOCX to PDF."""
    try:
        convert(docx_path, pdf_path)
        logger.info(f"Converted DOCX to PDF: {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"Error converting DOCX to PDF: {str(e)}")
        return None

def clean_text(text):
    """Clean text by removing headers, footers, figure captions, tables, and non-main content."""
    try:
        # Remove non-ASCII characters and normalize whitespace
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        lines = text.split('\n')
        cleaned_lines = []
        page_texts = []
        current_page = []

        for line in lines:
            if line.strip() == '\f' or not line.strip():
                if current_page:
                    page_texts.append(current_page)
                    current_page = []
                continue
            current_page.append(line.strip())
        if current_page:
            page_texts.append(current_page)

        header_candidates = set()
        footer_candidates = set()
        for page in page_texts:
            if not page:
                continue
            for line in page[:3]:
                if line and len(line) < 100:
                    header_candidates.add(line)
            for line in page[-3:]:
                if line and len(line) < 100:
                    footer_candidates.add(line)

        header_lines = {line for line in header_candidates if sum(line in page for page in page_texts) > len(page_texts) // 2}
        footer_lines = {line for line in footer_candidates if sum(line in page for page in page_texts) > len(page_texts) // 2}

        unwanted_patterns = [
            r'Canara Engineering College.*',
            r'Module \d+.*',
            r'\d{2}[A-Z]{2}\d{2,3}.*',
            r'Page \d+',
            r'^\d+$',
            r'^\s*Figure \d+\..*',
            r'^\s*Table \d+\..*',
            r'Copyright ©.*',
            r'^\d{4}-\d{4}.*',
            r'All rights reserved.*',
            r'^\s*Date:.*',
            r'^\s*\d+\s*/\s*\d+.*',
        ]

        for page in page_texts:
            for line in page:
                line = line.strip()
                if not line:
                    continue
                if line in header_lines or line in footer_lines:
                    continue
                if any(re.match(pattern, line, re.IGNORECASE) for pattern in unwanted_patterns):
                    continue
                if len(line) < 30 and not re.match(r'^\d+\.\d+\s+.*$|^Chapter \d+.*$|^Section \d+.*$', line):
                    continue
                cleaned_lines.append(line)

        cleaned_text = '\n'.join(cleaned_lines)
        cleaned_text = re.sub(r'\n\s*\n+', '\n\n', cleaned_text).strip()
        logger.debug(f"Cleaned text (first 500 chars): {cleaned_text[:500]}...")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text

def extract_text_pdfplumber(file_path):
    """Extract text from PDF using pdfplumber."""
    try:
        full_text = ""
        page_texts = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                full_text += text + "\n"
                page_texts.append((page_num, text))
        with open(os.path.join(TEMP_DIR, "raw_extracted_text.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)
        logger.info("Saved raw text to raw_extracted_text.txt")
        cleaned_text = clean_text(full_text)
        with open(os.path.join(TEMP_DIR, "extracted_text.txt"), "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        logger.info("Saved cleaned text to extracted_text.txt")
        return cleaned_text, page_texts
    except Exception as e:
        logger.error(f"Error extracting text with pdfplumber: {str(e)}")
        return None, []

def extract_text_pymupdf(file_path):
    """Extract text from PDF using PyMuPDF."""
    try:
        full_text = ""
        page_texts = []
        doc = fitz.open(file_path)
        for page_num, page in enumerate(doc):
            text = page.get_text("text") or ""
            full_text += text + "\n"
            page_texts.append((page_num, text))
        doc.close()
        with open(os.path.join(TEMP_DIR, "raw_extracted_text_pymupdf.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)
        logger.info("Saved raw text to raw_extracted_text_pymupdf.txt")
        cleaned_text = clean_text(full_text)
        with open(os.path.join(TEMP_DIR, "extracted_text_pymupdf.txt"), "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        logger.info("Saved cleaned text to extracted_text_pymupdf.txt")
        return cleaned_text, page_texts
    except Exception as e:
        logger.error(f"Error extracting text with PyMuPDF: {str(e)}")
        return None, []

def extract_images_pdfplumber(pdf_path):
    """Extract images and captions from PDF."""
    try:
        page_image_map = {}
        page_caption_map = {}
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                images = page.images
                page_images = []
                page_captions = []
                for img in images:
                    x0, top, x1, bottom = img['x0'], img['top'], img['x1'], img['bottom']
                    width, height = int(x1 - x0), int(bottom - top)
                    if width <= 0 or height <= 0:
                        continue
                    pdf_images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
                    if not pdf_images:
                        continue
                    pdf_image = pdf_images[0]
                    img_box = (int(x0), int(top), int(x1), int(bottom))
                    cropped_img = pdf_image.crop(img_box)
                    img_path = os.path.join(TEMP_DIR, f"image_{page_num}_{len(page_images)}.png")
                    cropped_img.save(img_path, format="PNG")
                    page_images.append(img_path)
                    caption_area = page.within_bbox((x0, bottom, x1, bottom + 50))
                    caption_text = caption_area.extract_text() or ""
                    if caption_text.strip():
                        page_captions.append(caption_text.strip())
                if page_images:
                    page_image_map[page_num] = page_images
                if page_captions:
                    page_caption_map[page_num] = page_captions
        with open(os.path.join(TEMP_DIR, "image_captions.txt"), "w", encoding="utf-8") as f:
            for page_num, captions in page_caption_map.items():
                f.write(f"Page {page_num + 1}:\n")
                for cap in captions:
                    f.write(f"  {cap}\n")
        logger.info(f"Extracted {sum(len(images) for images in page_image_map.values())} images")
        return page_image_map, page_caption_map
    except Exception as e:
        logger.error(f"Error extracting images: {str(e)}")
        return {}, {}

def summarize_image_captions(captions):
    """Summarize image captions."""
    if not captions:
        return "No image captions available."
    if not (TORCH_AVAILABLE or TF_AVAILABLE) or not pipeline:
        return "Error: Summarization unavailable due to missing PyTorch or TensorFlow."
    
    caption_text = "\n".join(captions)
    caption_text = re.sub(r'[^\x00-\x7F]+', ' ', caption_text)
    caption_text = re.sub(r'\s+', ' ', caption_text).strip()
    
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        # Limit caption text to avoid token errors
        summary = summarizer(caption_text[:900], max_length=150, min_length=30, do_sample=False)
        logger.info("Generated image caption summary")
        return summary[0]['summary_text'].strip()
    except Exception as e:
        logger.error(f"Error summarizing captions: {str(e)}")
        logger.error(f"Caption content: {caption_text[:500]}")
        return "Error summarizing captions."

def extract_chapters_fallback(text, page_image_map, page_caption_map, page_texts):
    """Fallback parsing using regex."""
    chapters = []
    current_chapter = None
    current_subchapter = None
    lines = text.split('\n')
    content_buffer = []

    chapter_patterns = [
        r'^(\d+\.\d+)\s+(.+)$',
        r'^Chapter\s+(\d+)\s*[:-]?\s*(.+)$',
        r'^Section\s+(\d+)\s*[:-]?\s*(.+)$',
        r'^([IVX]+)\.\s+(.+)$',
        r'^(Introduction|Abstract|Conclusion|Preface)\s*$',
        r'^Part\s+(\d+)\s*[:-]?\s*(.+)$',
    ]
    subchapter_patterns = [
        r'^(\d+\.\d+\.\d+)\s+(.+)$',
        r'^\s*(\d+\.\d+)\s+(.+)$',
    ]

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        chapter_match = None
        for pattern in chapter_patterns:
            match = re.match(pattern, line)
            if match:
                chapter_match = match
                break

        subchapter_match = None
        for pattern in subchapter_patterns:
            match = re.match(pattern, line)
            if match:
                subchapter_match = match
                break

        if chapter_match:
            if current_chapter and content_buffer:
                current_chapter["content"] = "\n".join(content_buffer).strip()
                if len(current_chapter["content"]) > 50:
                    chapters.append(current_chapter)
                content_buffer = []
            number = chapter_match.group(1) if len(chapter_match.groups()) > 1 else ""
            title = chapter_match.group(2) if len(chapter_match.groups()) > 1 else chapter_match.group(1)
            chapter_title = f"{number} {title}".strip() if number else title
            chapter_pages = []
            for page_num, page_text in page_texts:
                if chapter_title in page_text:
                    chapter_pages.append(page_num)
            chapter_images = []
            chapter_captions = []
            for page_num in chapter_pages:
                chapter_images.extend(page_image_map.get(page_num, []))
                chapter_captions.extend(page_caption_map.get(page_num, []))
            caption_summary = summarize_image_captions(chapter_captions)
            current_chapter = {
                "title": chapter_title,
                "content": "",
                "subchapters": [],
                "images": chapter_images,
                "caption_summary": caption_summary,
                "pages": chapter_pages
            }
            current_subchapter = None
        elif subchapter_match and current_chapter:
            if current_subchapter and content_buffer:
                current_subchapter["content"] = "\n".join(content_buffer).strip()
                if len(current_subchapter["content"]) > 50:
                    current_chapter["subchapters"].append(current_subchapter)
                content_buffer = []
            number, title = subchapter_match.groups()
            subchapter_title = f"{number} {title}".strip()
            subchapter_pages = current_chapter["pages"]
            subchapter_images = current_chapter["images"]
            subchapter_captions = chapter_captions
            caption_summary = summarize_image_captions(subchapter_captions)
            current_subchapter = {
                "title": subchapter_title,
                "content": "",
                "images": subchapter_images,
                "caption_summary": caption_summary,
                "pages": subchapter_pages
            }
        elif current_chapter and line and not any(keyword in line.lower() for keyword in ["figure", "table"]):
            content_buffer.append(line)

    if current_chapter and content_buffer:
        if current_subchapter:
            current_subchapter["content"] = "\n".join(content_buffer).strip()
            if len(current_subchapter["content"]) > 50:
                current_chapter["subchapters"].append(current_subchapter)
        current_chapter["content"] = "\n".join(content_buffer).strip()
        if len(current_chapter["content"]) > 50:
            chapters.append(current_chapter)

    logger.info(f"Extracted {len(chapters)} chapters with fallback parsing")
    return chapters

def summarize_text(text):
    """Summarize text using BART model with improved chunking."""
    if not (TORCH_AVAILABLE or TF_AVAILABLE) or not pipeline:
        logger.error("Summarization unavailable: PyTorch or TensorFlow not installed")
        return "Error: Summarization unavailable due to missing PyTorch or TensorFlow."
    
    try:
        if not text or len(text.strip()) < 50:
            logger.error("Input text is too short or empty for summarization")
            return "Error: Input text is too short or empty."
        
        # Remove problematic characters and normalize whitespace
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        max_chunk = 900  # Reduced to stay within token limit
        overlap = 150
        chunks = []
        for i in range(0, len(text), max_chunk - overlap):
            chunk = text[i:i + max_chunk]
            if len(chunk.strip()) < 30:
                continue
            # Check token count
            tokens = tokenizer(chunk, truncation=False, return_tensors="pt")
            if len(tokens['input_ids'][0]) > 1024:
                chunk = chunk[:int(len(chunk) * 0.9)]  # Reduce chunk size if too long
            chunks.append(chunk)
        
        if not chunks:
            logger.error("No valid chunks created for summarization")
            return "Error: No valid chunks created for summarization."
            
        summaries = []
        for chunk in chunks:
            try:
                summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            except Exception as e:
                logger.error(f"Error summarizing chunk: {str(e)}")
                logger.error(f"Chunk content: {chunk[:500]}")
                summaries.append("Error summarizing this section.")
        
        combined_summary = " ".join(summaries).strip()
        combined_summary = re.sub(r'\s+', ' ', combined_summary).strip()
        return combined_summary
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return f"Error summarizing text: {str(e)}"

def generate_summary_parts(text):
    """Generate different parts of the summary: conclusion, key points, and full summary."""
    if not (TORCH_AVAILABLE or TF_AVAILABLE) or not pipeline:
        logger.error("Summarization unavailable: PyTorch or TensorFlow not installed")
        return (
            "Error: Summarization unavailable due to missing PyTorch or TensorFlow.",
            "- Error: Unable to generate key points due to missing dependencies.",
            "Error: Summarization unavailable due to missing PyTorch or TensorFlow."
        )

    try:
        if not text or len(text.strip()) < 50:
            logger.error("Input text is too short or empty for summarization")
            return (
                "Error: Input text is too short or empty.",
                "- Error: Unable to generate key points due to insufficient content.",
                "Error: Input text is too short or empty."
            )

        # Clean text for summarization
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        # Conclusion
        conclusion_prompt = f"Provide a very concise conclusion (6 lines maximum) of the following content:\n\n{text[:900]}"
        try:
            tokens = tokenizer(conclusion_prompt, truncation=False, return_tensors="pt")
            if len(tokens['input_ids'][0]) > 1024:
                conclusion_prompt = conclusion_prompt[:int(len(conclusion_prompt) * 0.9)]
            conclusion = summarizer(conclusion_prompt, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
            conclusion = "\n".join(conclusion.split("\n")[:6]).strip()
        except Exception as e:
            logger.error(f"Error generating conclusion: {str(e)}")
            conclusion = "Error generating conclusion."

        # Key Points
        keypoints_prompt = f"Summarize the following content into 5-7 key points, formatted as bullet points:\n\n{{text}}"
        max_chunk = 900
        overlap = 150
        chunks = []
        for i in range(0, len(text), max_chunk - overlap):
            chunk = text[i:i + max_chunk]
            if len(chunk.strip()) < 30:
                continue
            tokens = tokenizer(chunk, truncation=False, return_tensors="pt")
            if len(tokens['input_ids'][0]) > 1024:
                chunk = chunk[:int(len(chunk) * 0.9)]
            chunks.append(chunk)
        
        keypoint_summaries = []
        for chunk in chunks:
            try:
                formatted_prompt = keypoints_prompt.format(text=chunk)
                tokens = tokenizer(formatted_prompt, truncation=False, return_tensors="pt")
                if len(tokens['input_ids'][0]) > 1024:
                    formatted_prompt = formatted_prompt[:int(len(formatted_prompt) * 0.9)]
                summary = summarizer(formatted_prompt, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
                keypoint_summaries.append(summary)
            except Exception as e:
                logger.error(f"Error summarizing chunk for key points: {str(e)}")
                logger.error(f"Chunk content: {chunk[:500]}")
                keypoint_summaries.append("Error summarizing this section.")
        
        combined_keypoints = " ".join(keypoint_summaries).strip()
        keypoints_lines = [line.strip() for line in combined_keypoints.split(". ") if line.strip()]
        keypoints = []
        for line in keypoints_lines[:7]:
            if not line.startswith('-'):
                keypoints.append(f"- {line}")
            else:
                keypoints.append(line)
        while len(keypoints) < 5:
            keypoints.append("- Additional point derived from content analysis.")
        
        # Full Summary
        chunk_summaries = []
        for chunk in chunks:
            try:
                tokens = tokenizer(chunk, truncation=False, return_tensors="pt")
                if len(tokens['input_ids'][0]) > 1024:
                    chunk = chunk[:int(len(chunk) * 0.9)]
                summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
                chunk_summaries.append(summary)
            except Exception as e:
                logger.error(f"Error summarizing chunk for full summary: {str(e)}")
                logger.error(f"Chunk content: {chunk[:500]}")
                chunk_summaries.append("Error summarizing this section.")
        
        full_summary = " ".join(chunk_summaries).strip()
        full_summary = re.sub(r'\s+', ' ', full_summary).strip()
        
        if len(full_summary) > 1000:
            try:
                tokens = tokenizer(full_summary, truncation=False, return_tensors="pt")
                if len(tokens['input_ids'][0]) > 1024:
                    full_summary = full_summary[:900]
                full_summary = summarizer(full_summary, max_length=500, min_length=200, do_sample=False)[0]['summary_text']
            except Exception as e:
                logger.error(f"Error in final summarization of full summary: {str(e)}")
        
        logger.info("Generated all summary parts")
        return conclusion, "\n".join(keypoints), full_summary
        
    except Exception as e:
        logger.error(f"Error generating summary parts: {str(e)}")
        fallback_summary = summarize_text(text)
        return (
            fallback_summary[:500] if fallback_summary else "Error: Unable to generate conclusion.",
            "- Error: Unable to generate key points.",
            fallback_summary if fallback_summary else "Error: Unable to generate full summary."
        )

def text_to_audiobook(text, output_file):
    """Convert text to audiobook."""
    try:
        # Clean text for audiobook
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        tts = gTTS(text=text, lang='en')
        tts.save(output_file)
        logger.info(f"Generated audiobook: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error generating audiobook: {str(e)}")
        return None

def process_file(file):
    """Process uploaded file and extract chapters."""
    if not file:
        st.error("Please upload a file.")
        return None, []

    # Save uploaded file to temporary directory
    file_ext = os.path.splitext(file.name)[1].lower()
    temp_file_path = os.path.join(TEMP_DIR, file.name)
    with open(temp_file_path, "wb") as f:
        f.write(file.read())

    if file_ext in ['.doc', '.docx']:
        temp_pdf = os.path.join(TEMP_DIR, "converted.pdf")
        result = convert_docx_to_pdf(temp_file_path, temp_pdf)
        if not result:
            st.error("Error: Failed to convert DOCX to PDF.")
            return None, []
        file_path = temp_pdf
    elif file_ext != '.pdf':
        st.error("Error: Unsupported file format. Please upload a PDF or DOC/DOCX file.")
        return None, []
    else:
        file_path = temp_file_path

    full_text, page_texts = extract_text_pdfplumber(file_path)
    if full_text is None or not full_text.strip():
        full_text, page_texts = extract_text_pymupdf(file_path)
        if not full_text:
            st.error("Error: Failed to extract text from PDF.")
            return None, []

    page_image_map, page_caption_map = extract_images_pdfplumber(file_path)
    chapters = extract_chapters_fallback(full_text, page_image_map, page_caption_map, page_texts)

    if not chapters:
        st.error("Error: No chapters extracted from the document.")
        return None, []

    section_titles = []
    for chapter in chapters:
        if chapter["title"]:
            section_titles.append(chapter["title"])
            for subchapter in chapter["subchapters"]:
                if subchapter["title"]:
                    section_titles.append(f"  {subchapter['title']}")

    return chapters, section_titles

def process_section(chapters, selected_section):
    """Process the selected section and generate outputs."""
    if not selected_section:
        return None

    selected_content = None
    selected_images = []
    selected_caption_summary = ""
    for chapter in chapters:
        if chapter["title"] == selected_section:
            selected_content = chapter["content"]
            selected_images = chapter["images"]
            selected_caption_summary = chapter["caption_summary"]
            break
        for subchapter in chapter["subchapters"]:
            if f"  {subchapter['title']}" == selected_section:
                selected_content = subchapter["content"]
                selected_images = subchapter["images"]
                selected_caption_summary = subchapter["caption_summary"]
                break
        if selected_content:
            break

    if not selected_content or len(selected_content.strip()) < 50:
        st.error("Error: Selected section content is too short or not found.")
        return {
            "conclusion": "Error: Selected section content is too short or not found.",
            "keypoints": "",
            "images": [],
            "caption_summary": "",
            "audiobook_path": None,
            "full_summary": "No full summary available."
        }

    conclusion, keypoints, full_summary = generate_summary_parts(selected_content)
    audio_path = os.path.join(TEMP_DIR, f"audio_{os.urandom(8).hex()}.mp3")
    audiobook_path = text_to_audiobook(full_summary, audio_path)

    return {
        "conclusion": conclusion,
        "keypoints": keypoints,
        "full_summary": full_summary,
        "audiobook_path": audiobook_path,
        "images": selected_images,
        "caption_summary": selected_caption_summary
    }

# Streamlit UI
st.title("Chapter Summarizer and Audiobook Generator")
st.markdown("Upload a PDF or DOC/DOCX file, select a chapter or subchapter, and view the summarized content, audiobook, extracted images, and image caption summary.")

# File uploader
uploaded_file = st.file_uploader("Upload PDF or DOC/DOCX File", type=["pdf", "doc", "docx"])

# Process file on upload
if uploaded_file and st.session_state['current_file'] != uploaded_file.name:
    with st.spinner("Processing file..."):
        chapters, section_titles = process_file(uploaded_file)
        if chapters:
            st.session_state['file_processed'] = True
            st.session_state['section_titles'] = section_titles
            st.session_state['chapters'] = chapters
            st.session_state['current_file'] = uploaded_file.name
            st.session_state['GENERATED_OUTPUTS'] = {}
            st.session_state['selected_section'] = None
            st.session_state['show_full'] = False
        else:
            st.session_state['file_processed'] = False
            st.session_state['section_titles'] = []
            st.session_state['chapters'] = []
            st.session_state['current_file'] = None

# Section dropdown
selected_section = st.selectbox(
    "Select Chapter/Subchapter",
    st.session_state['section_titles'],
    index=0 if st.session_state['section_titles'] else None,
    disabled=not st.session_state['file_processed']
)

# Process section button
if st.button("Process Section", disabled=not (uploaded_file and selected_section)):
    if uploaded_file and selected_section:
        with st.spinner("Processing section..."):
            result = process_section(st.session_state['chapters'], selected_section)
            if result:
                st.session_state['GENERATED_OUTPUTS'][selected_section] = result
                st.session_state['selected_section'] = selected_section
    else:
        st.warning("Please upload a file and select a section.")

# Display outputs
if st.session_state['selected_section'] and st.session_state['selected_section'] in st.session_state['GENERATED_OUTPUTS']:
    result = st.session_state['GENERATED_OUTPUTS'][st.session_state['selected_section']]
    
    st.subheader("Selected Chapter/Subchapter")
    st.text(st.session_state['selected_section'])

    st.subheader("Concise Conclusion (6 lines or less)")
    st.text_area("", result["conclusion"], height=100, disabled=True)

    st.subheader("Key Features (Bullet Points)")
    st.text_area("", result["keypoints"], height=200, disabled=True)

    if st.button("Show Full Summary and Audiobook"):
        st.session_state['show_full'] = True

    if st.session_state['show_full']:
        st.subheader("Full Chapter Summary")
        st.text_area("", result["full_summary"], height=400, disabled=True)

        st.subheader("Chapter Audiobook")
        if result["audiobook_path"]:
            st.audio(result["audiobook_path"])

    st.subheader("Section Images")
    if result["images"]:
        cols = st.columns(3)
        for i, img_path in enumerate(result["images"]):
            with cols[i % 3]:
                st.image(img_path, caption=f"Image {i+1}")
    else:
        st.write("No images available.")

    st.subheader("Image Caption Summary")
    st.text_area("", result["caption_summary"], height=150, disabled=True)
