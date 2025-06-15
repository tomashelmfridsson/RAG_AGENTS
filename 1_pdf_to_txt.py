import os
import openai
import fitz  # PyMuPDF
import cv2
import easyocr
import torch
import warnings
from dotenv import load_dotenv
from tqdm import tqdm

# Suppress all warnings
warnings.filterwarnings("ignore")

# Load API key from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define input and output directories
INPUT_DIR = "datapdf"
OUTPUT_DIR = "datatxt"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize EasyOCR reader with GPU support if available
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# Lists to track processed, skipped, and failed files
processed_files = []
skipped_files = []
failed_files = []

def extract_text_from_pdf(pdf_path):
    """
    Extracts raw text from a PDF file using PyMuPDF.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as pdf:
            with tqdm(total=len(pdf), desc=f"Extracting text from {os.path.basename(pdf_path)}", leave=True) as pbar:
                for page_num in range(len(pdf)):
                    page = pdf.load_page(page_num)
                    page_text = page.get_text()

                    # If no text is found, assume it's a scanned PDF
                    if not page_text.strip():
                        image = page.get_pixmap()
                        image_path = f"temp_image_{page_num}.png"
                        image.save(image_path)
                        page_text = recognize_text_from_image(image_path)
                        os.remove(image_path)

                    text += page_text + "\n"
                    pbar.update(1)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        failed_files.append(pdf_path)
    return text

def recognize_text_from_image(image_path):
    """
    Recognize text from an image using EasyOCR with GPU support.
    """
    try:
        results = reader.readtext(image_path, detail=0)
        recognized_text = "\n".join(results)
        return recognized_text if recognized_text else "No text detected"
    except Exception as e:
        print(f"Error during OCR: {e}")
        return "OCR failed"

def process_text_with_openai(raw_text):
    """
    Uses OpenAI API to process the extracted text (cleanup and formatting) using the new API interface.
    """
    try:
        response = openai.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that cleans up and formats PDF text data."},
                {"role": "user", "content": raw_text}
            ],
            max_tokens=3000,
            temperature=0.2
        )

        # Extract and return the cleaned-up text
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error processing text with OpenAI API: {e}")
        return raw_text

def save_text_to_file(text, output_path):
    """
    Saves the processed text to a .txt file.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(text)
    except Exception as e:
        print(f"Error saving text file {output_path}: {e}")
        failed_files.append(output_path)

def convert_pdf_to_txt():
    """
    Main function to convert all PDF files in the input directory to text files.
    """
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in the datapdf folder.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")

    with tqdm(total=len(pdf_files), desc="Processing PDFs", leave=True) as progress_bar:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(INPUT_DIR, pdf_file)
            txt_filename = os.path.splitext(pdf_file)[0] + ".txt"
            txt_path = os.path.join(OUTPUT_DIR, txt_filename)

            # Step 1: Extract raw text from PDF
            raw_text = extract_text_from_pdf(pdf_path)

            if not raw_text.strip():
                print(f"Skipped (empty text): {pdf_file}")
                skipped_files.append(pdf_file)
                progress_bar.update(1)
                continue

            # Step 2: Process the text using OpenAI API
            processed_text = process_text_with_openai(raw_text)

            if processed_text is None:
                print(f"Failed to process: {pdf_file}")
                failed_files.append(pdf_file)
                progress_bar.update(1)
                continue

            # Step 3: Save the processed text to a .txt file
            save_text_to_file(processed_text, txt_path)

            # Track the successfully processed file
            processed_files.append(pdf_file)
            print(f"Successfully saved: {txt_path}")
            progress_bar.update(1)

    # Summary of the results
    print("\nSummary:")
    print(f"Total PDF files: {len(pdf_files)}")
    print(f"Processed successfully: {len(processed_files)}")
    print(f"Skipped (empty text): {len(skipped_files)}")
    print(f"Failed (errors): {len(failed_files)}")

    if skipped_files:
        print("\nSkipped Files:")
        for file in skipped_files:
            print(f" - {file}")

    if failed_files:
        print("\nFailed Files:")
        for file in failed_files:
            print(f" - {file}")

if __name__ == "__main__":
    convert_pdf_to_txt()
    print("All files have been processed.")
