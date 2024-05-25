import os
import easyocr
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,"ekyc_logs.log"), level=logging.INFO, format=logging_str, filemode="a")


# def extract_text(image_path, confidence_threshold=0.8):
#     # Initialize EasyOCR reader
#     reader = easyocr.Reader(['en'])

#     # Read the image and extract text
#     result = reader.readtext(image_path)

#     # Filter the extracted text based on confidence score
#     filtered_texts = {}
#     for text in result:
#         bounding_box, recognized_text, confidence = text
#         if confidence > confidence_threshold:
#             filtered_texts[recognized_text] = bounding_box

#     return filtered_texts


def extract_text(image_path, confidence_threshold=0.3, languages=['en']):
    """
    Extracts and filters text from an image using OCR, based on a confidence threshold.

    Parameters:
    - image_path (str): Path to the image file.
    - confidence_threshold (float): Minimum confidence for text inclusion. Default is 0.3.
    - languages (list): OCR languages. Default is ['en'].

    Returns:
    - str: Filtered text separated by '|' if confidence is met, otherwise an empty string.

    Raises:
    - Exception: Outputs error message if OCR processing fails.
    """
    

    logging.info("Text Extraction Started...")
    # Initialize EasyOCR reader
    reader = easyocr.Reader(languages)
    
    try:
        logging.info("Inside Try-Catch...")
        # Read the image and extract text
        result = reader.readtext(image_path)
        filtered_text = "|"  # Initialize an empty string to store filtered text
        for text in result:
            bounding_box, recognized_text, confidence = text
            if confidence > confidence_threshold:
                filtered_text += recognized_text + "|"  # Append filtered text with newline

        return filtered_text 
    except Exception as e:
        print("An error occurred during text extraction:", e)
        logging.info(f"An error occurred during text extraction: {e}")
        return ""


    # Filter the extracted text based on confidence score
    