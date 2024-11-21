import google.generativeai as genai
from PIL import Image
import io

# Configure Google Generative AI API Key
def configure_genai(api_key):
    genai.configure(api_key=api_key)

# Handle image input and generate response based on image
def handle_image(uploaded_image, user_question=None):
    try:
        # Open the image file
        image = Image.open(uploaded_image)
        
        # Image processing
        model = genai.GenerativeModel('gemini-1.5-flash')
        if user_question:
            response = model.generate_content([user_question, image])
        else:
            response = model.generate_content(image)
        
        return response.text
    except Exception as e:
        return f"An error occurred while processing the image: {e}"
