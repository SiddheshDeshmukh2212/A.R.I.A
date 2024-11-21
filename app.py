import streamlit as st
from pdf_handler import get_pdf_text, get_text_chunks, get_vector_store
from image_handler import handle_image
from dotenv import load_dotenv
import os
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("API Key not found in environment variables.")

# Create the conversational chain for Q&A
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# Handle user input, search for similar documents, and get a response
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Loading FAISS index
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        
        return response["output_text"]
    except Exception as e:
        return f"An error occurred: {e}"

# Main Streamlit function
def main():
    st.set_page_config(page_title="Chat with PDF and Image", layout="wide")
    st.header("Chat with PDF and Image using Gemini 💁")

    # User input for querying the PDFs and Images
    user_question = st.text_input("Ask a Question from the PDF or Image Files")

    # File upload inputs for PDFs and Images
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, type="pdf")
        uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

        # PDF processing
        if st.button("Submit & Process PDF"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF Processing Complete!")
            else:
                st.warning("Please upload PDF files first.")

        # Image processing
        if st.button("Submit & Process Image"):
            if uploaded_image:
                with st.spinner("Processing Image..."):
                    handle_image(uploaded_image, user_question)
                    st.success("Image Processing Complete!")
            else:
                st.warning("Please upload an image first.")

    # Handle user question and display response
    if user_question:
        if pdf_docs:
            response = user_input(user_question)
            st.write(f"Reply: {response}")
        elif uploaded_image:
            # Process the image and get the answer
            response = handle_image(uploaded_image, user_question)
            st.write(f"Reply: {response}")
        else:
            st.write("Please upload a file (PDF/Image) first.")

# Run the app
if __name__ == "__main__":
    main()
