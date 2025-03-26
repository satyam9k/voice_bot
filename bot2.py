import os
import streamlit as st
import google.generativeai as genai
import PyPDF2
import speech_recognition as sr
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile
import base64

class VoiceQABot:
    def __init__(self):
        # Configuration for Gemini API
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Placeholder PDF path (modify as needed)
        self.pdf_path = "aboutme.pdf"
        
        # Initialize conversation history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        # Initialize embedding model for semantic search
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Gemini model
        self.generation_model = genai.GenerativeModel('gemini-pro')
        
        # Extract and embed document text
        if 'document_text' not in st.session_state:
            st.session_state.document_text = self.extract_text_from_pdf()
        if 'document_embeddings' not in st.session_state:
            st.session_state.document_embeddings = self.create_document_embeddings(st.session_state.document_text)

    def display_ui(self):
        """
        Display the custom UI for voice interaction
        """
        st.title("Voice Q&A Assistant")
        
        # Speech recognition section
        st.write("🎤 Click the button and speak your question")
        
        if st.button("Start Voice Input"):
            query = self.voice_input()
            
            if query:
                # Retrieve relevant context
                chunks, embeddings = st.session_state.document_embeddings
                context = self.retrieve_relevant_context(query, chunks, embeddings)

                # Generate response
                response = self.generate_response(query, context)

                # Display response
                st.write("**Response:**", response)

                # Add to conversation history
                st.session_state.conversation_history.append({
                    'query': query,
                    'response': response
                })

        # Conversation History
        st.subheader("Conversation History")
        for entry in st.session_state.conversation_history:
            st.markdown(f"**Q:** {entry['query']}")
            st.markdown(f"**A:** {entry['response']}")

    def voice_input(self):
        """
        Capture voice input using Google Speech Recognition
        """
        recognizer = sr.Recognizer()
        
        try:
            with sr.Microphone() as source:
                st.write("Listening... Please speak now.")
                
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for the first phrase and wait for it
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                try:
                    # Recognize speech using Google Speech Recognition
                    query = recognizer.recognize_google(audio)
                    st.write(f"You said: {query}")
                    return query
                
                except sr.UnknownValueError:
                    st.error("Sorry, I couldn't understand the audio. Please try again.")
                    return None
                
                except sr.RequestError as e:
                    st.error(f"Could not request results from Google Speech Recognition service; {e}")
                    return None
        
        except Exception as e:
            st.error(f"An error occurred during voice input: {e}")
            return None

    def extract_text_from_pdf(self):
        """
        Extract text from local PDF file
        """
        try:
            if not os.path.exists(self.pdf_path):
                st.error(f"PDF file not found at {self.pdf_path}")
                return ""
            with open(self.pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

    def create_document_embeddings(self, text):
        """
        Create embeddings for document text
        """
        # Split text into chunks
        chunks = text.split('\n')
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        # Create embeddings
        embeddings = self.embedding_model.encode(chunks)
        
        return chunks, embeddings

    def retrieve_relevant_context(self, query, chunks, embeddings):
        """
        Perform semantic search to find most relevant context
        """
        # Encode the query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        
        # Get top 3 most similar chunks
        top_indices = np.argsort(similarities)[-3:][::-1]
        relevant_contexts = [chunks[i] for i in top_indices]
        
        return " ".join(relevant_contexts)

    def generate_response(self, query, context):
        """
        Generate response using Gemini with retrieved context
        """
        full_prompt = f"""
        Context: {context}

        Question: {query}

        Please provide a concise and engaging answer based on the context.
        Speak in a conversational tone, as if you're Satyam.
        If the context doesn't contain specific information about the query,
        explain that you don't have enough information to answer confidently.
        """
        
        try:
            response = self.generation_model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"Sorry, I encountered an error generating a response: {e}"

def main():
    # Set page configuration
    st.set_page_config(page_title="Voice Q&A Assistant", page_icon="🎤")
    
    # Create and run the bot
    bot = VoiceQABot()
    bot.display_ui()

if __name__ == "__main__":
    main()
