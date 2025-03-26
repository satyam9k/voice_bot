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
import pygame
import time
import threading
import base64
from queue import Queue, Empty
from streamlit.runtime.scriptrunner import add_script_run_ctx

# --- Utility: Load image and convert to base64 ---
@st.cache_data
def get_base64_of_file(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Configuration for Gemini API
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# --- Custom CSS and UI Styling ---
def apply_custom_styling():
    """
    Apply custom CSS styling to the Streamlit app
    """
    st.markdown("""
    <style>
    /* Reset to default Streamlit styling with some minor enhancements */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Ensure text is clearly visible */
    body, .stMarkdown, .stText, .stDataFrame, .stMetric {
        color: #000000 !important;
    }
    
    /* Improved button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white !important;
        border: none;
        padding: 10px 25px;
        border-radius: 5px;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    
    /* Profile picture styling */
    .circular-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 200px;
        height: 200px;
        object-fit: cover;
        border-radius: 50%;
        border: 3px solid #4CAF50;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Bio card styling */
    .bio-card {
        background-color: #f4f4f4;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        max-width: 600px;
        margin: 20px auto;
    }
    
    /* Conversation history styling */
    .conversation-history {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Ensure all headers are visible */
    h1, h2, h3, h4, h5, h6 {
        color: #333333 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sound Wave Animation Functions ---
def create_sound_wave(num_bars=20, max_height=50, color="#4CAF50"):
    """
    Generate a single state of sound wave animation
    
    Args:
        num_bars (int): Number of bars in the wave
        max_height (int): Maximum height of bars
        color (str): Color of the bars
    
    Returns:
        str: HTML representation of the sound wave
    """
    heights = [int(max_height * (0.5 + 0.5 * np.sin(i + time.time() * 5))) for i in np.linspace(0, 2 * np.pi, num_bars)]
    bars = "".join([
        f'<div style="display: inline-block; width: 5px; height: {height}px; margin: 0 2px; background-color: {color}; transition: height 0.1s ease;"></div>' 
        for height in heights
    ])
    return f'<div style="display: flex; justify-content: center; align-items: center; height: 60px;">{bars}</div>'

def animate_sound_wave(duration=None, color="#4CAF50"):
    """
    Animate sound wave using Streamlit's session state
    
    Args:
        duration (float, optional): Duration to animate
        color (str): Color of the bars
    """
    start_time = time.time()

    if 'animation_stop' not in st.session_state:
        st.session_state.animation_stop = False
    st.session_state.animation_stop = False

    animation_placeholder = st.empty()
    
    def update_animation():
        while not st.session_state.animation_stop:
            if duration is not None and time.time() - start_time > duration:
                st.session_state.animation_stop = True
                break
            animation_placeholder.markdown(create_sound_wave(color=color), unsafe_allow_html=True)
            time.sleep(0.1)

        animation_placeholder.empty()

    animation_thread = threading.Thread(target=update_animation)
    add_script_run_ctx(animation_thread)
    animation_thread.start()
    
    return animation_thread

class VoiceQABot:
    def __init__(self):
        # Placeholder PDF path (modify as needed)
        self.pdf_path = "aboutme.pdf"
        profile_pic_path = "sk.jpeg"
        try:
            self.profile_pic_base64 = get_base64_of_file(profile_pic_path)
            self.profile_pic_html = f'<img src="data:image/jpeg;base64,{self.profile_pic_base64}" class="circular-img" alt="My Picture">'
        except Exception as e:
            self.profile_pic_html = "<p>Profile picture not found.</p>"
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        # Initialize embedding model for semantic search
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize Gemini model
        self.generation_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize pygame mixer for audio playback
        pygame.mixer.init()
        
        if 'document_text' not in st.session_state:
            st.session_state.document_text = self.extract_text_from_pdf()
        if 'document_embeddings' not in st.session_state:
            st.session_state.document_embeddings = self.create_document_embeddings(st.session_state.document_text)

    def display_ui(self):
        """
        Display the custom UI with profile picture and bio
        """
        apply_custom_styling()
        left_col, right_col = st.columns([1, 1])
        with left_col:
            st.markdown(self.profile_pic_html, unsafe_allow_html=True)
            st.markdown(
                """
                <div class="bio-card">
                    <h2>Hi, I'm Satyam</h2>
                    <p>
                        As an AI and Data Science professional, I believe technology can drive positive change. With expertise in
                        machine learning, I aim to solve complex challenges and create impactful solutions in areas like cybersecurity,
                        healthcare, and data-driven insights, pushing innovation for a better future.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with right_col:
            st.markdown("## Talk to me!")
            if st.button("🎤 Ask a Question"):
                self.handle_voice_interaction()

            st.markdown("## Conversation History")
            history_container = st.container()
            with history_container:
                st.markdown('<div class="conversation-history">', unsafe_allow_html=True)
                for entry in st.session_state.conversation_history:
                    st.markdown(f"**Q:** {entry['query']}")
                    st.markdown(f"**A:** {entry['response']}")
                st.markdown('</div>', unsafe_allow_html=True)

    def handle_voice_interaction(self):
        """
        Handle the entire voice interaction process
        """
        # Capture voice input
        query = self.voice_input()
        
        if query:
            # Retrieve relevant context
            chunks, embeddings = st.session_state.document_embeddings
            context = self.retrieve_relevant_context(query, chunks, embeddings)

            # Generate response
            response = self.generate_response(query, context)

            # Add to conversation history
            st.session_state.conversation_history.append({
                'query': query,
                'response': response
            })

            # Generate and play audio response
            try:
                audio_file = self.text_to_speech(response)
                self.play_audio(audio_file)
            except Exception as e:
                st.error(f"Error generating or playing audio: {e}")

    def voice_input(self):
        """
        Capture voice input from user
        """
        st.write("🎤 Speak your question...")
        
        animation_thread = animate_sound_wave(color="#4CAF50")
        
        # Use microphone as source
        with sr.Microphone() as source:
            # Adjust for ambient noise
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            try:
                # Listen for audio input
                audio = self.recognizer.listen(source, timeout=10)

                st.session_state.animation_stop = True
                animation_thread.join()

                query = self.recognizer.recognize_google(audio)
                
                return query
            except sr.UnknownValueError:
                error_audio = self.text_to_speech("Sorry, I couldn't understand you. Please ask your question again.")
                self.play_audio(error_audio)
                
                st.session_state.animation_stop = True
                animation_thread.join()
                return None
            except sr.RequestError:
                error_audio = self.text_to_speech("Could not request results from speech recognition service.")
                self.play_audio(error_audio)
                
                st.session_state.animation_stop = True
                animation_thread.join()
                return None
            except Exception as e:
                error_audio = self.text_to_speech("An error occurred during speech recognition.")
                self.play_audio(error_audio)
                
                st.session_state.animation_stop = True
                animation_thread.join()
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
        chunks = text.split('\n')
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        embeddings = self.embedding_model.encode(chunks)
        
        return chunks, embeddings

    def retrieve_relevant_context(self, query, chunks, embeddings):
        """
        Perform semantic search to find most relevant context
        """
        query_embedding = self.embedding_model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], embeddings)[0]
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

        Please provide a concise and engaging spoken-style answer based on the context.
        Talk from the first person perspective *as Satyam*.
        Speak as if you're having a natural conversation with someone.
        If the context doesn't contain specific information,
        give a response like, "Hmm, that's not something I'm familiar with right now, as it's not in my knowledge base. Maybe ask Satyam personally when he's around?"
        """
        response = self.generation_model.generate_content(full_prompt)
        return response.text

    def text_to_speech(self, text):
        """
        Convert text to speech with a male-like voice using gTTS
        """
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts = gTTS(text=text, lang='en', tld='co.uk')
        tts.save(temp_audio.name)
        return temp_audio.name

    def play_audio(self, audio_file):
        """
        Play audio file directly
        """
        try:
            animation_thread = animate_sound_wave(color="#FF5733")
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            st.session_state.animation_stop = True
            animation_thread.join()

            pygame.mixer.music.unload()
            os.unlink(audio_file)
        except Exception as e:
            st.error(f"Error playing audio: {e}")

    def run(self):
        """
        Main Streamlit application
        """
        self.display_ui()

def main():
    bot = VoiceQABot()
    bot.run()

if __name__ == "__main__":
    main()
