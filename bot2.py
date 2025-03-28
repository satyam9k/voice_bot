import os
import streamlit as st
import google.generativeai as genai
import PyPDF2
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import tempfile
import base64
import speech_recognition as sr
import time
import io 


try:

    from audio_recorder_streamlit import audio_recorder
except ImportError:
    st.error("Please install audio-recorder-streamlit: pip install audio-recorder-streamlit")
    audio_recorder = None

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


import logging
logging.getLogger('torch').setLevel(logging.ERROR)

# --- State Definitions ---
STATE_IDLE = "idle"
STATE_LISTENING = "listening"
STATE_PROCESSING = "processing"
STATE_SPEAKING = "speaking"

# --- Color Definitions ---
COLOR_IDLE = "#CCCCCC"
COLOR_LISTENING = "#FF4B4B" 
COLOR_PROCESSING = "#FFA500" 
COLOR_SPEAKING = "#4CAF50"

# --- Helper Functions ---

@st.cache_data
def get_base64_of_file(file_path):
    """Convert image file to base64 encoded string"""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.warning(f"Could not load image: {e}")
        return ""

def get_audio_base64(audio_file_path):
    """Convert audio file to base64 encoded string"""
    try:
        with open(audio_file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.error(f"Error encoding audio to base64: {e}")
        return None

def autoplay_audio_html(base64_audio_string, format="mp3"):
    """Generate HTML for auto-playing audio WITHOUT visible controls"""

    return f"""
    <audio autoplay style="width: 100%; display: none;"> <!-- Hidden -->
        <source src="data:audio/{format};base64,{base64_audio_string}" type="audio/{format}">
        Your browser does not support the audio element.
    </audio>
    """

# --- Styling and Soundwave ---

def apply_custom_styling():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown(f"""
    <style>
    /* General Styles */
    .stApp {{
        background: linear-gradient(135deg, #e0f2f1, #fff3e0);
        background-attachment: fixed;
    }}
    body, .stMarkdown, .stText {{
        color: #333333 !important; /* Darker text for better contrast */
        font-family: 'Roboto', sans-serif;
    }}

    /* Profile Picture */
    .circular-img {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 180px; /* Slightly smaller */
        height: 180px;
        object-fit: cover;
        border-radius: 50%;
        border: 4px solid #4CAF50; /* Thicker border */
        box-shadow: 0 6px 10px rgba(0,0,0,0.15); /* Enhanced shadow */
        margin-bottom: 15px; /* Spacing */
    }}

    /* Bio Card */
    .bio-card {{
        background-color: rgba(255,255,255,0.85); /* Slightly more opaque */
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        max-width: 600px;
        margin: 20px auto;
        border-left: 5px solid #4CAF50; /* Accent border */
    }}
    .bio-card h2 {{
        color: #2E7D32; /* Darker green heading */
        margin-bottom: 10px;
    }}
    .bio-card p {{
        line-height: 1.6;
    }}

    /* Conversation History */
    .conversation-history {{
        background-color: rgba(255,255,255,0.75);
        border: 1px solid #d0d0d0;
        border-radius: 10px;
        padding: 15px;
        max-height: 350px; /* Increased height */
        overflow-y: auto;
        width: 100%;
        box-sizing: border-box;
        margin-top: 15px; /* Spacing */
    }}
    .conversation-history .user-query {{
        color: #0D47A1; /* Blue for user */
        font-weight: bold;
        margin-bottom: 3px;
    }}
    .conversation-history .bot-response {{
        color: #333; /* Standard text for bot */
        margin-bottom: 15px; /* Space below response */
        padding-bottom: 10px; /* Space before divider */
        border-bottom: 1px dashed #e0e0e0; /* Divider */
    }}
    .conversation-history .bot-response:last-child {{
        border-bottom: none; /* No divider after last response */
    }}

    /* Soundwave Animation */
    .soundwave-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 60px; /* Increased height */
        margin: 10px 0;
    }}
    .soundwave-bar {{
        width: 6px; /* Thicker bars */
        height: 10px;
        margin: 0 4px; /* More spacing */
        /* background-color will be set by inline style */
        border-radius: 3px; /* Rounded tops */
        animation: soundwave 1.0s infinite ease-in-out alternate;
    }}

    @keyframes soundwave {{
        0% {{ height: 8px; opacity: 0.7; }}
        100% {{ height: 45px; opacity: 1.0; }} /* Taller and fully opaque */
    }}

    /* Stagger animation delays */
    .soundwave-bar:nth-child(1) {{ animation-delay: 0s; }}
    .soundwave-bar:nth-child(2) {{ animation-delay: 0.15s; }}
    .soundwave-bar:nth-child(3) {{ animation-delay: 0.3s; }}
    .soundwave-bar:nth-child(4) {{ animation-delay: 0.15s; }} /* Symmetric delay */
    .soundwave-bar:nth-child(5) {{ animation-delay: 0s; }} /* Symmetric delay */


    div[data-testid="stAudioRecorder"] > div > button {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important; /* Adjust padding if needed */
    }}
    /* Style the icon color directly if possible (already handled by component props) */
     /* div[data-testid="stAudioRecorder"] svg {{
         fill: #YourColor !important;
     }} */

    </style>
    """, unsafe_allow_html=True)

def get_soundwave_html(color="#42bd59", label=""):
    """Generate HTML for soundwave animation with dynamic color and optional label"""

    bars_html = "".join([
        f'<div class="soundwave-bar" style="background-color: {color};"></div>'
        for _ in range(5)
    ])
    label_html = f'<p style="text-align: center; color: #555; font-size: 0.9em; height: 1.2em;">{label}</p>' if label else '<p style="height: 1.2em;"></p>' 

    return f"""
    <div class="soundwave-container">
        {bars_html}
    </div>
    {label_html}
    """

# --- Main Class ---

class VoiceQABot:
    def __init__(self):

        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not self.GEMINI_API_KEY:
            st.error("GEMINI_API_KEY not found in .env file. Please set it.")
            st.stop()


        genai.configure(api_key=self.GEMINI_API_KEY)


        self.pdf_path = os.getenv("PDF_PATH", "aboutme.pdf")
        self.profile_pic_path = os.getenv("PROFILE_PIC_PATH", "sk.jpeg")


        try:
            self.profile_pic_base64 = get_base64_of_file(self.profile_pic_path)
            self.profile_pic_html = f'<img src="data:image/jpeg;base64,{self.profile_pic_base64}" class="circular-img" alt="My Picture">'
        except Exception as e:
            st.warning(f"Could not load profile picture: {e}")
            self.profile_pic_html = '<p style="text-align:center; color: red;">Profile picture not found.</p>'


        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        if 'app_state' not in st.session_state:
            st.session_state.app_state = STATE_IDLE
        if 'status_message' not in st.session_state:
            st.session_state.status_message = "" 


        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.generation_model = genai.GenerativeModel('gemini-2.0-flash')
        except Exception as e:
            st.error(f"Error initializing AI models: {e}")
            st.stop()

        self._load_document_embeddings()

    def _load_document_embeddings(self):
        """Load document text and create embeddings with error handling"""
        try:

            if 'document_text' not in st.session_state:
                st.session_state.document_text = self._extract_text_from_pdf()

            if 'document_embeddings' not in st.session_state or not st.session_state.document_text:
                if st.session_state.document_text:
                    st.session_state.document_embeddings = self._create_document_embeddings(
                        st.session_state.document_text
                    )
                else:
    
                    st.session_state.document_embeddings = ([], [])
                    st.warning("Could not load document text, context retrieval will be limited.")

        except Exception as e:
            st.error(f"Error loading document embeddings: {e}")
            st.session_state.document_text = ""
            st.session_state.document_embeddings = ([], [])

    def _set_state(self, state, message=""):
        """Helper to update state and message, triggering a rerun."""
        st.session_state.app_state = state
        st.session_state.status_message = message
       

    def display_ui(self):
        """Display the custom UI with profile and interaction components"""
        apply_custom_styling()



        # Create two columns
        left_col, right_col = st.columns([1, 1.2]) 

        # Left column - Profile and Bio
        with left_col:
            st.markdown(self.profile_pic_html, unsafe_allow_html=True)
            st.markdown(
                """
                <div class="bio-card">
                    <h2>Hi, I'm Satyam</h2>
                    <p>
                        An AI and Data Science enthusiast passionate about leveraging technology for positive impact.
                        My focus includes machine learning, cybersecurity, and data-driven problem-solving.
                        Feel free to ask me questions based on my background!
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Right column - Voice Interaction and History
        with right_col:
            st.markdown("## Talk to me!")
            st.markdown("Click the microphone, ask a question, and I'll respond.")

            current_state = st.session_state.get('app_state', STATE_IDLE)
            current_message = st.session_state.get('status_message', "")
            if current_state == STATE_LISTENING:
                soundwave_color = COLOR_LISTENING
            elif current_state == STATE_PROCESSING:
                soundwave_color = COLOR_PROCESSING
            elif current_state == STATE_SPEAKING:
                soundwave_color = COLOR_SPEAKING
            else: 
                soundwave_color = COLOR_IDLE

            soundwave_placeholder = st.empty()
            soundwave_placeholder.markdown(
                get_soundwave_html(color="#42bd59", label=current_message),
                unsafe_allow_html=True
            )


            audio_output_placeholder = st.empty()

            if audio_recorder:
    
                audio_bytes = audio_recorder(
                    recording_color=COLOR_LISTENING, 
                    neutral_color=COLOR_SPEAKING,  
                    icon_name="microphone",
                    icon_size="2x",
                    pause_threshold=2.0, 
                    sample_rate=16000, 
                    key="audio_input"
                )

                if audio_bytes:
        
                    self._set_state(STATE_LISTENING, "Processing audio...") 
  
                    self._handle_audio_input(audio_bytes, audio_output_placeholder)
  

            else:
                st.warning("Audio recording functionality requires the 'audio-recorder-streamlit' library.")

            # Conversation History
            st.markdown("## Conversation History")
            conversation_container = st.container()
            with conversation_container:
                st.markdown('<div class="conversation-history">', unsafe_allow_html=True)

                if not st.session_state.conversation_history:
                     st.markdown("<p style='color:#777;'>No conversation yet. Ask me something!</p>", unsafe_allow_html=True)
                else:
                    for i, entry in enumerate(st.session_state.conversation_history):
                        st.markdown(f'<p class="user-query">You ({len(st.session_state.conversation_history)-i}): {entry["query"]}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="bot-response">Bot: {entry["response"]}</p>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

    def _handle_audio_input(self, audio_bytes, audio_output_placeholder):
        """Handle audio input processing, response generation, and audio output"""
        temp_audio_path = None 
        bot_audio_path = None

        try:
            # 1. Save recorded audio temporarily

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name

            # Update state
            self._set_state(STATE_PROCESSING, "Transcribing...")

            # 2. Transcribe audio
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:

                audio = recognizer.record(source)

            query = recognizer.recognize_google(audio)


            if not query:
                raise ValueError("Transcription resulted in empty text.")

            # 3. Process query and generate response
            self._set_state(STATE_PROCESSING, "Thinking...")
            context = self._retrieve_relevant_context(query)
            response = self._generate_response(query, context)

            # 4. Update conversation history
            st.session_state.conversation_history.insert(0, {
                'query': query,
                'response': response
            })


            # 5. Generate bot's audio response
            self._set_state(STATE_SPEAKING, "Speaking...")
            bot_audio_path = self._text_to_speech(response)
            if not bot_audio_path:
                 raise ValueError("Failed to generate speech audio.") # Handle TTS failure

            # 6. Encode bot audio to Base64
            audio_base64 = get_audio_base64(bot_audio_path)
            if not audio_base64:
                raise ValueError("Failed to encode speech audio to Base64.")

            # 7. Display hidden auto-playing audio using HTML
            audio_html = autoplay_audio_html(audio_base64)
            audio_output_placeholder.markdown(audio_html, unsafe_allow_html=True)



        except sr.UnknownValueError:
            error_msg = "Sorry, I couldn't understand the audio. Please try speaking clearly."
            st.warning(error_msg)
            self._play_error_message(error_msg, audio_output_placeholder)
        except sr.RequestError as e:
            error_msg = f"Could not connect to speech recognition service: {e}. Check internet connection."
            st.error(error_msg)
            self._play_error_message(error_msg, audio_output_placeholder)
        except ValueError as ve: 
             error_msg = f"Processing error: {ve}. Please try again."
             st.error(error_msg)
             self._play_error_message(error_msg, audio_output_placeholder)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}. Please try again."
            st.exception(e) 
            self._play_error_message(error_msg, audio_output_placeholder)
        finally:
   
            if temp_audio_path and os.path.exists(temp_audio_path):
                try: os.unlink(temp_audio_path)
                except Exception as e_unlink: st.warning(f"Could not delete temp user audio file: {e_unlink}")
            if bot_audio_path and os.path.exists(bot_audio_path):
                try: os.unlink(bot_audio_path)
                except Exception as e_unlink: st.warning(f"Could not delete temp bot audio file: {e_unlink}")

     
            self._set_state(STATE_IDLE, "")


    def _play_error_message(self, error_text, audio_output_placeholder):
        """Generates and plays an audio message for errors."""
        bot_audio_path = None
        try:
            self._set_state(STATE_SPEAKING, "Error occurred...") 
            bot_audio_path = self._text_to_speech(error_text)
            if bot_audio_path:
                audio_base64 = get_audio_base64(bot_audio_path)
                if audio_base64:
                    audio_html = autoplay_audio_html(audio_base64)
                    audio_output_placeholder.markdown(audio_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to play error message audio: {e}")
        finally:
            if bot_audio_path and os.path.exists(bot_audio_path):
                try: os.unlink(bot_audio_path)
                except Exception as e_unlink: st.warning(f"Could not delete temp error audio file: {e_unlink}")


    def _extract_text_from_pdf(self):
        """Extract text from PDF with error handling"""
        try:
            if not os.path.exists(self.pdf_path):
                st.warning(f"PDF file not found at {self.pdf_path}")
                return ""

            text = ""
            with open(self.pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                if pdf_reader.is_encrypted:
                     st.warning(f"PDF file '{self.pdf_path}' is encrypted and cannot be read.")
                     return ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                           text += page_text + "\n" 
                    except Exception as page_e:
                        st.warning(f"Could not extract text from page {page_num + 1} of PDF: {page_e}")
            return text.strip() 
        except Exception as e:
            st.error(f"Error reading PDF '{self.pdf_path}': {e}")
            return ""

    def _create_document_embeddings(self, text):
        """Create embeddings for document text"""
        chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
        if not chunks:
            st.warning("No text chunks found after splitting the document.")
            return [], []

        try:
            with st.spinner(f"Creating embeddings for {len(chunks)} text chunks..."):
                 embeddings = self.embedding_model.encode(chunks, show_progress_bar=False) 
            #st.info("Embeddings created successfully.")
            return chunks, np.array(embeddings) 
        except Exception as e:
            st.error(f"Failed to create text embeddings: {e}")
            return [], []


    def _retrieve_relevant_context(self, query, top_k=3):
        """Retrieve most relevant context for the query"""
        try:
            chunks, embeddings = st.session_state.get('document_embeddings', ([], []))

            if not chunks or not isinstance(embeddings, np.ndarray) or embeddings.shape[0] == 0:
                 st.warning("No document embeddings available for context retrieval.")
                 return "No background context is available." 

            query_embedding = self.embedding_model.encode([query])[0]
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            num_chunks = len(chunks)
            actual_top_k = min(top_k, num_chunks)
            if actual_top_k <= 0:
                return "No relevant context found."

            top_indices = np.argsort(similarities)[-actual_top_k:][::-1]
            relevant_contexts = [chunks[i] for i in top_indices]
            return "\n---\n".join(relevant_contexts)

        except Exception as e:
            st.error(f"Error retrieving context: {e}")
            return "Error retrieving background information."


    def _generate_response(self, query, context):
        """Generate response using Gemini with retrieved context"""
        try:
            full_prompt = f"""
            You are Satyam, an AI assistant representing Satyam Kumar.
            Use the following context about Satyam to answer the user's question.
            Speak in a friendly, conversational, and first-person style (use "I", "my", etc.).
            Keep your answers concise and engaging, suitable for a spoken conversation.
            Dont use the phrase "hey there". Keep conversation tone professional.
            If the context doesn't contain specific information,
            give a response like, "Hmm, that's not something I'm familiar with right now, as it's not in my knowledge base. Maybe ask Satyam personally when he's around?"


            Context about Satyam:
            ---
            {context}
            ---

            User's Question: {query}

            Your Spoken Answer (as Satyam):
            """

            response = self.generation_model.generate_content(full_prompt)

            # Check for refusals or empty responses
            response_text = response.text.strip()
            if not response_text or any(phrase in response_text.lower() for phrase in ["cannot fulfill", "don't have information", "don't know"]):
                 if "No background context" in context or "Error retrieving" in context:
                     return "I couldn't find specific details about that in my current knowledge base. Perhaps ask me something else about my background or skills?"
                 else:
                     return "Hmm, that's an interesting question! While I have some background info, I don't have the specific details to answer that right now."

            return response_text

        except Exception as e:
            st.error(f"Error generating response from AI model: {e}")
            return "I seem to be having trouble formulating a response right now. Sorry about that!"

    def _text_to_speech(self, text):
        """Convert text to speech using gTTS and return the temp file path"""
        temp_audio_path = None
        try:
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_audio_path = temp_audio_file.name
            tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
            tts.save(temp_audio_path)
            temp_audio_file.close()
            return temp_audio_path
        except Exception as e:
            st.error(f"Error generating speech with gTTS: {e}")
            if temp_audio_path and os.path.exists(temp_audio_path):
                 try: os.unlink(temp_audio_path)
                 except Exception: pass
            return None

    def run(self):
        """Main application runner"""
        self.display_ui()

# --- Entry Point ---

def main():
    """Streamlit app entry point"""
    st.set_page_config(layout="wide", page_title="Satyam's AI Voice Assistant")
    bot = VoiceQABot()
    bot.run()

if __name__ == "__main__":
    main()
