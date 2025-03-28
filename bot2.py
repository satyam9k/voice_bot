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
import io # Needed for base64 encoding of audio

# Avoid import issues
try:
    # Note: audio_recorder_streamlit may need to be installed via pip
    from audio_recorder_streamlit import audio_recorder 
except ImportError:
    st.error("Please install audio-recorder-streamlit: pip install audio-recorder-streamlit")
    audio_recorder = None

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Disable PyTorch logging to reduce noise
import logging
logging.getLogger('torch').setLevel(logging.ERROR)

# Prevent asyncio warnings (less critical now but good practice)
# import asyncio
# try:
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# except AttributeError:
#     pass # Not applicable on non-Windows or different asyncio setups

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
    """Generate HTML for auto-playing audio"""
    # Browsers often restrict autoplay without user interaction.
    # Recording audio *is* interaction, so this should usually work.
    # The 'controls' attribute provides a fallback UI.
    return f"""
    <audio autoplay controls style="width: 100%;">
        <source src="data:audio/{format};base64,{base64_audio_string}" type="audio/{format}">
        Your browser does not support the audio element.
    </audio>
    """

# --- Styling and Soundwave ---

def apply_custom_styling():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
    <style>
    /* General Styles */
    .stApp {
        background: linear-gradient(135deg, #e0f2f1, #fff3e0);
        background-attachment: fixed;
    }
    body, .stMarkdown, .stText {
        color: #333333 !important; /* Darker text for better contrast */
        font-family: 'Roboto', sans-serif;
    }

    /* Profile Picture */
    .circular-img {
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
    }

    /* Bio Card */
    .bio-card {
        background-color: rgba(255,255,255,0.85); /* Slightly more opaque */
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        max-width: 600px;
        margin: 20px auto;
        border-left: 5px solid #4CAF50; /* Accent border */
    }
    .bio-card h2 {
        color: #2E7D32; /* Darker green heading */
        margin-bottom: 10px;
    }
    .bio-card p {
        line-height: 1.6;
    }

    /* Conversation History */
    .conversation-history {
        background-color: rgba(255,255,255,0.75);
        border: 1px solid #d0d0d0;
        border-radius: 10px;
        padding: 15px;
        max-height: 350px; /* Increased height */
        overflow-y: auto;
        width: 100%;
        box-sizing: border-box;
        margin-top: 15px; /* Spacing */
    }
    .conversation-history .user-query {
        color: #0D47A1; /* Blue for user */
        font-weight: bold;
        margin-bottom: 3px;
    }
    .conversation-history .bot-response {
        color: #333; /* Standard text for bot */
        margin-bottom: 15px; /* Space below response */
        padding-bottom: 10px; /* Space before divider */
        border-bottom: 1px dashed #e0e0e0; /* Divider */
    }
    .conversation-history .bot-response:last-child {
        border-bottom: none; /* No divider after last response */
    }


    /* Soundwave Animation */
    .soundwave-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 60px; /* Increased height */
        margin: 10px 0;
    }
    .soundwave-bar {
        width: 6px; /* Thicker bars */
        height: 10px;
        margin: 0 4px; /* More spacing */
        background-color: #4CAF50;
        border-radius: 3px; /* Rounded tops */
        animation: soundwave 1.0s infinite ease-in-out alternate;
    }

    @keyframes soundwave {
        0% { height: 8px; opacity: 0.7; }
        100% { height: 45px; opacity: 1.0; } /* Taller and fully opaque */
    }

    /* Stagger animation delays */
    .soundwave-bar:nth-child(1) { animation-delay: 0s; }
    .soundwave-bar:nth-child(2) { animation-delay: 0.15s; }
    .soundwave-bar:nth-child(3) { animation-delay: 0.3s; }
    .soundwave-bar:nth-child(4) { animation-delay: 0.15s; } /* Symmetric delay */
    .soundwave-bar:nth-child(5) { animation-delay: 0s; } /* Symmetric delay */
    </style>
    """, unsafe_allow_html=True)

def get_soundwave_html(label="Processing..."):
    """Generate HTML for soundwave animation with a label"""
    return f"""
    <div class="soundwave-container">
        <div class="soundwave-bar"></div>
        <div class="soundwave-bar"></div>
        <div class="soundwave-bar"></div>
        <div class="soundwave-bar"></div>
        <div class="soundwave-bar"></div>
    </div>
    <p style="text-align: center; color: #555; font-size: 0.9em;">{label}</p>
    """

# --- Main Class ---

class VoiceQABot:
    def __init__(self):
        # Configuration and error handling for API key
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not self.GEMINI_API_KEY:
            st.error("GEMINI_API_KEY not found in .env file. Please set it.")
            st.stop()

        # Configure Gemini API
        genai.configure(api_key=self.GEMINI_API_KEY)

        # PDF and Profile Picture paths
        self.pdf_path = os.getenv("PDF_PATH", "aboutme.pdf")
        self.profile_pic_path = os.getenv("PROFILE_PIC_PATH", "sk.jpeg")

        # Profile Picture
        try:
            self.profile_pic_base64 = get_base64_of_file(self.profile_pic_path)
            self.profile_pic_html = f'<img src="data:image/jpeg;base64,{self.profile_pic_base64}" class="circular-img" alt="My Picture">'
        except Exception as e:
            st.warning(f"Could not load profile picture: {e}")
            self.profile_pic_html = '<p style="text-align:center; color: red;">Profile picture not found.</p>'

        # Initialize conversation history (newest first)
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        # Initialize models
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.generation_model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            st.error(f"Error initializing AI models: {e}")
            st.stop()

        # Load document text and embeddings
        self._load_document_embeddings()

    def _load_document_embeddings(self):
        """Load document text and create embeddings with error handling"""
        try:
            # Ensure document text is loaded
            if 'document_text' not in st.session_state:
                st.session_state.document_text = self._extract_text_from_pdf()

            # Create embeddings if not already done
            if 'document_embeddings' not in st.session_state or not st.session_state.document_text:
                if st.session_state.document_text:
                    st.session_state.document_embeddings = self._create_document_embeddings(
                        st.session_state.document_text
                    )
                else:
                    # Handle case where PDF text couldn't be loaded
                    st.session_state.document_embeddings = ([], [])
                    st.warning("Could not load document text, context retrieval will be limited.")

        except Exception as e:
            st.error(f"Error loading document embeddings: {e}")
            st.session_state.document_text = ""
            st.session_state.document_embeddings = ([], [])

    def display_ui(self):
        """Display the custom UI with profile and interaction components"""
        apply_custom_styling()

        st.markdown("<h1 style='color: #FF5733;'>AI Voice Assistant</h1>",
                    unsafe_allow_html=True)
        st.markdown("---")

        # Create two columns
        left_col, right_col = st.columns([1, 1.2]) # Give right col slightly more space

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

            # Placeholder for soundwaves and status messages
            status_placeholder = st.empty()
            audio_output_placeholder = st.empty() # For the autoplay audio player

            # Voice Input
            if audio_recorder:
                audio_bytes = audio_recorder(
                    text="Click to Speak",
                    recording_color="#FF4B4B", # Red when recording
                    neutral_color="#4CAF50",  # Green when idle
                    icon_name="microphone",
                    icon_size="2x",
                    pause_threshold=2.0, # Seconds of silence before stopping
                    sample_rate=16000 # Common sample rate
                )

                # Process audio if recorded
                if audio_bytes:
                    # Show user soundwave animation immediately
                    status_placeholder.markdown(get_soundwave_html("Listening..."), unsafe_allow_html=True)
                    # Process input (includes bot response generation and audio playback)
                    self._handle_audio_input(audio_bytes, status_placeholder, audio_output_placeholder)
      

            else:
                st.warning("Audio recording functionality requires the 'audio-recorder-streamlit' library.")

            # Conversation History
            st.markdown("## Conversation History")
            conversation_container = st.container()
            with conversation_container:
                st.markdown('<div class="conversation-history">', unsafe_allow_html=True)

                # Iterate directly - newest are at the start due to insert(0, ...)
                if not st.session_state.conversation_history:
                     st.markdown("<p style='color:#777;'>No conversation yet. Ask me something!</p>", unsafe_allow_html=True)
                else:
                    for i, entry in enumerate(st.session_state.conversation_history):
                        st.markdown(f'<p class="user-query">You ({len(st.session_state.conversation_history)-i}): {entry["query"]}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="bot-response">Bot: {entry["response"]}</p>', unsafe_allow_html=True)
                        # Removed the <hr> - using border-bottom in CSS now

                st.markdown('</div>', unsafe_allow_html=True)

    def _handle_audio_input(self, audio_bytes, status_placeholder, audio_output_placeholder):
        """Handle audio input processing, response generation, and audio output"""
        temp_audio_path = None # Initialize to ensure cleanup check works
        bot_audio_path = None

        try:
            # 1. Save recorded audio temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name

            # Update status
            status_placeholder.markdown(get_soundwave_html("Transcribing..."), unsafe_allow_html=True)

            # 2. Transcribe audio
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                # Adjust for ambient noise (optional but good)
                # recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source)
            
            # Use recognize_google - ensure internet connection
            query = recognizer.recognize_google(audio)
            st.write(f"Heard: {query}") # For debugging

            if not query:
                raise ValueError("Transcription resulted in empty text.")

            # 3. Process query and generate response
            status_placeholder.markdown(get_soundwave_html("Thinking..."), unsafe_allow_html=True)
            context = self._retrieve_relevant_context(query)
            response = self._generate_response(query, context)

            # 4. Update conversation history (insert at the beginning)
            st.session_state.conversation_history.insert(0, {
                'query': query,
                'response': response
            })

            # 5. Generate bot's audio response
            status_placeholder.markdown(get_soundwave_html("Speaking..."), unsafe_allow_html=True)
            bot_audio_path = self._text_to_speech(response)
            if not bot_audio_path:
                 raise ValueError("Failed to generate speech audio.") # Handle TTS failure

            # 6. Encode bot audio to Base64
            audio_base64 = get_audio_base64(bot_audio_path)
            if not audio_base64:
                raise ValueError("Failed to encode speech audio to Base64.")

            # 7. Display auto-playing audio using HTML
            audio_html = autoplay_audio_html(audio_base64)
            audio_output_placeholder.markdown(audio_html, unsafe_allow_html=True)

            # 8. Clear status placeholder after audio starts playing
            status_placeholder.empty()

        except sr.UnknownValueError:
            error_msg = "Sorry, I couldn't understand the audio. Please try speaking clearly."
            st.warning(error_msg)
            self._play_error_message(error_msg, status_placeholder, audio_output_placeholder)
        except sr.RequestError as e:
            error_msg = f"Could not connect to speech recognition service: {e}. Check internet connection."
            st.error(error_msg)
            self._play_error_message(error_msg, status_placeholder, audio_output_placeholder)
        except ValueError as ve: # Catch specific value errors (empty transcription, TTS fail)
             error_msg = f"Processing error: {ve}. Please try again."
             st.error(error_msg)
             self._play_error_message(error_msg, status_placeholder, audio_output_placeholder)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}. Please try again."
            st.exception(e) # Log the full traceback for debugging
            self._play_error_message(error_msg, status_placeholder, audio_output_placeholder)
        finally:
            # Clean up temporary files
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except Exception as e_unlink:
                    st.warning(f"Could not delete temp user audio file: {e_unlink}")
            if bot_audio_path and os.path.exists(bot_audio_path):
                try:
                    os.unlink(bot_audio_path)
                except Exception as e_unlink:
                    st.warning(f"Could not delete temp bot audio file: {e_unlink}")
            # Ensure status placeholder is cleared if an error occurred before step 8
            # Check if the placeholder still holds content before clearing
            # This check is tricky with st.empty(), safer to just call empty() if needed
            # status_placeholder.empty() # Let's rely on _play_error_message to clear it


    def _play_error_message(self, error_text, status_placeholder, audio_output_placeholder):
        """Generates and plays an audio message for errors."""
        bot_audio_path = None
        try:
            status_placeholder.markdown(get_soundwave_html("Error Occurred..."), unsafe_allow_html=True)
            bot_audio_path = self._text_to_speech(error_text)
            if bot_audio_path:
                audio_base64 = get_audio_base64(bot_audio_path)
                if audio_base64:
                    audio_html = autoplay_audio_html(audio_base64)
                    audio_output_placeholder.markdown(audio_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to play error message audio: {e}") # Fallback to text error
        finally:
            if bot_audio_path and os.path.exists(bot_audio_path):
                try:
                    os.unlink(bot_audio_path)
                except Exception as e_unlink:
                     st.warning(f"Could not delete temp error audio file: {e_unlink}")
            # Clear the status placeholder after attempting to play error audio
            status_placeholder.empty()


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
                        if page_text: # Check if text extraction returned something
                           text += page_text + "\n" # Add newline between pages
                    except Exception as page_e:
                        st.warning(f"Could not extract text from page {page_num + 1} of PDF: {page_e}")
            return text.strip() # Remove leading/trailing whitespace
        except Exception as e:
            st.error(f"Error reading PDF '{self.pdf_path}': {e}")
            return ""

    def _create_document_embeddings(self, text):
        """Create embeddings for document text"""
        # Split text into manageable chunks (e.g., by paragraph or sentence groups)
        # Simple split by newline here, might need more sophisticated chunking for long docs
        chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
        if not chunks:
            st.warning("No text chunks found after splitting the document.")
            return [], []

        try:
            # Create embeddings
            st.info(f"Creating embeddings for {len(chunks)} text chunks...")
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
            st.info("Embeddings created successfully.")
            return chunks, np.array(embeddings) # Ensure embeddings are numpy array
        except Exception as e:
            st.error(f"Failed to create text embeddings: {e}")
            return [], []


    def _retrieve_relevant_context(self, query, top_k=3):
        """Retrieve most relevant context for the query"""
        try:
            # Retrieve stored embeddings
            chunks, embeddings = st.session_state.get('document_embeddings', ([], []))

            if not chunks or embeddings is None or embeddings.shape[0] == 0:
                 st.warning("No document embeddings available for context retrieval.")
                 return "No background context is available." # Return neutral message

            # Embed the query
            query_embedding = self.embedding_model.encode([query])[0]

            # Compute cosine similarities
            similarities = cosine_similarity([query_embedding], embeddings)[0]

            # Get top_k most similar chunks
            # Ensure top_k is not larger than the number of chunks
            num_chunks = len(chunks)
            actual_top_k = min(top_k, num_chunks)
            if actual_top_k <= 0:
                return "No relevant context found."

            # Get indices of top similarities, sorted descending
            top_indices = np.argsort(similarities)[-actual_top_k:][::-1]

            # Get the corresponding text chunks
            relevant_contexts = [chunks[i] for i in top_indices]

            # Combine the contexts
            return "\n---\n".join(relevant_contexts) # Join with separator for clarity

        except Exception as e:
            st.error(f"Error retrieving context: {e}")
            return "Error retrieving background information." # Return error message


    def _generate_response(self, query, context):
        """Generate response using Gemini with retrieved context"""
        try:
            full_prompt = f"""
            You are Satyam, an AI assistant representing Satyam Kumar.
            Use the following context about Satyam to answer the user's question.
            Speak in a friendly, conversational, and first-person style (use "I", "my", etc.).
            Keep your answers concise and engaging, suitable for a spoken conversation.

            Context about Satyam:
            ---
            {context}
            ---

            User's Question: {query}

            Your Spoken Answer (as Satyam):
            """

            # Generate response using the Gemini model
            response = self.generation_model.generate_content(full_prompt)

            # Basic check for refusal or empty response (adjust as needed based on model behavior)
            if not response.text or "cannot fulfill" in response.text.lower() or "don't know" in response.text.lower():
                 # If context was limited, provide a more specific fallback
                 if "No background context" in context or "Error retrieving" in context:
                     return "I couldn't find specific details about that in my current knowledge base. Perhaps ask me something else about my background or skills?"
                 else:
                     # Generic fallback if context was present but didn't help
                     return "Hmm, that's an interesting question! While I have some background info, I don't have the specific details to answer that right now."

            return response.text

        except Exception as e:
            st.error(f"Error generating response from AI model: {e}")
            return "I seem to be having trouble formulating a response right now. Sorry about that!"

    def _text_to_speech(self, text):
        """Convert text to speech using gTTS and return the temp file path"""
        temp_audio_path = None
        try:
            # Create a temporary file to save the audio
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_audio_path = temp_audio_file.name

            # Use gTTS to generate speech
            # Consider different tlds for different accents: 'com' (US), 'co.uk' (UK), 'com.au' (AU), 'co.in' (IN)
            tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
            tts.save(temp_audio_path)

            # Close the file handle before returning the path
            temp_audio_file.close()

            return temp_audio_path
        except Exception as e:
            st.error(f"Error generating speech with gTTS: {e}")
            # Clean up if file was created but save failed
            if temp_audio_path and os.path.exists(temp_audio_path):
                 try:
                     os.unlink(temp_audio_path)
                 except Exception: pass # Ignore cleanup error
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
