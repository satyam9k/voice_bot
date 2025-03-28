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
    st.stop()
    audio_recorder = None # Ensure it's defined even on error before stop

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Disable PyTorch logging to reduce noise
import logging
logging.getLogger('torch').setLevel(logging.ERROR)

# --- State Definitions ---
# States are still used for logic flow and status messages, even if not for soundwave color
STATE_IDLE = "idle"
STATE_LISTENING = "listening" # Used briefly after recording stops
STATE_PROCESSING = "processing"
STATE_SPEAKING = "speaking" # Used while TTS is generated/played

# --- Color Definitions ---
# Only need Green for soundwave and Mic button interaction
COLOR_MIC_RECORDING = "#FF4B4B" # Red for mic icon *during* recording (component parameter)
COLOR_MIC_IDLE = "#4CAF50"      # Green for mic icon when idle/ready (component parameter)
COLOR_SOUNDWAVE_STATIC = "#4CAF50" # Static Green for the soundwave animation

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
    # Using the static green color for the soundwave bars
    soundwave_bar_color = COLOR_SOUNDWAVE_STATIC

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
        border: 4px solid {COLOR_SOUNDWAVE_STATIC}; /* Use consistent green border */
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
        border-left: 5px solid {COLOR_SOUNDWAVE_STATIC}; /* Accent border (consistent green) */
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

    /* Soundwave Animation (Static Color) */
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
        background-color: {soundwave_bar_color}; /* Static green color */
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

    /* --- Style for the Microphone Button --- */
    /* Attempt to target the button within the audio recorder component */
    /* This selector might need adjustment based on the component's HTML structure */
    div[data-testid="stAudioRecorder"] > div > button {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important; /* Adjust padding if needed */
    }}
    /* Icon color is handled by component props `recording_color` and `neutral_color` */

    </style>
    """, unsafe_allow_html=True)

def get_soundwave_html(label=""):
    """Generate HTML for soundwave animation with STATIC green color and optional label"""
    # Color is now defined in CSS, no need to pass it here.
    bars_html = "".join([
        f'<div class="soundwave-bar"></div>' # No inline style needed for color
        for _ in range(5)
    ])
    label_html = f'<p style="text-align: center; color: #555; font-size: 0.9em; height: 1.2em;">{label}</p>' if label else '<p style="height: 1.2em;"></p>' # Reserve space even if no label

    return f"""
    <div class="soundwave-container">
        {bars_html}
    </div>
    {label_html}
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

        # Initialize application state and status message
        if 'app_state' not in st.session_state:
            st.session_state.app_state = STATE_IDLE
        if 'status_message' not in st.session_state:
            st.session_state.status_message = "" # To show text like "Listening..."


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
                    st.session_state.document_embeddings = ([], np.array([])) # Ensure it's a tuple with numpy array
                    st.warning("Could not load document text, context retrieval will be limited.")

        except Exception as e:
            st.error(f"Error loading document embeddings: {e}")
            st.session_state.document_text = ""
            st.session_state.document_embeddings = ([], np.array([]))

    def _set_state(self, state, message=""):
        """Helper to update state and message, triggering a rerun."""
        st.session_state.app_state = state
        st.session_state.status_message = message
        # Streamlit automatically reruns when session_state changes.

    def display_ui(self):
        """Display the custom UI with profile and interaction components"""
        apply_custom_styling()

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

            # Get current status message for the label below the soundwave
            current_message = st.session_state.get('status_message', "")

            # Display the soundwave *always* with static green color
            soundwave_placeholder = st.empty()
            # No need to pass color, it's defined in CSS now
            soundwave_placeholder.markdown(
                get_soundwave_html(label=current_message),
                unsafe_allow_html=True
            )

            # Placeholder for the hidden audio player
            audio_output_placeholder = st.empty()

            # Voice Input Component
            if audio_recorder:
                # Use defined colors for the mic button states
                audio_bytes = audio_recorder(
                    recording_color=COLOR_MIC_RECORDING, # Red icon when recording
                    neutral_color=COLOR_MIC_IDLE,       # Green icon when idle/ready
                    icon_name="microphone",
                    icon_size="2x",
                    pause_threshold=2.0, # Seconds of silence before stopping
                    sample_rate=16000, # Common sample rate
                    key="audio_input" # Add a key for stability
                )

                # Process audio *only if* new audio bytes are received
                if audio_bytes:
                    # State is set internally within _handle_audio_input
                    # Immediately call the handler upon receiving audio bytes
                    self._handle_audio_input(audio_bytes, audio_output_placeholder)
                    # State will be reset to idle within the handler's finally block

            else:
                # This case should ideally not be reached if st.stop() works in __init__
                st.warning("Audio recording functionality requires the 'audio-recorder-streamlit' library.")

            # Conversation History
            st.markdown("## Conversation History")
            conversation_container = st.container()
            with conversation_container:
                st.markdown('<div class="conversation-history">', unsafe_allow_html=True)

                if not st.session_state.conversation_history:
                     st.markdown("<p style='color:#777;'>No conversation yet. Ask me something!</p>", unsafe_allow_html=True)
                else:
                    # Display newest first (already handled by inserting at index 0)
                    for i, entry in enumerate(st.session_state.conversation_history):
                        st.markdown(f'<p class="user-query">You ({len(st.session_state.conversation_history)-i}): {entry["query"]}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="bot-response">Bot: {entry["response"]}</p>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

    def _handle_audio_input(self, audio_bytes, audio_output_placeholder):
        """Handle audio input processing, response generation, and audio output"""
        temp_audio_path = None # Initialize to ensure cleanup check works
        bot_audio_path = None

        try:
            # 1. Set state to indicate processing starting (briefly shows 'Listening')
            self._set_state(STATE_LISTENING, "Processing audio...")

            # 2. Save recorded audio temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name

            # Update state
            self._set_state(STATE_PROCESSING, "Transcribing...")

            # 3. Transcribe audio
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                # recognizer.adjust_for_ambient_noise(source, duration=0.5) # Optional
                audio = recognizer.record(source)

            # Use recognize_google - ensure internet connection
            query = recognizer.recognize_google(audio)
            st.write(f"Heard: {query}") # Keep for debugging, or comment out

            if not query:
                raise ValueError("Transcription resulted in empty text.")

            # 4. Process query and generate response
            self._set_state(STATE_PROCESSING, "Thinking...")
            context = self._retrieve_relevant_context(query)
            response = self._generate_response(query, context)

            # 5. Update conversation history (insert at the beginning for newest first)
            st.session_state.conversation_history.insert(0, {
                'query': query,
                'response': response
            })
            # A rerun will happen due to state changes anyway, explicitly might not be needed

            # 6. Generate bot's audio response
            self._set_state(STATE_SPEAKING, "Speaking...")
            bot_audio_path = self._text_to_speech(response)
            if not bot_audio_path:
                 raise ValueError("Failed to generate speech audio.") # Handle TTS failure

            # 7. Encode bot audio to Base64
            audio_base64 = get_audio_base64(bot_audio_path)
            if not audio_base64:
                raise ValueError("Failed to encode speech audio to Base64.")

            # 8. Display hidden auto-playing audio using HTML
            audio_html = autoplay_audio_html(audio_base64)
            audio_output_placeholder.markdown(audio_html, unsafe_allow_html=True)

            # State remains 'Speaking' until the finally block resets it to 'Idle'

        except sr.UnknownValueError:
            error_msg = "Sorry, I couldn't understand the audio. Please try speaking clearly."
            st.warning(error_msg)
            self._play_error_message(error_msg, audio_output_placeholder)
        except sr.RequestError as e:
            error_msg = f"Could not connect to speech recognition service: {e}. Check internet connection."
            st.error(error_msg)
            self._play_error_message(error_msg, audio_output_placeholder)
        except ValueError as ve: # Catch specific value errors (empty transcription, TTS fail)
             error_msg = f"Processing error: {ve}. Please try again."
             st.error(error_msg)
             self._play_error_message(error_msg, audio_output_placeholder)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}. Please try again."
            st.exception(e) # Log the full traceback for debugging
            self._play_error_message(error_msg, audio_output_placeholder)
        finally:
            # Clean up temporary files
            if temp_audio_path and os.path.exists(temp_audio_path):
                try: os.unlink(temp_audio_path)
                except Exception as e_unlink: st.warning(f"Could not delete temp user audio file: {e_unlink}")
            if bot_audio_path and os.path.exists(bot_audio_path):
                try: os.unlink(bot_audio_path)
                except Exception as e_unlink: st.warning(f"Could not delete temp bot audio file: {e_unlink}")

            # Reset state to Idle AFTER processing/audio playback starts (or error handling)
            # A small delay can make the "Speaking..." label linger slightly longer if desired
            # time.sleep(0.5) # Optional small delay before resetting state
            self._set_state(STATE_IDLE, "")


    def _play_error_message(self, error_text, audio_output_placeholder):
        """Generates and plays an audio message for errors."""
        bot_audio_path = None
        try:
            # Use 'Speaking' state semantically for playing audio, even error audio
            self._set_state(STATE_SPEAKING, "Error occurred...")
            bot_audio_path = self._text_to_speech(error_text)
            if bot_audio_path:
                audio_base64 = get_audio_base64(bot_audio_path)
                if audio_base64:
                    audio_html = autoplay_audio_html(audio_base64)
                    audio_output_placeholder.markdown(audio_html, unsafe_allow_html=True)
            # State reset happens in the *calling* function's finally block, which is crucial
        except Exception as e:
            st.error(f"Failed to play error message audio: {e}") # Fallback to text error
        finally:
            # Clean up the error audio file if created
            if bot_audio_path and os.path.exists(bot_audio_path):
                try: os.unlink(bot_audio_path)
                except Exception as e_unlink: st.warning(f"Could not delete temp error audio file: {e_unlink}")
            # DO NOT reset state here.


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
        chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
        if not chunks:
            st.warning("No text chunks found after splitting the document.")
            return [], np.array([]) # Return empty list and empty numpy array

        try:
            # Create embeddings
            with st.spinner(f"Creating embeddings for {len(chunks)} text chunks..."):
                 # Specify np.float32 for compatibility if needed, but usually default is fine
                 embeddings = self.embedding_model.encode(chunks, show_progress_bar=False).astype(np.float32)
            # st.info("Embeddings created successfully.")
            return chunks, embeddings # Return chunks and numpy array
        except Exception as e:
            st.error(f"Failed to create text embeddings: {e}")
            return [], np.array([])


    def _retrieve_relevant_context(self, query, top_k=3):
        """Retrieve most relevant context for the query"""
        try:
            # Ensure correct retrieval from session state
            doc_data = st.session_state.get('document_embeddings', ([], np.array([])))
            if not isinstance(doc_data, tuple) or len(doc_data) != 2:
                 st.error("Invalid format for document embeddings in session state.")
                 return "Error accessing background information."

            chunks, embeddings = doc_data

            # Validate embeddings data
            if not chunks or not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[0] != len(chunks):
                 st.warning("No valid document embeddings available for context retrieval.")
                 return "No background context is available." # Return neutral message

            # Encode query, ensure it's also float32 if embeddings are
            query_embedding = self.embedding_model.encode([query]).astype(np.float32)[0]

            # Ensure query embedding is 1D
            if query_embedding.ndim != 1:
                raise ValueError("Query embedding is not a 1D vector.")

            # Reshape query embedding for cosine_similarity (expects 2D)
            query_embedding_2d = query_embedding.reshape(1, -1)

            # Calculate similarities
            similarities = cosine_similarity(query_embedding_2d, embeddings)[0] # Get the 1D array of similarities

            # Get top K indices
            num_chunks = embeddings.shape[0]
            actual_top_k = min(top_k, num_chunks)
            if actual_top_k <= 0:
                return "No relevant context found."

            # argsort gives indices of smallest to largest, so take the last `actual_top_k` and reverse them
            top_indices = np.argsort(similarities)[-actual_top_k:][::-1]

            relevant_contexts = [chunks[i] for i in top_indices]
            return "\n---\n".join(relevant_contexts)

        except Exception as e:
            st.error(f"Error retrieving context: {e}")
            st.exception(e) # Log full traceback for debugging context issues
            return "Error retrieving background information."


    def _generate_response(self, query, context):
        """Generate response using Gemini with retrieved context and length guidance"""
        try:
            full_prompt = f"""
            You are Satyam, an AI assistant representing Satyam Kumar.
            Use the following context about Satyam to answer the user's question.
            Speak in a friendly, conversational, and first-person style (use "I", "my", etc.).
            Keep your answers detailed enough to be informative but not overly long. Aim for a moderate length suitable for a spoken conversation (e.g., 2-4 sentences typically, but adjust based on the question's complexity). Avoid extremely short (1-word) or excessively long (paragraph+) responses unless necessary.

            Context about Satyam:
            ---
            {context}
            ---

            User's Question: {query}

            Your Spoken Answer (as Satyam):
            """

            response = self.generation_model.generate_content(full_prompt)

            # Basic check for safety/empty responses (Gemini might have specific attributes for this)
            # Accessing response.text is common for simple cases
            response_text = response.text.strip()

            # Handle potential refusals or non-answers more robustly
            if not response_text or "cannot fulfill" in response_text.lower() or "don't have information" in response_text.lower() or "don't know" in response_text.lower():
                 # Provide a more helpful fallback based on context availability
                 if "No background context is available" in context or "Error retrieving" in context:
                     return "I couldn't find specific details about that in my current knowledge base. Could you ask something else about my skills or background?"
                 else:
                     # Acknowledge the question but state limitation
                     return "That's an interesting question! While I have some information about myself, I don't have the specific details to answer that fully right now. Feel free to ask about my general experience or skills!"

            return response_text

        except Exception as e:
            st.error(f"Error generating response from AI model: {e}")
            # Provide a user-friendly error message
            return "I seem to be having a bit of trouble thinking of a response right now. Please try asking again shortly!"


    def _text_to_speech(self, text):
        """Convert text to speech using gTTS and return the temp file path"""
        temp_audio_path = None
        try:
            # Use NamedTemporaryFile for automatic cleanup potential (though we manually unlink)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                temp_audio_path = temp_audio_file.name
                tts = gTTS(text=text, lang='en', tld='co.in', slow=False) # Using 'co.in' TLD
                tts.save(temp_audio_path)
            # temp_audio_file is closed automatically here
            return temp_audio_path
        except Exception as e:
            st.error(f"Error generating speech with gTTS: {e}")
            # Attempt cleanup if file path was assigned but save failed
            if temp_audio_path and os.path.exists(temp_audio_path):
                 try: os.unlink(temp_audio_path)
                 except Exception: pass # Ignore cleanup error
            return None # Indicate failure

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

# import os
# import streamlit as st
# import google.generativeai as genai
# import PyPDF2
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from gtts import gTTS
# import tempfile
# import base64
# import speech_recognition as sr
# import time
# import io # Needed for base64 encoding of audio

# # Avoid import issues
# try:
#     # Note: audio_recorder_streamlit may need to be installed via pip
#     from audio_recorder_streamlit import audio_recorder
# except ImportError:
#     st.error("Please install audio-recorder-streamlit: pip install audio-recorder-streamlit")
#     audio_recorder = None

# # Load environment variables from .env file
# from dotenv import load_dotenv
# load_dotenv()

# # Disable PyTorch logging to reduce noise
# import logging
# logging.getLogger('torch').setLevel(logging.ERROR)

# # --- State Definitions ---
# STATE_IDLE = "idle"
# STATE_LISTENING = "listening"
# STATE_PROCESSING = "processing"
# STATE_SPEAKING = "speaking"

# # --- Color Definitions ---
# COLOR_IDLE = "#CCCCCC" # Grey for idle
# COLOR_LISTENING = "#FF4B4B" # Red for listening/recording
# COLOR_PROCESSING = "#FFA500" # Orange for thinking
# COLOR_SPEAKING = "#4CAF50" # Green for speaking

# # --- Helper Functions ---

# @st.cache_data
# def get_base64_of_file(file_path):
#     """Convert image file to base64 encoded string"""
#     try:
#         with open(file_path, "rb") as f:
#             data = f.read()
#         return base64.b64encode(data).decode()
#     except Exception as e:
#         st.warning(f"Could not load image: {e}")
#         return ""

# def get_audio_base64(audio_file_path):
#     """Convert audio file to base64 encoded string"""
#     try:
#         with open(audio_file_path, "rb") as f:
#             data = f.read()
#         return base64.b64encode(data).decode()
#     except Exception as e:
#         st.error(f"Error encoding audio to base64: {e}")
#         return None

# def autoplay_audio_html(base64_audio_string, format="mp3"):
#     """Generate HTML for auto-playing audio WITHOUT visible controls"""
#     # Browsers often restrict autoplay without user interaction.
#     # Recording audio *is* interaction, so this should usually work.
#     # Removed 'controls' attribute
#     return f"""
#     <audio autoplay style="width: 100%; display: none;"> <!-- Hidden -->
#         <source src="data:audio/{format};base64,{base64_audio_string}" type="audio/{format}">
#         Your browser does not support the audio element.
#     </audio>
#     """

# # --- Styling and Soundwave ---

# def apply_custom_styling():
#     """Apply custom CSS styling to the Streamlit app"""
#     st.markdown(f"""
#     <style>
#     /* General Styles */
#     .stApp {{
#         background: linear-gradient(135deg, #e0f2f1, #fff3e0);
#         background-attachment: fixed;
#     }}
#     body, .stMarkdown, .stText {{
#         color: #333333 !important; /* Darker text for better contrast */
#         font-family: 'Roboto', sans-serif;
#     }}

#     /* Profile Picture */
#     .circular-img {{
#         display: block;
#         margin-left: auto;
#         margin-right: auto;
#         width: 180px; /* Slightly smaller */
#         height: 180px;
#         object-fit: cover;
#         border-radius: 50%;
#         border: 4px solid #4CAF50; /* Thicker border */
#         box-shadow: 0 6px 10px rgba(0,0,0,0.15); /* Enhanced shadow */
#         margin-bottom: 15px; /* Spacing */
#     }}

#     /* Bio Card */
#     .bio-card {{
#         background-color: rgba(255,255,255,0.85); /* Slightly more opaque */
#         padding: 25px;
#         border-radius: 12px;
#         box-shadow: 0 5px 15px rgba(0,0,0,0.1);
#         max-width: 600px;
#         margin: 20px auto;
#         border-left: 5px solid #4CAF50; /* Accent border */
#     }}
#     .bio-card h2 {{
#         color: #2E7D32; /* Darker green heading */
#         margin-bottom: 10px;
#     }}
#     .bio-card p {{
#         line-height: 1.6;
#     }}

#     /* Conversation History */
#     .conversation-history {{
#         background-color: rgba(255,255,255,0.75);
#         border: 1px solid #d0d0d0;
#         border-radius: 10px;
#         padding: 15px;
#         max-height: 350px; /* Increased height */
#         overflow-y: auto;
#         width: 100%;
#         box-sizing: border-box;
#         margin-top: 15px; /* Spacing */
#     }}
#     .conversation-history .user-query {{
#         color: #0D47A1; /* Blue for user */
#         font-weight: bold;
#         margin-bottom: 3px;
#     }}
#     .conversation-history .bot-response {{
#         color: #333; /* Standard text for bot */
#         margin-bottom: 15px; /* Space below response */
#         padding-bottom: 10px; /* Space before divider */
#         border-bottom: 1px dashed #e0e0e0; /* Divider */
#     }}
#     .conversation-history .bot-response:last-child {{
#         border-bottom: none; /* No divider after last response */
#     }}

#     /* Soundwave Animation */
#     .soundwave-container {{
#         display: flex;
#         justify-content: center;
#         align-items: center;
#         height: 60px; /* Increased height */
#         margin: 10px 0;
#     }}
#     .soundwave-bar {{
#         width: 6px; /* Thicker bars */
#         height: 10px;
#         margin: 0 4px; /* More spacing */
#         /* background-color will be set by inline style */
#         border-radius: 3px; /* Rounded tops */
#         animation: soundwave 1.0s infinite ease-in-out alternate;
#     }}

#     @keyframes soundwave {{
#         0% {{ height: 8px; opacity: 0.7; }}
#         100% {{ height: 45px; opacity: 1.0; }} /* Taller and fully opaque */
#     }}

#     /* Stagger animation delays */
#     .soundwave-bar:nth-child(1) {{ animation-delay: 0s; }}
#     .soundwave-bar:nth-child(2) {{ animation-delay: 0.15s; }}
#     .soundwave-bar:nth-child(3) {{ animation-delay: 0.3s; }}
#     .soundwave-bar:nth-child(4) {{ animation-delay: 0.15s; }} /* Symmetric delay */
#     .soundwave-bar:nth-child(5) {{ animation-delay: 0s; }} /* Symmetric delay */

#     /* --- Style for the Microphone Button --- */
#     /* Attempt to target the button within the audio recorder component */
#     /* This selector might need adjustment based on the component's HTML structure */
#     div[data-testid="stAudioRecorder"] > div > button {{
#         background-color: transparent !important;
#         border: none !important;
#         box-shadow: none !important;
#         padding: 0 !important; /* Adjust padding if needed */
#     }}
#     /* Style the icon color directly if possible (already handled by component props) */
#      /* div[data-testid="stAudioRecorder"] svg {{
#          fill: #YourColor !important;
#      }} */

#     </style>
#     """, unsafe_allow_html=True)

# def get_soundwave_html(color="#CCCCCC", label=""):
#     """Generate HTML for soundwave animation with dynamic color and optional label"""
#     # Generate 5 bars with the specified background color via inline style
#     bars_html = "".join([
#         f'<div class="soundwave-bar" style="background-color: {color};"></div>'
#         for _ in range(5)
#     ])
#     label_html = f'<p style="text-align: center; color: #555; font-size: 0.9em; height: 1.2em;">{label}</p>' if label else '<p style="height: 1.2em;"></p>' # Reserve space even if no label

#     return f"""
#     <div class="soundwave-container">
#         {bars_html}
#     </div>
#     {label_html}
#     """

# # --- Main Class ---

# class VoiceQABot:
#     def __init__(self):
#         # Configuration and error handling for API key
#         self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#         if not self.GEMINI_API_KEY:
#             st.error("GEMINI_API_KEY not found in .env file. Please set it.")
#             st.stop()

#         # Configure Gemini API
#         genai.configure(api_key=self.GEMINI_API_KEY)

#         # PDF and Profile Picture paths
#         self.pdf_path = os.getenv("PDF_PATH", "aboutme.pdf")
#         self.profile_pic_path = os.getenv("PROFILE_PIC_PATH", "sk.jpeg")

#         # Profile Picture
#         try:
#             self.profile_pic_base64 = get_base64_of_file(self.profile_pic_path)
#             self.profile_pic_html = f'<img src="data:image/jpeg;base64,{self.profile_pic_base64}" class="circular-img" alt="My Picture">'
#         except Exception as e:
#             st.warning(f"Could not load profile picture: {e}")
#             self.profile_pic_html = '<p style="text-align:center; color: red;">Profile picture not found.</p>'

#         # Initialize conversation history (newest first)
#         if 'conversation_history' not in st.session_state:
#             st.session_state.conversation_history = []

#         # Initialize application state
#         if 'app_state' not in st.session_state:
#             st.session_state.app_state = STATE_IDLE
#         if 'status_message' not in st.session_state:
#             st.session_state.status_message = "" # To show text like "Listening..."


#         # Initialize models
#         try:
#             self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#             self.generation_model = genai.GenerativeModel('gemini-1.5-flash')
#         except Exception as e:
#             st.error(f"Error initializing AI models: {e}")
#             st.stop()

#         # Load document text and embeddings
#         self._load_document_embeddings()

#     def _load_document_embeddings(self):
#         """Load document text and create embeddings with error handling"""
#         try:
#             # Ensure document text is loaded
#             if 'document_text' not in st.session_state:
#                 st.session_state.document_text = self._extract_text_from_pdf()

#             # Create embeddings if not already done
#             if 'document_embeddings' not in st.session_state or not st.session_state.document_text:
#                 if st.session_state.document_text:
#                     st.session_state.document_embeddings = self._create_document_embeddings(
#                         st.session_state.document_text
#                     )
#                 else:
#                     # Handle case where PDF text couldn't be loaded
#                     st.session_state.document_embeddings = ([], [])
#                     st.warning("Could not load document text, context retrieval will be limited.")

#         except Exception as e:
#             st.error(f"Error loading document embeddings: {e}")
#             st.session_state.document_text = ""
#             st.session_state.document_embeddings = ([], [])

#     def _set_state(self, state, message=""):
#         """Helper to update state and message, triggering a rerun."""
#         st.session_state.app_state = state
#         st.session_state.status_message = message
#         # No explicit rerun needed, Streamlit handles it when session_state changes

#     def display_ui(self):
#         """Display the custom UI with profile and interaction components"""
#         apply_custom_styling()

#         # st.markdown("<h1 style='color: #FF5733;'>AI Voice Assistant</h1>",
#         #             unsafe_allow_html=True)
#         # st.markdown("---")

#         # Create two columns
#         left_col, right_col = st.columns([1, 1.2]) # Give right col slightly more space

#         # Left column - Profile and Bio
#         with left_col:
#             st.markdown(self.profile_pic_html, unsafe_allow_html=True)
#             st.markdown(
#                 """
#                 <div class="bio-card">
#                     <h2>Hi, I'm Satyam</h2>
#                     <p>
#                         An AI and Data Science enthusiast passionate about leveraging technology for positive impact.
#                         My focus includes machine learning, cybersecurity, and data-driven problem-solving.
#                         Feel free to ask me questions based on my background!
#                     </p>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )

#         # Right column - Voice Interaction and History
#         with right_col:
#             st.markdown("## Talk to me!")
#             st.markdown("Click the microphone, ask a question, and I'll respond.")

#             # Determine soundwave color and label based on current state
#             current_state = st.session_state.get('app_state', STATE_IDLE)
#             current_message = st.session_state.get('status_message', "")
#             if current_state == STATE_LISTENING:
#                 soundwave_color = COLOR_LISTENING
#             elif current_state == STATE_PROCESSING:
#                 soundwave_color = COLOR_PROCESSING
#             elif current_state == STATE_SPEAKING:
#                 soundwave_color = COLOR_SPEAKING
#             else: # IDLE
#                 soundwave_color = COLOR_IDLE

#             # Display the soundwave *always*
#             soundwave_placeholder = st.empty()
#             soundwave_placeholder.markdown(
#                 get_soundwave_html(color=soundwave_color, label=current_message),
#                 unsafe_allow_html=True
#             )

#             # Placeholder for the hidden audio player
#             audio_output_placeholder = st.empty()

#             # Voice Input Component
#             if audio_recorder:
#                 # The icon colors are controlled by the component's parameters
#                 audio_bytes = audio_recorder(
#                     recording_color=COLOR_LISTENING, # Red icon when recording
#                     neutral_color=COLOR_SPEAKING,    # Green icon when idle
#                     icon_name="microphone",
#                     icon_size="2x",
#                     pause_threshold=2.0, # Seconds of silence before stopping
#                     sample_rate=16000, # Common sample rate
#                     key="audio_input" # Add a key for stability
#                 )

#                 # Process audio *only if* new audio bytes are received
#                 if audio_bytes:
#                     # Immediately update state to listening WHILE recording is happening
#                     # (Note: audio_recorder blocks until recording finishes)
#                     # So, we set the state *after* it returns bytes
#                     self._set_state(STATE_LISTENING, "Processing audio...") # Show listening color briefly after recording stops
#                     # Process input (includes state changes for processing/speaking)
#                     self._handle_audio_input(audio_bytes, audio_output_placeholder)
#                     # State will be set back to idle in _handle_audio_input's finally block

#             else:
#                 st.warning("Audio recording functionality requires the 'audio-recorder-streamlit' library.")

#             # Conversation History
#             st.markdown("## Conversation History")
#             conversation_container = st.container()
#             with conversation_container:
#                 st.markdown('<div class="conversation-history">', unsafe_allow_html=True)

#                 if not st.session_state.conversation_history:
#                      st.markdown("<p style='color:#777;'>No conversation yet. Ask me something!</p>", unsafe_allow_html=True)
#                 else:
#                     for i, entry in enumerate(st.session_state.conversation_history):
#                         st.markdown(f'<p class="user-query">You ({len(st.session_state.conversation_history)-i}): {entry["query"]}</p>', unsafe_allow_html=True)
#                         st.markdown(f'<p class="bot-response">Bot: {entry["response"]}</p>', unsafe_allow_html=True)

#                 st.markdown('</div>', unsafe_allow_html=True)

#     def _handle_audio_input(self, audio_bytes, audio_output_placeholder):
#         """Handle audio input processing, response generation, and audio output"""
#         temp_audio_path = None # Initialize to ensure cleanup check works
#         bot_audio_path = None

#         try:
#             # 1. Save recorded audio temporarily
#             # (State is already LISTENING briefly)
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
#                 temp_audio.write(audio_bytes)
#                 temp_audio_path = temp_audio.name

#             # Update state
#             self._set_state(STATE_PROCESSING, "Transcribing...")

#             # 2. Transcribe audio
#             recognizer = sr.Recognizer()
#             with sr.AudioFile(temp_audio_path) as source:
#                 # recognizer.adjust_for_ambient_noise(source, duration=0.5) # Optional
#                 audio = recognizer.record(source)

#             # Use recognize_google - ensure internet connection
#             query = recognizer.recognize_google(audio)
#             # st.write(f"Heard: {query}") # For debugging - remove in production

#             if not query:
#                 raise ValueError("Transcription resulted in empty text.")

#             # 3. Process query and generate response
#             self._set_state(STATE_PROCESSING, "Thinking...")
#             context = self._retrieve_relevant_context(query)
#             response = self._generate_response(query, context)

#             # 4. Update conversation history (insert at the beginning)
#             st.session_state.conversation_history.insert(0, {
#                 'query': query,
#                 'response': response
#             })
#             # Force immediate UI update for history
#             # st.experimental_rerun() # Usually not needed if state changes handle it

#             # 5. Generate bot's audio response
#             self._set_state(STATE_SPEAKING, "Speaking...")
#             bot_audio_path = self._text_to_speech(response)
#             if not bot_audio_path:
#                  raise ValueError("Failed to generate speech audio.") # Handle TTS failure

#             # 6. Encode bot audio to Base64
#             audio_base64 = get_audio_base64(bot_audio_path)
#             if not audio_base64:
#                 raise ValueError("Failed to encode speech audio to Base64.")

#             # 7. Display hidden auto-playing audio using HTML
#             audio_html = autoplay_audio_html(audio_base64)
#             audio_output_placeholder.markdown(audio_html, unsafe_allow_html=True)

#             # 8. Keep "Speaking..." state until audio likely finishes?
#             #    This is tricky. We'll reset to idle in finally.
#             #    A more complex solution could use JS events, but let's keep it simple.

#         except sr.UnknownValueError:
#             error_msg = "Sorry, I couldn't understand the audio. Please try speaking clearly."
#             st.warning(error_msg)
#             self._play_error_message(error_msg, audio_output_placeholder)
#         except sr.RequestError as e:
#             error_msg = f"Could not connect to speech recognition service: {e}. Check internet connection."
#             st.error(error_msg)
#             self._play_error_message(error_msg, audio_output_placeholder)
#         except ValueError as ve: # Catch specific value errors (empty transcription, TTS fail)
#              error_msg = f"Processing error: {ve}. Please try again."
#              st.error(error_msg)
#              self._play_error_message(error_msg, audio_output_placeholder)
#         except Exception as e:
#             error_msg = f"An unexpected error occurred: {str(e)}. Please try again."
#             st.exception(e) # Log the full traceback for debugging
#             self._play_error_message(error_msg, audio_output_placeholder)
#         finally:
#             # Clean up temporary files
#             if temp_audio_path and os.path.exists(temp_audio_path):
#                 try: os.unlink(temp_audio_path)
#                 except Exception as e_unlink: st.warning(f"Could not delete temp user audio file: {e_unlink}")
#             if bot_audio_path and os.path.exists(bot_audio_path):
#                 try: os.unlink(bot_audio_path)
#                 except Exception as e_unlink: st.warning(f"Could not delete temp bot audio file: {e_unlink}")

#             # Reset state to Idle after processing (or error handling) is complete
#             # Add a small delay perhaps, otherwise the 'Speaking' state vanishes instantly
#             # time.sleep(1) # Optional small delay
#             self._set_state(STATE_IDLE, "")


#     def _play_error_message(self, error_text, audio_output_placeholder):
#         """Generates and plays an audio message for errors."""
#         bot_audio_path = None
#         try:
#             self._set_state(STATE_SPEAKING, "Error occurred...") # Use speaking state/color for error audio
#             bot_audio_path = self._text_to_speech(error_text)
#             if bot_audio_path:
#                 audio_base64 = get_audio_base64(bot_audio_path)
#                 if audio_base64:
#                     audio_html = autoplay_audio_html(audio_base64)
#                     audio_output_placeholder.markdown(audio_html, unsafe_allow_html=True)
#             # State reset happens in the calling function's finally block
#         except Exception as e:
#             st.error(f"Failed to play error message audio: {e}") # Fallback to text error
#         finally:
#             if bot_audio_path and os.path.exists(bot_audio_path):
#                 try: os.unlink(bot_audio_path)
#                 except Exception as e_unlink: st.warning(f"Could not delete temp error audio file: {e_unlink}")
#             # DO NOT reset state here, let the caller's finally block handle it.


#     def _extract_text_from_pdf(self):
#         """Extract text from PDF with error handling"""
#         try:
#             if not os.path.exists(self.pdf_path):
#                 st.warning(f"PDF file not found at {self.pdf_path}")
#                 return ""

#             text = ""
#             with open(self.pdf_path, 'rb') as pdf_file:
#                 pdf_reader = PyPDF2.PdfReader(pdf_file)
#                 if pdf_reader.is_encrypted:
#                      st.warning(f"PDF file '{self.pdf_path}' is encrypted and cannot be read.")
#                      return ""
#                 for page_num, page in enumerate(pdf_reader.pages):
#                     try:
#                         page_text = page.extract_text()
#                         if page_text: # Check if text extraction returned something
#                            text += page_text + "\n" # Add newline between pages
#                     except Exception as page_e:
#                         st.warning(f"Could not extract text from page {page_num + 1} of PDF: {page_e}")
#             return text.strip() # Remove leading/trailing whitespace
#         except Exception as e:
#             st.error(f"Error reading PDF '{self.pdf_path}': {e}")
#             return ""

#     def _create_document_embeddings(self, text):
#         """Create embeddings for document text"""
#         chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
#         if not chunks:
#             st.warning("No text chunks found after splitting the document.")
#             return [], []

#         try:
#             # Create embeddings
#             with st.spinner(f"Creating embeddings for {len(chunks)} text chunks..."):
#                  embeddings = self.embedding_model.encode(chunks, show_progress_bar=False) # Disable bar in spinner
#             #st.info("Embeddings created successfully.")
#             return chunks, np.array(embeddings) # Ensure embeddings are numpy array
#         except Exception as e:
#             st.error(f"Failed to create text embeddings: {e}")
#             return [], []


#     def _retrieve_relevant_context(self, query, top_k=3):
#         """Retrieve most relevant context for the query"""
#         try:
#             chunks, embeddings = st.session_state.get('document_embeddings', ([], []))

#             if not chunks or not isinstance(embeddings, np.ndarray) or embeddings.shape[0] == 0:
#                  st.warning("No document embeddings available for context retrieval.")
#                  return "No background context is available." # Return neutral message

#             query_embedding = self.embedding_model.encode([query])[0]
#             similarities = cosine_similarity([query_embedding], embeddings)[0]
#             num_chunks = len(chunks)
#             actual_top_k = min(top_k, num_chunks)
#             if actual_top_k <= 0:
#                 return "No relevant context found."

#             top_indices = np.argsort(similarities)[-actual_top_k:][::-1]
#             relevant_contexts = [chunks[i] for i in top_indices]
#             return "\n---\n".join(relevant_contexts)

#         except Exception as e:
#             st.error(f"Error retrieving context: {e}")
#             return "Error retrieving background information."


#     def _generate_response(self, query, context):
#         """Generate response using Gemini with retrieved context"""
#         try:
#             full_prompt = f"""
#             You are Satyam, an AI assistant representing Satyam Kumar.
#             Use the following context about Satyam to answer the user's question.
#             Speak in a friendly, conversational, and first-person style (use "I", "my", etc.).
#             Keep your answers concise and engaging, suitable for a spoken conversation.

#             Context about Satyam:
#             ---
#             {context}
#             ---

#             User's Question: {query}

#             Your Spoken Answer (as Satyam):
#             """

#             response = self.generation_model.generate_content(full_prompt)

#             # Check for refusals or empty responses
#             response_text = response.text.strip()
#             if not response_text or any(phrase in response_text.lower() for phrase in ["cannot fulfill", "don't have information", "don't know"]):
#                  if "No background context" in context or "Error retrieving" in context:
#                      return "I couldn't find specific details about that in my current knowledge base. Perhaps ask me something else about my background or skills?"
#                  else:
#                      return "Hmm, that's an interesting question! While I have some background info, I don't have the specific details to answer that right now."

#             return response_text

#         except Exception as e:
#             st.error(f"Error generating response from AI model: {e}")
#             return "I seem to be having trouble formulating a response right now. Sorry about that!"

#     def _text_to_speech(self, text):
#         """Convert text to speech using gTTS and return the temp file path"""
#         temp_audio_path = None
#         try:
#             temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#             temp_audio_path = temp_audio_file.name
#             tts = gTTS(text=text, lang='en', tld='co.in', slow=False)
#             tts.save(temp_audio_path)
#             temp_audio_file.close()
#             return temp_audio_path
#         except Exception as e:
#             st.error(f"Error generating speech with gTTS: {e}")
#             if temp_audio_path and os.path.exists(temp_audio_path):
#                  try: os.unlink(temp_audio_path)
#                  except Exception: pass
#             return None

#     def run(self):
#         """Main application runner"""
#         self.display_ui()

# # --- Entry Point ---

# def main():
#     """Streamlit app entry point"""
#     st.set_page_config(layout="wide", page_title="Satyam's AI Voice Assistant")
#     bot = VoiceQABot()
#     bot.run()

# if __name__ == "__main__":
#     main()
