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
    st.stop() # Stop execution if recorder is not available

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Disable PyTorch logging to reduce noise
import logging
logging.getLogger('torch').setLevel(logging.ERROR)

# --- State Definitions ---
STATE_IDLE = "idle"
STATE_LISTENING = "listening" # Represents processing after recording stops
STATE_PROCESSING = "processing"
STATE_SPEAKING = "speaking"

# --- Color Definitions ---
COLOR_IDLE = "#CCCCCC" # Grey for idle soundwave
COLOR_LISTENING = "#FF4B4B" # Red for listening/recording (ICON color during recording)
COLOR_PROCESSING = "#FFA500" # Orange for thinking soundwave
COLOR_SPEAKING = "#4CAF50" # Green for speaking soundwave (and idle ICON color)

# --- Helper Functions ---

@st.cache_data # Cache image loading
def get_base64_of_file(file_path):
    """Convert image file to base64 encoded string"""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.warning(f"File not found: {file_path}")
        return ""
    except Exception as e:
        st.warning(f"Could not load image ({file_path}): {e}")
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
    # Browsers often restrict autoplay without user interaction.
    # Recording audio *is* interaction, so this should usually work.
    # Removed 'controls' attribute
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
        border: 4px solid {COLOR_SPEAKING}; /* Use speaking color for border */
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
        border-left: 5px solid {COLOR_SPEAKING}; /* Accent border */
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
        margin-bottom: 20px; /* Space below history */
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
        margin: 10px 0 0 0; /* Top margin, less bottom margin */
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

    /* --- Style for the Microphone Button (Component specific) --- */
    /* Attempt to target the button within the audio recorder */
    div[data-testid="stAudioRecorder"] button {{
        border: none !important;
        box-shadow: none !important;
        /* Color is handled by component props (recording_color, neutral_color) */
        /* Padding/margin adjustments might be needed depending on visual */
        margin-top: 10px; /* Add some space above the icon */
    }}
    div[data-testid="stAudioRecorder"] {{
        display: flex;          /* Use flexbox */
        justify-content: center; /* Center horizontally */
        width: 100%;             /* Take full width if needed */
    }}


    </style>
    """, unsafe_allow_html=True)

def get_soundwave_html(color="#CCCCCC", label=""):
    """Generate HTML for soundwave animation with dynamic color and optional label"""
    # Generate 5 bars with the specified background color via inline style
    bars_html = "".join([
        f'<div class="soundwave-bar" style="background-color: {color};"></div>'
        for _ in range(5)
    ])
    # Ensure the label takes up space even if empty to prevent layout shifts
    label_html = f'<p style="text-align: center; color: #555; font-size: 0.9em; height: 1.2em; margin-top: 5px; margin-bottom: 10px;">{label} </p>' # Added nbsp for empty space

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
        self.profile_pic_base64 = get_base64_of_file(self.profile_pic_path)
        if self.profile_pic_base64:
            self.profile_pic_html = f'<img src="data:image/jpeg;base64,{self.profile_pic_base64}" class="circular-img" alt="My Picture">'
        else:
             st.warning(f"Could not load profile picture from: {self.profile_pic_path}")
             self.profile_pic_html = '<div style="text-align:center; color: red; height: 180px; border: 2px dashed red; border-radius: 50%; padding: 10px;">Profile picture not found.</div>'


        # Initialize conversation history (newest first)
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []

        # Initialize application state
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
                if not st.session_state.document_text:
                     st.warning(f"No text extracted from '{self.pdf_path}'. Context-based answers will be limited.")

            # Create embeddings if not already done or if text was empty
            if 'document_embeddings' not in st.session_state or not st.session_state.document_text:
                if st.session_state.document_text:
                    st.session_state.document_embeddings = self._create_document_embeddings(
                        st.session_state.document_text
                    )
                else:
                    # Handle case where PDF text couldn't be loaded
                    st.session_state.document_embeddings = ([], np.array([])) # Use empty numpy array
                    st.warning("Document embeddings could not be created (no text).")
            # Ensure embeddings are loaded correctly
            elif isinstance(st.session_state.document_embeddings, tuple) and len(st.session_state.document_embeddings) == 2:
                pass # Looks okay
            else:
                 st.warning("Re-initializing document embeddings due to unexpected format.")
                 if st.session_state.document_text:
                    st.session_state.document_embeddings = self._create_document_embeddings(
                        st.session_state.document_text
                    )
                 else:
                    st.session_state.document_embeddings = ([], np.array([]))


        except Exception as e:
            st.error(f"Error loading document embeddings: {e}")
            st.session_state.document_text = ""
            st.session_state.document_embeddings = ([], np.array([]))

    def _set_state(self, state, message=""):
        """Helper to update state and message, triggering a rerun."""
        # Check if state or message actually changed to avoid unnecessary reruns
        if st.session_state.app_state != state or st.session_state.status_message != message:
            st.session_state.app_state = state
            st.session_state.status_message = message
            # Streamlit reruns automatically when session_state changes.
            # No explicit rerun needed here, prevents potential infinite loops.

    def display_ui(self):
        """Display the custom UI with profile and interaction components"""
        apply_custom_styling()

        st.markdown("<h1 style='text-align: center; color: #2E7D32;'>AI Voice Assistant</h1>",
                    unsafe_allow_html=True)
        #st.markdown("---") # Removed horizontal rule

        # Create two columns
        left_col, right_col = st.columns([1, 1.2], gap="large") # Give right col slightly more space

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
            st.markdown("<h2 style='text-align: center;'>Talk to me!</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Click the microphone, ask a question, and I'll respond.</p>", unsafe_allow_html=True)

            # --- Interaction Area ---
            interaction_container = st.container()
            with interaction_container:
                # Determine soundwave color and label based on current state
                current_state = st.session_state.get('app_state', STATE_IDLE)
                current_message = st.session_state.get('status_message', "")
                if current_state == STATE_PROCESSING:
                    soundwave_color = COLOR_PROCESSING
                elif current_state == STATE_SPEAKING:
                    soundwave_color = COLOR_SPEAKING
                elif current_state == STATE_LISTENING: # Use processing color for the brief post-record phase
                    soundwave_color = COLOR_PROCESSING
                else: # IDLE
                    soundwave_color = COLOR_IDLE

                # Display the soundwave *always*
                soundwave_placeholder = st.empty()
                soundwave_placeholder.markdown(
                    get_soundwave_html(color=soundwave_color, label=current_message),
                    unsafe_allow_html=True
                )

                # Voice Input Component - Centered
                col1, col2, col3 = st.columns([1, 1, 1]) # Create columns for centering
                with col2: # Use the middle column
                    if audio_recorder:
                        # Icon colors controlled by component parameters
                        audio_bytes = audio_recorder(
                            text="", # No text label on the button itself
                            recording_color=COLOR_LISTENING, # Red icon when recording
                            neutral_color=COLOR_SPEAKING,    # Green icon when idle/ready
                            icon_name="microphone",
                            icon_size="3x", # Slightly larger icon
                            pause_threshold=2.0, # Seconds of silence before stopping
                            sample_rate=16000, # Common sample rate
                            key="audio_input" # Add a key for stability
                        )
                    else:
                        st.warning("Audio recording functionality requires the 'audio-recorder-streamlit' library.")
                        audio_bytes = None


            # Placeholder for the hidden audio player (outside the interaction container)
            audio_output_placeholder = st.empty()

            # Process audio *only if* new audio bytes are received and we are idle
            # Prevents processing if user clicks again while processing/speaking
            if audio_bytes and st.session_state.app_state == STATE_IDLE:
                 # Process input (includes state changes for processing/speaking)
                self._handle_audio_input(audio_bytes, audio_output_placeholder)
                # State will be set back to idle in _handle_audio_input's finally block

            # Conversation History
            st.markdown("<h2 style='text-align: center; margin-top: 30px;'>Conversation History</h2>", unsafe_allow_html=True)
            conversation_container = st.container()
            with conversation_container:
                st.markdown('<div class="conversation-history">', unsafe_allow_html=True)

                if not st.session_state.conversation_history:
                     st.markdown("<p style='color:#777; text-align: center;'>No conversation yet. Ask me something!</p>", unsafe_allow_html=True)
                else:
                    # Display newest first
                    for i, entry in enumerate(st.session_state.conversation_history):
                        st.markdown(f'<p class="user-query">You ({len(st.session_state.conversation_history)-i}): {entry["query"]}</p>', unsafe_allow_html=True)
                        st.markdown(f'<p class="bot-response">Bot: {entry["response"]}</p>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)


    def _handle_audio_input(self, audio_bytes, audio_output_placeholder):
        """Handle audio input processing, response generation, and audio output"""
        temp_audio_path = None # Initialize to ensure cleanup check works
        bot_audio_path = None
        query = "" # Initialize query

        try:
            # 0. Update state: Indicate processing has started (after recording icon stops)
            self._set_state(STATE_LISTENING, "Processing audio...")

            # 1. Save recorded audio temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name

            # 2. Transcribe audio
            self._set_state(STATE_PROCESSING, "Transcribing...")
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                try:
                    # recognizer.adjust_for_ambient_noise(source, duration=0.5) # Optional: Can sometimes help
                    audio = recognizer.record(source)
                    # Use recognize_google - ensure internet connection
                    query = recognizer.recognize_google(audio)
                    st.write(f"Heard: {query}") # For debugging - keep for now
                except sr.WaitTimeoutError:
                     raise sr.UnknownValueError("No speech detected in the audio.") # Treat timeout as no speech


            if not query:
                raise ValueError("Transcription resulted in empty text.")

            # 3. Process query and generate response
            self._set_state(STATE_PROCESSING, "Thinking...")
            context = self._retrieve_relevant_context(query)
            response = self._generate_response(query, context)

            # 4. Update conversation history (insert at the beginning for newest first)
            st.session_state.conversation_history.insert(0, {
                'query': query,
                'response': response
            })
            # Limit history length (e.g., keep last 10 interactions)
            max_history = 10
            if len(st.session_state.conversation_history) > max_history:
                st.session_state.conversation_history = st.session_state.conversation_history[:max_history]


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

            # 8. Keep "Speaking..." state. It will be reset in the finally block.
            #    We could add a slight delay here if needed, but the automatic
            #    reset in finally is usually sufficient.

        except sr.UnknownValueError:
            error_msg = "Sorry, I couldn't understand the audio. Please try speaking clearly."
            st.warning(error_msg)
            if query: # Add query to history even if response failed
                 st.session_state.conversation_history.insert(0, {'query': query, 'response': f"Bot Error: {error_msg}"})
            self._play_error_message(error_msg, audio_output_placeholder)
        except sr.RequestError as e:
            error_msg = f"Connection error with speech recognition: {e}."
            st.error(error_msg)
            if query:
                 st.session_state.conversation_history.insert(0, {'query': query, 'response': f"Bot Error: {error_msg}"})
            self._play_error_message(error_msg, audio_output_placeholder)
        except ValueError as ve: # Catch specific value errors (empty transcription, TTS fail)
             error_msg = f"Processing error: {ve}. Please try again."
             st.error(error_msg)
             if query:
                 st.session_state.conversation_history.insert(0, {'query': query, 'response': f"Bot Error: {error_msg}"})
             self._play_error_message(error_msg, audio_output_placeholder)
        except Exception as e:
            error_msg = f"An unexpected error occurred: {str(e)}. Please try again."
            st.exception(e) # Log the full traceback for debugging
            if query:
                 st.session_state.conversation_history.insert(0, {'query': query, 'response': f"Bot Error: {error_msg}"})
            self._play_error_message(error_msg, audio_output_placeholder)
        finally:
            # Clean up temporary files
            if temp_audio_path and os.path.exists(temp_audio_path):
                try: os.unlink(temp_audio_path)
                except Exception as e_unlink: st.warning(f"Could not delete temp user audio file: {e_unlink}")
            if bot_audio_path and os.path.exists(bot_audio_path):
                try: os.unlink(bot_audio_path)
                except Exception as e_unlink: st.warning(f"Could not delete temp bot audio file: {e_unlink}")

            # Reset state to Idle *after* processing (or error handling) is complete.
            # No artificial delay needed, state change will trigger rerun.
            self._set_state(STATE_IDLE, "")
            # Explicitly trigger one more rerun AFTER setting state to idle
            # to ensure the UI (soundwave/label) fully updates back to idle state.
            # Use st.rerun() which replaced st.experimental_rerun()
            st.rerun()


    def _play_error_message(self, error_text, audio_output_placeholder):
        """Generates and plays an audio message for errors."""
        bot_audio_path = None
        try:
            # Use speaking state/color for error audio feedback
            self._set_state(STATE_SPEAKING, "Error occurred...")
            bot_audio_path = self._text_to_speech(error_text)
            if bot_audio_path:
                audio_base64 = get_audio_base64(bot_audio_path)
                if audio_base64:
                    audio_html = autoplay_audio_html(audio_base64)
                    # Display the audio player immediately
                    audio_output_placeholder.markdown(audio_html, unsafe_allow_html=True)
            else:
                # If TTS fails for the error message, just log it. The state will be reset by caller.
                st.error("Additionally, failed to generate audio for the error message.")

            # State reset happens in the calling function's finally block

        except Exception as e:
            st.error(f"Failed to play error message audio: {e}") # Fallback to text error
        finally:
            # Clean up temp error audio file if it exists
            if bot_audio_path and os.path.exists(bot_audio_path):
                try: os.unlink(bot_audio_path)
                except Exception as e_unlink: st.warning(f"Could not delete temp error audio file: {e_unlink}")
            # DO NOT reset state here; let the main handler's finally block do it.


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
                     # Attempt default empty password decryption
                     try:
                        pdf_reader.decrypt('')
                     except Exception:
                         st.warning(f"PDF file '{self.pdf_path}' is encrypted and could not be decrypted.")
                         return ""

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text: # Check if text extraction returned something
                           text += page_text.replace('\u0000', '') + "\n" # Remove null bytes and add newline
                    except Exception as page_e:
                        st.warning(f"Could not extract text from page {page_num + 1} of PDF: {page_e}")
            return text.strip() # Remove leading/trailing whitespace
        except Exception as e:
            st.error(f"Error reading PDF '{self.pdf_path}': {e}")
            return ""

    @st.cache_data(show_spinner=False) # Cache embeddings based on text
    def _create_document_embeddings(_self, text): # Use _self because self is not available in cached functions implicitly
        """Create embeddings for document text (Cached)"""
        chunks = [chunk.strip() for chunk in text.split('\n') if len(chunk.strip()) > 10] # Only embed meaningful chunks
        if not chunks:
            st.warning("No text chunks found after splitting the document (min length 10).")
            return [], np.array([])

        try:
            # Get the embedding model (assuming it's initialized in __init__)
            # We need to access it without using 'self' directly in cached func
            # A bit hacky: access via class instance if available or re-init temporarily
            # Better: Pass model as arg, but streamlit cache doesn't handle complex objects well.
            # For simplicity here, assume it's available via an instance or re-init if needed
            # Note: This part might need refinement depending on how caching interacts with class instances.
            # A common pattern is to initialize models outside the class or pass them explicitly.
            # Let's stick to the simple approach for now.
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # Re-init inside if needed

            st.info(f"Creating embeddings for {len(chunks)} text chunks...") # Use info instead of spinner inside cache
            embeddings = embedding_model.encode(chunks, show_progress_bar=True) # Can show bar here
            st.info("Embeddings created successfully.")
            return chunks, np.array(embeddings) # Ensure embeddings are numpy array
        except Exception as e:
            st.error(f"Failed to create text embeddings: {e}")
            return [], np.array([])


    def _retrieve_relevant_context(self, query, top_k=3):
        """Retrieve most relevant context for the query"""
        try:
            # Retrieve from session state
            doc_data = st.session_state.get('document_embeddings')
            if not doc_data or not isinstance(doc_data, tuple) or len(doc_data) != 2:
                st.warning("Document embeddings not found or invalid in session state.")
                return "No background context is available."

            chunks, embeddings = doc_data

            # Validate embeddings format
            if not isinstance(chunks, list) or not isinstance(embeddings, np.ndarray):
                 st.warning("Invalid format for chunks or embeddings in session state.")
                 return "No background context is available."

            if not chunks or embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[0] != len(chunks):
                 st.warning("No document embeddings available or mismatch between chunks and embeddings.")
                 return "No background context is available." # Return neutral message

            query_embedding = self.embedding_model.encode([query])[0]
            similarities = cosine_similarity([query_embedding], embeddings)[0]

            num_chunks = len(chunks)
            actual_top_k = min(top_k, num_chunks)
            if actual_top_k <= 0:
                return "No relevant context found." # Should not happen if chunks exist

            # Get indices of top_k highest similarities
            top_indices = np.argsort(similarities)[-actual_top_k:][::-1]

            # Filter based on a minimum similarity threshold (optional but recommended)
            similarity_threshold = 0.3 # Adjust as needed
            relevant_contexts = []
            for i in top_indices:
                if similarities[i] >= similarity_threshold:
                    relevant_contexts.append(chunks[i])
                # else: # Debugging
                #     print(f"Chunk skipped (similarity {similarities[i]:.2f} < {similarity_threshold}): {chunks[i][:50]}...")


            if not relevant_contexts:
                 return "I looked through my background info, but couldn't find specific details relevant to your question."

            return "\n---\n".join(relevant_contexts)

        except Exception as e:
            st.error(f"Error retrieving context: {e}")
            return "Error retrieving background information."


    def _generate_response(self, query, context):
        """Generate response using Gemini with retrieved context"""
        try:
            # More robust prompt
            full_prompt = f"""
            You are Satyam, an AI assistant embodying Satyam Kumar, designed for voice interaction.
            Your personality is friendly, helpful, and concise. You are speaking directly to the user.
            Use the first person ("I", "my", "me").
            Keep responses relatively short and conversational, suitable for being spoken aloud.
            Answer the user's question based *primarily* on the provided context about Satyam.
            If the context doesn't contain the answer, politely state that you don't have that specific information in your background details, rather than making something up or refusing generically.

            Provided Context about Satyam:
            ---
            {context}
            ---

            User's Question: "{query}"

            Your Spoken Answer (as Satyam):
            """

            # Configure safety settings (optional, adjust as needed)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            response = self.generation_model.generate_content(
                full_prompt,
                safety_settings=safety_settings
                )

            # Check response validity and safety feedback
            try:
                 response_text = response.text.strip()
            except ValueError:
                # If the response was blocked due to safety settings
                st.warning(f"Response blocked due to safety settings. Prompt feedback: {response.prompt_feedback}")
                response_text = "I'm sorry, I can't provide a response to that question based on my safety guidelines."
            except Exception as resp_e:
                 st.error(f"Error accessing generated response text: {resp_e}")
                 response_text = "I encountered an issue generating the response text."


            # Handle empty or generic refusal-like responses specifically
            if not response_text:
                 return "I seem to be having trouble formulating a response right now. Could you try asking differently?"
            elif any(phrase in response_text.lower() for phrase in ["cannot fulfill your request", "don't have information", "i do not know", "i don't know that", "based on the text provided"]):
                 # Check if context was actually available
                 if "No background context" in context or "Error retrieving" in context or "couldn't find specific details" in context:
                     return "I couldn't find specific details about that in my current knowledge base. Perhaps ask me something else about my background or skills?"
                 else:
                     # Context was provided, but model still couldn't answer
                     return "Hmm, that's an interesting question! While I looked through my background details, I don't have the specific information to answer that precisely."

            return response_text

        except Exception as e:
            st.error(f"Error generating response from AI model: {e}")
            return "I seem to be having trouble formulating a response right now. Sorry about that!"

    def _text_to_speech(self, text):
        """Convert text to speech using gTTS and return the temp file path"""
        temp_audio_path = None
        if not text: # Prevent gTTS error on empty string
             st.warning("Attempted to generate speech for empty text.")
             return None
        try:
            # Use io.BytesIO instead of NamedTemporaryFile initially
            mp3_fp = io.BytesIO()
            tts = gTTS(text=text, lang='en', tld='com', slow=False) # Use 'com' TLD for potentially better quality/stability
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0) # Rewind the buffer

            # Now save the buffer content to a temporary file for playback/encoding
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            temp_audio_path = temp_audio_file.name
            temp_audio_file.write(mp3_fp.read())
            temp_audio_file.close()
            mp3_fp.close()

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
    # Initialize the bot only once
    if 'voice_bot' not in st.session_state:
        st.session_state.voice_bot = VoiceQABot()

    # Run the bot's display logic
    st.session_state.voice_bot.run()


if __name__ == "__main__":
    main()
