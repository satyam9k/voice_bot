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
# import audio_recorder_streamlit as audio_recorder

# # Load environment variables from .env file
# from dotenv import load_dotenv 
# load_dotenv()

# # --- Utility: Load image and convert to base64 ---
# @st.cache_data
# def get_base64_of_file(file_path):
#     with open(file_path, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# # Configuration for Gemini API
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 

# if not GEMINI_API_KEY:
#     st.error("GEMINI_API_KEY not found in the .env file. Please create a .env file with GEMINI_API_KEY=<your_api_key>")
# else:
#     genai.configure(api_key=GEMINI_API_KEY)

# # --- Custom CSS and UI Styling ---
# def apply_custom_styling():
#     """
#     Apply custom CSS styling to the Streamlit app
#     """
#     st.markdown("""
#     <style>
#     /* Reset to default Streamlit styling with some minor enhancements */
#     .stApp {
#         background-color: #ffffff;
#     }
    
#     /* Ensure text is clearly visible */
#     body, .stMarkdown, .stText, .stDataFrame, .stMetric {
#         color: #000000 !important;
#     }
    
#     /* Improved button styling */
#     .stButton>button {
#         background-color: #4CAF50;
#         color: white !important;
#         border: none;
#         padding: 10px 25px;
#         border-radius: 5px;
#         font-size: 16px;
#         font-weight: bold;
#         transition: background-color 0.3s ease;
#         width: 100%;
#     }
#     .stButton>button:hover {
#         background-color: #45a049;
#     }
    
#     /* Profile picture styling */
#     .circular-img {
#         display: block;
#         margin-left: auto;
#         margin-right: auto;
#         width: 200px;
#         height: 200px;
#         object-fit: cover;
#         border-radius: 50%;
#         border: 3px solid #4CAF50;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#     }
    
#     /* Bio card styling */
#     .bio-card {
#         background-color: #f4f4f4;
#         padding: 20px;
#         border-radius: 10px;
#         box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         max-width: 600px;
#         margin: 20px auto;
#     }
    
#     .conversation-history {
#         background-color: #f9f9f9;
#         border: 1px solid #e0e0e0;
#         border-radius: 10px;
#         padding: 15px;
#         max-height: 300px;  
#         overflow-y: auto;  
#         width: 100%;  
#         box-sizing: border-box;  
#     }
    
#     /* Ensure all headers are visible */
#     h1, h2, h3, h4, h5, h6 {
#         color: #333333 !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# class VoiceQABot:
#     def __init__(self):
#         # Placeholder PDF path (modify as needed)
#         self.pdf_path = "aboutme.pdf"
#         profile_pic_path = "sk.jpeg"
        
#         # Try to load profile picture
#         try:
#             self.profile_pic_base64 = get_base64_of_file(profile_pic_path)
#             self.profile_pic_html = f'<img src="data:image/jpeg;base64,{self.profile_pic_base64}" class="circular-img" alt="My Picture">'
#         except Exception as e:
#             self.profile_pic_html = "<p>Profile picture not found.</p>"
        
#         # Initialize conversation history
#         if 'conversation_history' not in st.session_state:
#             st.session_state.conversation_history = []
        
#         # Initialize embedding model for semantic search
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
#         # Initialize Gemini model
#         self.generation_model = genai.GenerativeModel('gemini-2.0-flash')
        
#         # Initialize session state variables
#         if 'document_text' not in st.session_state:
#             st.session_state.document_text = self.extract_text_from_pdf()
#         if 'document_embeddings' not in st.session_state:
#             st.session_state.document_embeddings = self.create_document_embeddings(st.session_state.document_text)

#     def display_ui(self):
#         """
#         Display the custom UI with profile picture and bio
#         """
#         # Apply custom styling
#         apply_custom_styling()
        
#         # Create two columns
#         left_col, right_col = st.columns([1, 1])
        
#         # Left column - Profile and Bio
#         with left_col:
#             # Display profile picture
#             st.markdown(self.profile_pic_html, unsafe_allow_html=True)
            
#             # Display bio card
#             st.markdown(
#                 """
#                 <div class="bio-card">
#                     <h2>Hi, I'm Satyam</h2>
#                     <p>
#                         As an AI and Data Science professional, I believe technology can drive positive change. With expertise in
#                         machine learning, I aim to solve complex challenges and create impactful solutions in areas like cybersecurity,
#                         healthcare, and data-driven insights, pushing innovation for a better future.
#                     </p>
#                 </div>
#                 """,
#                 unsafe_allow_html=True
#             )
        
#         # Right column - Voice Interaction
#         with right_col:
#             st.markdown("## Talk to me!")
            
#             # Voice Input using audio-recorder-streamlit
#             audio_bytes = audio_recorder.audio_recorder(
#                 text="Click to record",
#                 recording_color="#e8b62c",
#                 neutral_color="#cacaca",
#                 icon_name="microphone",
#                 icon_size="2x"
#             )
            
#             # Process audio if recorded
#             if audio_bytes:
#                 self.handle_audio_input(audio_bytes)
            
#             # Conversation History
#             st.markdown("## Conversation History")
#             conversation_container = st.container()
#             with conversation_container:
#                 st.markdown('<div class="conversation-history">', unsafe_allow_html=True)
#                 for entry in st.session_state.conversation_history:
#                     st.markdown(f"**Q:** {entry['query']}")
#                     st.markdown(f"**A:** {entry['response']}")
#                     st.markdown("---")
#                 st.markdown('</div>', unsafe_allow_html=True)

#     def handle_audio_input(self, audio_bytes):
#         """
#         Handle the audio input processing
#         """
#         try:
#             # Save audio to temporary file
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
#                 temp_audio.write(audio_bytes)
#                 temp_audio_path = temp_audio.name

#             # Use speech recognition to transcribe
#             import speech_recognition as sr
#             recognizer = sr.Recognizer()
            
#             with sr.AudioFile(temp_audio_path) as source:
#                 audio = recognizer.record(source)
#                 query = recognizer.recognize_google(audio)

#             # Remove temporary audio file
#             os.unlink(temp_audio_path)

#             if query:
#                 # Retrieve relevant context
#                 chunks, embeddings = st.session_state.document_embeddings
#                 context = self.retrieve_relevant_context(query, chunks, embeddings)

#                 # Generate response
#                 response = self.generate_response(query, context)

#                 # Add to conversation history
#                 st.session_state.conversation_history.append({
#                     'query': query,
#                     'response': response
#                 })

#                 # Generate audio response
#                 audio_file = self.text_to_speech(response)
                
#                 # Play audio (display download link for Streamlit Cloud)
#                 with open(audio_file, 'rb') as audio_file_obj:
#                     st.audio(audio_file_obj, format='audio/mp3')
                
#                 # Clean up the audio file
#                 os.unlink(audio_file)

#         except sr.UnknownValueError:
#             st.error("Sorry, I couldn't understand the audio. Please try again.")
#         except sr.RequestError:
#             st.error("Could not request results from speech recognition service.")
#         except Exception as e:
#             st.error(f"An error occurred: {e}")

#     def extract_text_from_pdf(self):
#         """
#         Extract text from local PDF file
#         """
#         try:
#             # Check if file exists
#             if not os.path.exists(self.pdf_path):
#                 st.error(f"PDF file not found at {self.pdf_path}")
#                 return ""

#             # Open and read PDF
#             with open(self.pdf_path, 'rb') as pdf_file:
#                 pdf_reader = PyPDF2.PdfReader(pdf_file)
#                 text = ""
#                 for page in pdf_reader.pages:
#                     text += page.extract_text() + "\n"
#             return text
#         except Exception as e:
#             st.error(f"Error reading PDF: {e}")
#             return ""

#     def create_document_embeddings(self, text):
#         """
#         Create embeddings for document text
#         """
#         # Split text into chunks (sentences or paragraphs)
#         chunks = text.split('\n')
        
#         # Remove empty chunks
#         chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
#         # Create embeddings
#         embeddings = self.embedding_model.encode(chunks)
        
#         return chunks, embeddings

#     def retrieve_relevant_context(self, query, chunks, embeddings):
#         """
#         Perform semantic search to find most relevant context
#         """
#         # Embed the query
#         query_embedding = self.embedding_model.encode([query])[0]
        
#         # Compute cosine similarities
#         similarities = cosine_similarity([query_embedding], embeddings)[0]
        
#         # Get top 3 most similar chunks
#         top_indices = np.argsort(similarities)[-3:][::-1]
#         relevant_contexts = [chunks[i] for i in top_indices]
        
#         return " ".join(relevant_contexts)

#     def generate_response(self, query, context):
#         """
#         Generate response using Gemini with retrieved context
#         """
#         # Construct prompt with context
#         full_prompt = f"""
#         Context: {context}

#         Question: {query}

#         Please provide a concise and engaging spoken-style answer based on the context.
#         Talk from the first person perspective *as Satyam*.
#         Speak as if you're having a natural conversation with someone.
#         If the context doesn't contain specific information,
#         give a response like, "Hmm, that's not something I'm familiar with right now, as it's not in my knowledge base. Maybe ask Satyam personally when he's around?"
#         """

#         # Generate response
#         response = self.generation_model.generate_content(full_prompt)
#         return response.text

#     def text_to_speech(self, text):
#         """
#         Convert text to speech with a male-like voice using gTTS
#         """
#         # Create a temporary file
#         temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
#         # Use English (UK) for a slightly more male-sounding voice
#         tts = gTTS(text=text, lang='en', tld='co.za')
#         tts.save(temp_audio.name)
#         return temp_audio.name

#     def run(self):
#         """
#         Main Streamlit application
#         """
#         # Display custom UI
#         self.display_ui()

# def main():
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

# Avoid import issues
try:
    import audio_recorder_streamlit as audio_recorder
except ImportError:
    st.error("Please install audio-recorder-streamlit: pip install audio-recorder-streamlit")
    audio_recorder = None

# Load environment variables from .env file
from dotenv import load_dotenv 
load_dotenv()

# Disable PyTorch logging to reduce noise
import logging
logging.getLogger('torch').setLevel(logging.ERROR)

# Prevent asyncio warnings
import asyncio
try:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except AttributeError:
    pass

# Utility function to load base64 image
@st.cache_data
def get_base64_of_file(file_path):
    """
    Convert image file to base64 encoded string
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        st.warning(f"Could not load image: {e}")
        return ""

# Custom CSS for Streamlit app
def apply_custom_styling():
    """
    Apply custom CSS styling to the Streamlit app
    """
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #e0f2f1, #fff3e0);
        background-attachment: fixed;
    }
    
    body, .stMarkdown, .stText {
        color: #000000 !important;
        font-family: 'Roboto', sans-serif;
    }
    
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
    
    .bio-card {
        background-color: rgba(255,255,255,0.8);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        max-width: 600px;
        margin: 20px auto;
    }
    
    .conversation-history {
        background-color: rgba(255,255,255,0.7);
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        max-height: 300px;  
        overflow-y: auto;  
        width: 100%;  
        box-sizing: border-box;
    }
    
    .soundwave-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 50px;
    }
    
    .soundwave-bar {
        width: 5px;
        height: 20px;
        margin: 0 3px;
        background-color: #4CAF50;
        animation: soundwave 0.7s infinite alternate;
    }
    
    @keyframes soundwave {
        0% { height: 10px; }
        100% { height: 40px; }
    }
    
    .soundwave-bar:nth-child(2) { animation-delay: 0.1s; }
    .soundwave-bar:nth-child(3) { animation-delay: 0.2s; }
    .soundwave-bar:nth-child(4) { animation-delay: 0.3s; }
    .soundwave-bar:nth-child(5) { animation-delay: 0.4s; }
    </style>
    """, unsafe_allow_html=True)

# Soundwave animation HTML
def get_soundwave_html():
    """
    Generate HTML for soundwave animation
    """
    return """
    <div class="soundwave-animation">
        <div class="soundwave-bar"></div>
        <div class="soundwave-bar"></div>
        <div class="soundwave-bar"></div>
        <div class="soundwave-bar"></div>
        <div class="soundwave-bar"></div>
    </div>
    """

class VoiceQABot:
    def __init__(self):
        # Configuration and error handling for API key
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not self.GEMINI_API_KEY:
            st.error("GEMINI_API_KEY not found in .env file. Please set it.")
            st.stop()
        
        # Configure Gemini API
        genai.configure(api_key=self.GEMINI_API_KEY)
        
        # PDF and Profile Picture paths (make these configurable)
        self.pdf_path = os.getenv("PDF_PATH", "aboutme.pdf")
        self.profile_pic_path = os.getenv("PROFILE_PIC_PATH", "sk.jpeg")
        
        # Profile Picture
        try:
            self.profile_pic_base64 = get_base64_of_file(self.profile_pic_path)
            self.profile_pic_html = f'<img src="data:image/jpeg;base64,{self.profile_pic_base64}" class="circular-img" alt="My Picture">'
        except Exception as e:
            st.warning(f"Could not load profile picture: {e}")
            self.profile_pic_html = "<p>Profile picture not found.</p>"
        
        # Initialize conversation history
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        # Initialize models
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.generation_model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            st.error(f"Error initializing models: {e}")
            st.stop()
        
        # Load document text and embeddings
        self._load_document_embeddings()

    def _load_document_embeddings(self):
        """
        Load document text and create embeddings with error handling
        """
        try:
            # Ensure document text is loaded
            if 'document_text' not in st.session_state:
                st.session_state.document_text = self._extract_text_from_pdf()
            
            # Create embeddings if not already done
            if 'document_embeddings' not in st.session_state:
                st.session_state.document_embeddings = self._create_document_embeddings(
                    st.session_state.document_text
                )
        except Exception as e:
            st.error(f"Error loading document embeddings: {e}")
            st.session_state.document_text = ""
            st.session_state.document_embeddings = ([], [])

    def display_ui(self):
        """
        Display the custom UI with profile and interaction components
        """
        # Apply styling
        apply_custom_styling()
        
        # Create two columns
        left_col, right_col = st.columns([1, 1])
        
        # Left column - Profile and Bio
        with left_col:
            # Display profile picture
            st.markdown(self.profile_pic_html, unsafe_allow_html=True)
            
            # Display bio
            st.markdown(
                """
                <div class="bio-card">
                    <h2>Hi, I'm Satyam</h2>
                    <p>
                        As an AI and Data Science professional, I believe technology can drive positive change. 
                        With expertise in machine learning, I aim to solve complex challenges and create impactful 
                        solutions in areas like cybersecurity, healthcare, and data-driven insights, pushing 
                        innovation for a better future.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Right column - Voice Interaction
        with right_col:
            st.markdown("## Talk to me!")
            
            # Voice Input 
            if audio_recorder:
                # Recording state and soundwave placeholders
                recording_state = st.empty()
                soundwave_placeholder = st.empty()
                audio_bytes = audio_recorder.audio_recorder(
                    text="Click to record",
                    recording_color="#e8b62c",
                    neutral_color="#cacaca",
                    icon_name="microphone",
                    icon_size="2x"
                )
                
                # Process audio if recorded
                if audio_bytes:
                    # Show soundwave during processing
                    soundwave_placeholder.markdown(get_soundwave_html(), unsafe_allow_html=True)
                    
                    # Process input
                    self._handle_audio_input(audio_bytes, recording_state, soundwave_placeholder)
            else:
                st.warning("Audio recording not available. Please install audio-recorder-streamlit.")
            
            # Conversation History
            st.markdown("## Conversation History")
            conversation_container = st.container()
            with conversation_container:
                st.markdown('<div class="conversation-history">', unsafe_allow_html=True)
                
                # Reverse the conversation history to show latest first
                for entry in reversed(st.session_state.conversation_history):
                    st.markdown(f"**Q:** {entry['query']}")
                    st.markdown(f"**A:** {entry['response']}")
                    st.markdown("---")
                
                st.markdown('</div>', unsafe_allow_html=True)

    def _handle_audio_input(self, audio_bytes, recording_state, soundwave_placeholder):
        """
        Handle audio input processing with comprehensive error handling
        """
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio_path = temp_audio.name

            # Transcribe audio
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio_path) as source:
                audio = recognizer.record(source)
                query = recognizer.recognize_google(audio)

            # Clean up temporary audio file
            os.unlink(temp_audio_path)

            if not query:
                # Convert error to speech
                error_response = "Sorry, I couldn't understand the audio. Please try again."
                audio_file = self._text_to_speech(error_response)
                st.audio(audio_file, format='audio/mp3')
                os.unlink(audio_file)
                soundwave_placeholder.empty()
                return

            # Process query
            context = self._retrieve_relevant_context(query)
            response = self._generate_response(query, context)

            # Update conversation history
            st.session_state.conversation_history.insert(0, {
                'query': query,
                'response': response
            })

            # Generate and auto-play audio response
            audio_file = self._text_to_speech(response)
            st.audio(audio_file, format='audio/mp3', start_time=0)
            
            # Clean up audio file
            os.unlink(audio_file)
            
            # Clear soundwave
            soundwave_placeholder.empty()

        except sr.UnknownValueError:
            # Convert error to speech
            error_response = "Sorry, I couldn't understand the audio. Please try again."
            audio_file = self._text_to_speech(error_response)
            st.audio(audio_file, format='audio/mp3')
            os.unlink(audio_file)
            soundwave_placeholder.empty()
        except sr.RequestError:
            # Convert error to speech
            error_response = "Could not request results from speech recognition service. Please try again."
            audio_file = self._text_to_speech(error_response)
            st.audio(audio_file, format='audio/mp3')
            os.unlink(audio_file)
            soundwave_placeholder.empty()
        except Exception as e:
            # Convert error to speech
            error_response = f"An unexpected error occurred: {str(e)}. Please try again."
            audio_file = self._text_to_speech(error_response)
            st.audio(audio_file, format='audio/mp3')
            os.unlink(audio_file)
            soundwave_placeholder.empty()

    def _extract_text_from_pdf(self):
        """
        Extract text from PDF with error handling
        """
        try:
            if not os.path.exists(self.pdf_path):
                st.warning(f"PDF file not found at {self.pdf_path}")
                return ""

            with open(self.pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = " ".join(page.extract_text() for page in pdf_reader.pages)
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

    def _create_document_embeddings(self, text):
        """
        Create embeddings for document text
        """
        # Split and clean text chunks
        chunks = [chunk.strip() for chunk in text.split('\n') if chunk.strip()]
        
        # Create embeddings
        embeddings = self.embedding_model.encode(chunks)
        
        return chunks, embeddings

    def _retrieve_relevant_context(self, query):
        """
        Retrieve most relevant context for the query
        """
        try:
            # Retrieve stored embeddings
            chunks, embeddings = st.session_state.document_embeddings
            
            # Embed the query
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Compute cosine similarities
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            
            # Get top 3 most similar chunks
            top_indices = np.argsort(similarities)[-3:][::-1]
            relevant_contexts = [chunks[i] for i in top_indices]
            
            return " ".join(relevant_contexts)
        except Exception as e:
            st.error(f"Error retrieving context: {e}")
            return ""

    def _generate_response(self, query, context):
        """
        Generate response using Gemini with retrieved context
        """
        try:
            full_prompt = f"""
            Context: {context}

            Question: {query}

            Please provide a concise and engaging spoken-style answer based on the context.
            Talk from the first person perspective *as Satyam*.
            Speak as if you're having a natural conversation with someone.
            If the context doesn't contain specific information,
            give a response like, "Hmm, that's not something I'm familiar with right now, as it's not in my knowledge base. Maybe ask Satyam personally when he's around?"
            """

            # Generate response
            response = self.generation_model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "I'm having trouble generating a response right now. Sorry about that!"

    def _text_to_speech(self, text):
        """
        Convert text to speech with error handling
        """
        try:
            # Create a temporary file
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            
            # Use gTTS to generate speech
            tts = gTTS(text=text, lang='en', tld='co.za')
            tts.save(temp_audio.name)
            
            return temp_audio.name
        except Exception as e:
            st.error(f"Error generating speech: {e}")
            return None

    def run(self):
        """
        Main application runner
        """
        self.display_ui()

def main():
    """
    Streamlit app entry point
    """
    bot = VoiceQABot()
    bot.run()

if __name__ == "__main__":
    main()
