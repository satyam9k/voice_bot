# Voice-Controlled Q&A Bot

This project is a voice-controlled question-and-answer bot that uses the Gemini AI model, speech recognition, and text-to-speech to interact with users. It can answer questions based on the content of a provided PDF document.

## Setup and Installation

1.  **Clone the Repository:**

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Gemini API Key:**
    *   Get a Gemini API key from Google AI Studio.
    *   **(Environment Variable):**
        *   Create a `.env` file in the same directory as `bot2.py`.
        *   Add the following line to `.env`, replacing `YOUR_API_KEY` with your actual key:
            ```
            GEMINI_API_KEY=YOUR_API_KEY
            ```
        *   Make sure that the `python-dotenv` package is installed.

4.  **Prepare the PDF:**
    *   Place the PDF document you want the bot to use (e.g., `about_me.pdf`) in the same directory as `bot2.py`.
    *   Update the `self.pdf_path` variable in the `VoiceQABot` class in `bot2.py` to point to your PDF file.
    *   Place the profile picture (e.g., `sk.jpeg`) in the same directory as `bot2.py`.
    *   Update the `profile_pic_path` variable in the `VoiceQABot` class in `bot2.py` to point to your image file.

5.  **Run the Application:**
    ```bash
    streamlit run bot2.py
    ```

## Usage

*   Click the "🎤 Ask a Question" button to start voice input.
*   Speak your question clearly into your microphone.
*   The bot will process your question, generate a response, and speak the response back to you.
*   The conversation history will be displayed in the right column.

## Notes

*   **Microphone:** Ensure your microphone is properly configured and accessible.
*   **Internet Connection:** An active internet connection is required for the Gemini API and speech recognition to work.
*   **PDF and Image Path:** Make sure that the paths to the pdf and the image are correct.

## Dependencies

*   `streamlit`
*   `google-generativeai`
*   `PyPDF2`
*   `speech_recognition`
*   `numpy`
*   `sentence-transformers`
*   `scikit-learn`
*   `gTTS`
*   `pygame`
* `python-dotenv` 


