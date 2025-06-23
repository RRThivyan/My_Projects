# import json  
# import os  
# from dotenv import load_dotenv  
# from google.cloud import speech  
# from openai import OpenAI  
  
# # Load environment variables  
# load_dotenv()  
# api_key = os.getenv("OPENAI_API_KEY")  
# client = OpenAI(api_key=api_key)  
  
# # Set Google Cloud credentials  
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gtts_api.json"  
  
# # Load availability and appointments from JSON files  
# def load_data():  
#     with open("data/availability.json", "r") as f:  
#         availability = json.load(f)  
#     with open("data/appointments.json", "r") as f:  
#         appointments = json.load(f)  
#     return availability, appointments  
  
# def save_appointments(appointments):  
#     with open("data/appointments.json", "w") as f:  
#         json.dump(appointments, f, indent=4)  
  
# # Parse natural language input using OpenAI's Chat API  
# def parse_user_input(user_input):  
#     messages = [  
#         {  
#             "role": "system",  
#             "content": (  
#                 "You are a helpful assistant that extracts appointment details from user input. "  
#                 "Always respond in **valid JSON format** with the following fields:\n"  
#                 "- DoctorSpecialty: The type of doctor or specialty (e.g., dentist, cardiologist).\n"  
#                 "- PreferredDate: The preferred date.\n"  
#                 "- PreferredTime: The preferred time (morning, afternoon, evening).\n"  
#                 "Example:\n"  
#                 "User input: 'I need a dentist appointment on Wednesday afternoon'\n"  
#                 "Response: {\"DoctorSpecialty\": \"dentist\", \"PreferredDate\": \"Wednesday\", \"PreferredTime\": \"afternoon\"}"  
#             )  
#         },  
#         {  
#             "role": "user",  
#             "content": f"User input: '{user_input}'\nExtract the appointment details and respond in valid JSON format."  
#         }  
#     ]  
  
#     response = client.chat.completions.create(  
#         model="gpt-4",  
#         messages=messages,  
#         max_tokens=150  
#     )  
  
#     content = response.choices[0].message.content.strip()  
  
#     try:  
#         # Attempt to parse the response as JSON  
#         details = json.loads(content)  
#         mapped_details = {  
#             "specialty": details.get("DoctorSpecialty", "").lower(),  
#             "date": details.get("PreferredDate", ""),  
#             "time": details.get("PreferredTime", "").lower()  
#         }  
#         return mapped_details  
#     except json.JSONDecodeError:  
#         raise ValueError(f"Failed to parse response from OpenAI. Raw content: {content}")  
  
# # Fetch available slots based on user preferences  
# def get_available_slots(preferences, availability):  
#     specialty = preferences["specialty"]  
#     preferred_date = preferences["date"]  
#     preferred_time = preferences["time"]  
  
#     matching_slots = []  
#     if specialty in availability:  
#         for slot in availability[specialty]:  
#             if slot["date"] == preferred_date and preferred_time in slot["time"]:  
#                 matching_slots.append(slot)  
#     return matching_slots  
  
# # Book an appointment  
# def book_appointment(preferences, slot):  
#     _, appointments = load_data()  
#     appointment = {  
#         "specialty": preferences["specialty"],  
#         "date": slot["date"],  
#         "time": slot["time"],  
#         "user": preferences.get("user", "Anonymous")  # Default to "Anonymous" if user is not provided  
#     }  
#     appointments.append(appointment)  
#     save_appointments(appointments)  
#     return appointment  
  
# # Google Speech-to-Text Integration  
# def transcribe_audio(audio_file_path):  
#     client = speech.SpeechClient()  
#     with open(audio_file_path, "rb") as audio_file:  
#         audio_content = audio_file.read()  
  
#     audio = speech.RecognitionAudio(content=audio_content)  
#     config = speech.RecognitionConfig(  
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  
#         language_code="en-US",  # Automatically detect sample rate from the WAV file  
#     )  
  
#     response = client.recognize(config=config, audio=audio)  
  
#     for result in response.results:  
#         return result.alternatives[0].transcript  
  
#     return None  



import json  
import os  
from dotenv import load_dotenv  
from google.cloud import speech  
from openai import OpenAI  
  
# Load environment variables  
load_dotenv()  
api_key = os.getenv("OPENAI_API_KEY")  
client = OpenAI(api_key=api_key)  
  
# Set Google Cloud credentials  
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "gtts_api.json"  
  
# Google Speech-to-Text Integration  
def transcribe_audio(audio_file_path):  
    """  
    Transcribes audio from a file using Google Speech-to-Text API.  
    """  
    client = speech.SpeechClient()  
    with open(audio_file_path, "rb") as audio_file:  
        audio_content = audio_file.read()  
  
    audio = speech.RecognitionAudio(content=audio_content)  
    config = speech.RecognitionConfig(  
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  
        language_code="en-US",  # Automatically detect sample rate from the WAV file  
    )  
  
    response = client.recognize(config=config, audio=audio)  
  
    for result in response.results:  
        return result.alternatives[0].transcript  
  
    return None  
  
# Chatbot Conversation Handler  
def chatbot_reply(messages):  
    """  
    Sends the conversation context to OpenAI and gets the assistant's reply.  
    """  
    response = client.chat.completions.create(  
        model="gpt-4",  
        messages=messages,  
        max_tokens=150  
    )  
  
    reply = response.choices[0].message.content.strip()  
    return reply  