# import gradio as gr  
# from main import parse_user_input, get_available_slots, book_appointment, load_data, transcribe_audio  
  
# # Load availability data  
# availability, _ = load_data()  
  
# # Interactive Assistant Function  
# def assistant(user_input, audio_file=None):  
#     # If audio file is provided, transcribe it first  
#     if audio_file is not None:  
#         user_input = transcribe_audio(audio_file)  
  
#     try:  
#         # Parse user input  
#         preferences = parse_user_input(user_input)  
  
#         # Fetch available slots  
#         available_slots = get_available_slots(preferences, availability)  
  
#         if not available_slots:  
#             return f"Sorry, no matching slots found for your preferences: {preferences}"  
  
#         # Display available slots  
#         response = f"Available slots:\n"  
#         for slot in available_slots:  
#             response += f"- {slot['date']} {slot['time']}\n"  
  
#         return response  
#     except Exception as e:  
#         return f"Error: {str(e)}"  
  
# # Gradio Interface  
# interface = gr.Interface(  
#     fn=assistant,  
#     inputs=[  
#         gr.Textbox(label="Enter your appointment request"),  
#         gr.Audio(type="filepath", label="Or upload your voice request"),  # Corrected  
#     ],  
#     outputs="text",  
#     title="Hospital Appointment Scheduler",  
#     description="Interact with the assistant to schedule hospital appointments by typing or speaking your preferences.",  
# )  
  
# if __name__ == "__main__":  
#     interface.launch(share=True)  

import gradio as gr  
from main import transcribe_audio, chatbot_reply
# , parse_user_input, get_available_slots, book_appointment, load_data  
  
# Initialize conversation context  
conversation = [{"role": "system", "content": "You are a helpful assistant that helps users schedule hospital appointments."}]  
  
# Chatbot function  
def chatbot(user_input=None, audio_file=None):  
    """  
    Handles user input (text or audio), processes it, and returns the assistant's reply.  
    """  
    global conversation  
  
    # If audio file is provided, transcribe it first  
    if audio_file is not None:  
        user_input = transcribe_audio(audio_file)  
  
    if user_input:  
        # Add user input to conversation  
        conversation.append({"role": "user", "content": user_input})  
  
        # Get assistant reply from OpenAI  
        assistant_reply = chatbot_reply(conversation)  
  
        # Add assistant reply to conversation  
        conversation.append({"role": "assistant", "content": assistant_reply})  
  
        return assistant_reply  
    else:  
        return "Please provide a text or audio input."  
  
# Gradio Interface  
interface = gr.Interface(  
    fn=chatbot,  
    inputs=[  
        gr.Textbox(label="Enter your message"),  
        gr.Audio(type="filepath", label="Or upload your voice message"),  
    ],  
    outputs="text",  
    title="Hospital Appointment Chatbot",  
    description="Interact with the assistant to schedule hospital appointments. You can use text or voice input.",  
)  
  
if __name__ == "__main__":  
    interface.launch(share=True)  