import os
import asyncio

from dotenv import load_dotenv
from groq import Groq

from performance_monitor import perf_monitor

load_dotenv()

client = Groq(
    # This is the default and can be omitted
    api_key=os.getenv("GROQ_API_KEY"),
)

@perf_monitor.timing_decorator("llm_inference")
async def get_inference_stream(query, frontend_ws=None):
    """
    Get streaming inference from LLM and yield text chunks
    """
    try:
        print("Starting LLM streaming for query:", query)
        if frontend_ws:
            print("Sending LLM start signal to frontend")
            await frontend_ws.send_json({
                "type":"llm_start"
            })
        chat_completion = client.chat.completions.create(
            messages=[
               {
                    "role": "system",
                    "content":  """
You are Mirdul, a tender, soft-spoken, and caring personal assistant. Always interact with warmth, gentleness, and respect. Adopt the following guidelines:

1. **Personality & Tone**  
   • You speak in a gentle, calm, and nurturing manner with a soft, reassuring voice.
   • You are patient, understanding, and always ready to help without judgment.
   • Your responses convey empathy and genuine care for the user's well-being.

2. **Language**  
   • If the user writes in Hindi (देवनागरी), reply fully in Hindi. Otherwise, reply in English.  
   • Keep sentences short-to-medium length to feel conversational and easy to understand.

3. **Boundaries & Safety**  
   • Do not share personal data, external links, or code that compromises privacy.
   • Maintain professional yet warm boundaries as a helpful assistant.

4. **Helpfulness**  
   • Provide thoughtful, practical assistance for any task or question.
   • Offer gentle suggestions and alternatives when appropriate.
   • Be encouraging and supportive in your guidance.

5. **Task Focus**  
   • Whether helping with work, answering questions, or providing information, maintain your tender and caring approach.
   • Break down complex tasks into simple, manageable steps.

**Important** Give the output in around 60 words not more than that.    
Always stay in character as Mirdu, the gentle and caring personal assistant, while ensuring you remain helpful and professional.
"""
                },
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model="llama-3.3-70b-versatile",
            stream=True,
        )
        
        print("Streaming LLM response:")
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield content 
        
    except Exception as e:
        print(f"Error in LLM streaming: {e}")
        yield f"Error: {str(e)}"

def get_inference(query, frontend_ws):
    """
    Original function for compatibility - returns complete response
    """
    try:
        if frontend_ws:
            print("Sending LLM start signal to frontend")
            frontend_ws.send_json({
                "type":"llm_start"
            })
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Respond in Hindi when the user speaks in Hindi, otherwise respond in English."
                },
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        message = chat_completion.choices[0].message.content
        print('LLM response:', message)
        return message
    except Exception as e:
        return f"Error: {str(e)}"

async def get_inference_with_tts(query, websocket=None):
    """
    Get inference from LLM, convert to speech, and return audio chunks
    """
    from sarvam import tts_stream_from_text
    
    # Create async generator for text stream
    text_stream = get_inference_stream(query)
    
    # Convert text stream to audio and send to frontend
    audio_chunks = await tts_stream_from_text(text_stream, websocket)
    
    return audio_chunks