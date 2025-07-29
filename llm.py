import os
import asyncio

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(
    # This is the default and can be omitted
    api_key=os.getenv("GROQ_API_KEY"),
)

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
                    "content": "You are a helpful assistant. Respond in Hindi when the user speaks in Hindi, otherwise respond in English."
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