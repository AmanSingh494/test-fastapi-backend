import asyncio
import base64
from sarvamai import AsyncSarvamAI, AudioOutput
import websockets
from dotenv import load_dotenv
load_dotenv()
import os
import time
# Add this to your main.py imports
from performance_monitor import perf_monitor

SARVAM_API_KEY=os.getenv('SARVAM_API_KEY')

async def check_sarvam_connection_health(sarvam_ws):
    """Check if Sarvam WebSocket connection is healthy"""
    try:
        # Check if the connection is still open
        if hasattr(sarvam_ws, '_websocket') and sarvam_ws._websocket.closed:
            print("Sarvam WebSocket is closed")
            return False
            
        # Try to send a ping if available
        if hasattr(sarvam_ws, 'ping'):
            try:
                await asyncio.wait_for(sarvam_ws.ping(), timeout=5.0)
                print("Sarvam WebSocket ping successful")
                return True
            except Exception as ping_error:
                print(f"Sarvam WebSocket ping failed: {ping_error}")
                return False
        
        print("Sarvam WebSocket appears to be open")
        return True
        
    except Exception as e:
        print(f"Error checking Sarvam connection health: {e}")
        return False
    
@perf_monitor.timing_decorator("tts_processing")
async def tts_stream_from_text(text_stream, frontend_ws=None, sarvam_ws=None, task_id=None, tts_start=None, processing_start_time = None):
    """
    Convert streaming text to audio and optionally send to frontend via websocket
    Buffers text chunks and sends to Sarvam when they reach minimum size
    """
    try:
        if frontend_ws:
            await frontend_ws.send_json({
                "type": "tts_start",
                "task_id": task_id
            })
            
        if not sarvam_ws:
            print("Error: No Sarvam WebSocket connection provided")
            return []
            
        is_healthy = await check_sarvam_connection_health(sarvam_ws)
        if not is_healthy:
            print("Sarvam WebSocket connection is not healthy")
            return []

        # Buffer for accumulating text chunks
        text_buffer = ""
        min_chunk_size = 100  # Minimum characters before sending to Sarvam
        max_chunk_size = 400  # Maximum to stay under 500 char recommendation
        full_text = ""  # Keep track of all text for completion message
        chunk_count = 0
        audio_chunks = []

        print(f"📝 Starting text streaming with buffer size: {min_chunk_size}-{max_chunk_size} chars")

        # streaming text to sarvam {start}
        # async for text_chunk in text_stream:
        #     if text_chunk:
        #         # print(f"📥 Received text chunk: '{text_chunk}' ({len(text_chunk)} chars)")
        #         text_buffer += text_chunk
        #         full_text += text_chunk
                
        #         # Check if buffer has enough content or reached max size
        #         if len(text_buffer) >= min_chunk_size or len(text_buffer) >= max_chunk_size:
        #             # Send the buffered text to Sarvam
        #             try:
        #                 # Split at word boundaries if buffer is too large
        #                 text_to_send = text_buffer
        #                 if len(text_buffer) > max_chunk_size:
        #                     # Find last space before max_chunk_size
        #                     last_space = text_buffer.rfind(' ', 0, max_chunk_size)
        #                     if last_space > min_chunk_size:  # Only split if we still have meaningful text
        #                         text_to_send = text_buffer[:last_space].strip()
        #                         text_buffer = text_buffer[last_space:].strip()  # Keep remainder
        #                     else:
        #                         text_to_send = text_buffer[:max_chunk_size]
        #                         text_buffer = text_buffer[max_chunk_size:]
        #                 else:
        #                     text_buffer = ""  # Clear buffer since we're sending all of it
                        
        #                 if text_to_send.strip():  # Only send non-empty text
        #                     print(f"🚀 Sending buffered text to Sarvam: '{text_to_send}' ({len(text_to_send)} chars)")
        #                     await sarvam_ws.convert(text_to_send)
        #                     print(f"✅ Successfully sent {len(text_to_send)} chars to Sarvam")
                        
        #             except Exception as convert_error:
        #                 print(f"❌ Error in convert(): {convert_error}")
        #                 raise convert_error
        
        # # Send any remaining text in buffer
        # if text_buffer.strip():
        #     try:
        #         print(f"🚀 Sending final buffered text: '{text_buffer}' ({len(text_buffer)} chars)")
        #         await sarvam_ws.convert(text_buffer)
        #         print(f"✅ Successfully sent final {len(text_buffer)} chars to Sarvam")
        #     except Exception as convert_error:
        #         print(f"❌ Error sending final buffer: {convert_error}")
        #         raise convert_error

        # Flush after all text has been sent
        # print(f"📤 Total text processed: '{full_text}' ({len(full_text)} chars)")
        # try:
        #     await sarvam_ws.flush()
        #     print("✅ TTS buffer flushed successfully")
        # except Exception as flush_error:
        #     print(f"❌ Error in flush(): {flush_error}")
        #     raise flush_error
        # {end}

        full_text = ""
        async for text_chunk in text_stream:
                if text_chunk:
                    full_text += text_chunk
                    # Send chunk to TTS (you can also send incrementally)
                    print(f"Sending to TTS: {text_chunk}", end="")
        if not sarvam_ws:
                print("Error: No Sarvam WebSocket connection provided")
                return []
        is_healthy = await check_sarvam_connection_health(sarvam_ws)
        if not is_healthy:
            print("Sarvam WebSocket connection is not healthy")
            return []
        # Send complete text to TTS
        print('full text: ',full_text)
        try:
            await sarvam_ws.convert(full_text)
            print("Successfully sent text to Sarvam")
        except Exception as convert_error:
            print(f"Error in convert(): {convert_error}")
            raise convert_error

        try:
            await sarvam_ws.flush()
            print("TTS buffer flushed successfully")
        except Exception as flush_error:
            print(f"Error in flush(): {flush_error}")
            raise flush_error
        
            # Collect and send audio chunks
        chunk_count = 0
        audio_chunks = []
        
        last_chunk_time = time.time()
        silence_threshold = 2.0  # 2 seconds of silence = completion
        try:
            async with asyncio.timeout(15.0):  # 15 second timeout
                async for message in sarvam_ws:
                    # Check for cancellation during audio processing
                    # if tts_request.cancelled:
                    #     print(f"TTS request {task_id} cancelled during audio collection")
                    #     break
                    current_task = asyncio.current_task()
                    if current_task and current_task.cancelled():
                        print(f"🛑 Task cancelled during audio collection for {task_id}")
                        break
                    if isinstance(message, AudioOutput):
                        chunk_count += 1
                        last_chunk_time = time.time()

                        audio_chunk = base64.b64decode(message.data.audio)
                        audio_chunks.append(audio_chunk)
                        
                        # Send audio chunk to frontend
                        if frontend_ws:
                            try:
                                print(f"📻 Sending audio chunk {chunk_count} to frontend after {(time.time()-tts_start)*1000:.1f}ms of tts start and after {(time.time() - processing_start_time)*1000:.1f}ms of process start")
                                await frontend_ws.send_json({
                                    "type": "audio_chunk",
                                    "data": message.data.audio,
                                    "chunk_number": chunk_count,
                                    "format": "mp3",
                                    "task_id": task_id,
                                    "can_be_interrupted": True
                                })
                                print(f"📻 Sent audio chunk {chunk_count} to frontend after {(time.time() - tts_start)*1000:.1f}ms")
                            except Exception as e:
                                print(f"❌ Error sending audio chunk: {e}")
                                
                    elif hasattr(message, 'error') or 'ErrorResponse' in str(type(message)):
                        # Handle Sarvam ErrorResponse
                        print(f"🚨 Sarvam Error Response detected!")
                        print(f"   Type: {type(message)}")
                        
                        error_details = {}
                        for attr in ['error', 'message', 'code', 'details', 'status', 'reason', 'data']:
                            if hasattr(message, attr):
                                value = getattr(message, attr)
                                error_details[attr] = value
                                print(f"   {attr}: {value}")
                        
                        if frontend_ws:
                            await frontend_ws.send_json({
                                "type": "tts_error",
                                "task_id": task_id,
                                "error_details": error_details
                            })
                        break
                        
                    else:
                        print(f"❓ Unexpected message: {type(message)} - {message}")

                    current_time = time.time()
                    if current_time - last_chunk_time > silence_threshold:
                        print(f"TTS completion detected for task {task_id} (silence threshold)")
                        break                        
        except asyncio.TimeoutError:
            print(f"⏱️ Timeout waiting for audio chunks from Sarvam for task id: {task_id}")
            if frontend_ws:
                await frontend_ws.send_json({
                    "type": "error",
                    "message": "Timeout waiting for audio response",
                    "task_id": task_id
                })
        except asyncio.CancelledError:
            print(f"🛑 Audio processing cancelled for task {task_id}")
            raise

        print(f"✅ Generated {chunk_count} audio chunks")
        
        # Send completion message
        if frontend_ws:
            try:
                await frontend_ws.send_json({
                    "type": "audio_complete",
                    "total_chunks": chunk_count,
                    "text": full_text,
                })
            except Exception as e:
                print(f"Error sending completion message: {e}")
        
        return audio_chunks

    except asyncio.CancelledError:
        print(f"🛑 TTS streaming was cancelled for task {task_id}")
        raise
    except Exception as e:
        print(f"❌ Error in TTS streaming for task {task_id}: {e}")
        if frontend_ws and hasattr(frontend_ws, 'client_state') and frontend_ws.client_state.value == 1:
            print("Sending message to frontend")
            try:
                await frontend_ws.send_json({
                    "type": "error",
                    "message": f"TTS Error: {str(e)}",
                    "task_id": task_id
                })
            except:
                pass
        return []

async def send_sarvam_pings(ws):
    """
    Sends periodic ping messages to the Sarvam WebSocket to keep the connection alive.
    """
    try:
        while True:
            await asyncio.sleep(10)  # Send a ping every 5 seconds
            if not ws._websocket.closed: # Check if the underlying WebSocket is still open
                await ws.ping()
                print("Sent Sarvam ping.")
            else:
                print("Sarvam WebSocket already closed, stopping pings.")
                break
    except asyncio.CancelledError:
        print("Sarvam ping task was cancelled.")
    except Exception as e:
        print(f"Error in Sarvam ping task: {e}")
