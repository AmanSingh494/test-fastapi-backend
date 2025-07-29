# main.py
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json
import logging
import os
from datetime import datetime
from dotenv import load_dotenv
from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveTranscriptionEvents,
    LiveOptions,
)
import threading
from llm import get_inference, get_inference_stream
from sarvamai import AsyncSarvamAI, AudioOutput
from sarvam import tts_stream_from_text, send_sarvam_pings
import base64  
import time

from performance_monitor import perf_monitor

load_dotenv()
# Initialize FastAPI app
app = FastAPI()

app.state.frontend_ws_map = {} # Maps frontend_ws_id to the actual WebSocket object
app.state.deepgram_connections = {} # Maps frontend_ws_id to Deepgram connection
app.state.sarvam_connections = {} # Maps frontend_ws_id to sarvam AI TTS connection
app.state.sarvam_clients = {} # Maps frontend_ws_id to sarvam AI client
app.state.sarvam_context_managers = {} # Maps frontend_ws_id to sarvam AI context manager
app.state.connection_tasks = {} 
app.state.tts_tasks = {}

current_utterance_transcript = []
utterance_lock = threading.Lock() 

# Configure logging for better visibility in the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

if not DEEPGRAM_API_KEY:
    logger.error("DEEPGRAM_API_KEY not found in environment variables. Please set it in your .env file.")

if not SARVAM_API_KEY:
    logger.error("SARVAM_API_KEY not found in environment variables. Please set it in your .env file.")

async def handle_deepgram_message(result, **kwargs):
    """Handle messages from Deepgram"""
    global current_utterance_transcript
    try:
        logger.debug(f"Handling Deepgram message: {result}")
        frontend_ws_id = kwargs.get('frontend_ws_id')
        frontend_ws = app.state.frontend_ws_map.get(frontend_ws_id)
        
        if not frontend_ws:
            logger.error(f"Frontend WS not found for {frontend_ws_id}")
            return

        sentence = result.channel.alternatives[0].transcript
        is_final = result.is_final

        if len(sentence) == 0:
            return
        if frontend_ws_id in app.state.tts_tasks:
            tts_task = app.state.tts_tasks[frontend_ws_id]
            if not tts_task.done():
                logger.info(f"🛑 INTERRUPTING: Ongoing TTS detected, user is speaking")
                await frontend_ws.send_json({
                    "type": 'audio_interrupt'
                })

                # Cancel the current TTS task
                tts_task.cancel()
                try:
                    await asyncio.wait_for(tts_task, timeout=1.0)
                    logger.info(f"TTS task interrupted successfully for {frontend_ws_id}")
                except asyncio.CancelledError:
                    logger.info(f"TTS task was cancelled for {frontend_ws_id}")
                except asyncio.TimeoutError:
                    logger.warning(f"TTS task cancellation timed out for {frontend_ws_id}")
                except Exception as e:
                    logger.error(f"Error cancelling TTS task for {frontend_ws_id}: {e}")
        if(sentence and is_final):
            # Lock to safely update the current utterance transcript
            with utterance_lock:
                current_utterance_transcript.append(sentence)

        logger.info(f"Deepgram transcript: '{sentence}' (final: {is_final}) for frontend {frontend_ws_id}")
        
        await frontend_ws.send_json({
            "type": "stt_result",
            "text": sentence,
            "is_final": is_final,
            "confidence": result.channel.alternatives[0].confidence if hasattr(result.channel.alternatives[0], 'confidence') else None
        })

    except Exception as e:
        logger.error(f"Error handling Deepgram message for {frontend_ws_id}: {e}")

async def handle_deepgram_error(error, **kwargs):
    """Handle errors from Deepgram"""
    try:
        frontend_ws_id = kwargs.get('frontend_ws_id')
        frontend_ws = app.state.frontend_ws_map.get(frontend_ws_id)
        
        if frontend_ws:
            await frontend_ws.send_json({
                "type": "error",
                "message": f"Deepgram Error: {error}"
            })
        
        logger.error(f"Deepgram error for {frontend_ws_id}: {error}")
    except Exception as e:
        logger.error(f"Error handling Deepgram error for {frontend_ws_id}: {e}")

async def handle_deepgram_utterance_end(utterance_end, **kwargs):
    """Handle utterance end from Deepgram - LIGHTWEIGHT VERSION"""
    global current_utterance_transcript
    try:
        frontend_ws_id = kwargs.get('frontend_ws_id')
        frontend_ws = app.state.frontend_ws_map.get(frontend_ws_id)
        
        if not frontend_ws:
            logger.error(f"Frontend WS not found for {frontend_ws_id}")
            return

        with utterance_lock:
            full_transcript = " ".join(current_utterance_transcript).strip()
            current_utterance_transcript = []

        if not full_transcript:
            logger.warning(f"Received empty utterance end for {frontend_ws_id}")
            return
        
        logger.info(f"Processing utterance: '{full_transcript}' for {frontend_ws_id}")
        if frontend_ws_id in app.state.tts_tasks:
                old_task = app.state.tts_tasks[frontend_ws_id]
                if not old_task.done():
                    logger.info(f"Cancelling previous TTS task for {frontend_ws_id}")
                    old_task.cancel()
                    try:
                        # Wait for the cancellation to complete with timeout
                        await asyncio.wait_for(old_task, timeout=2.0)
                        logger.info(f"Previous TTS task cancelled successfully for {frontend_ws_id}")
                    except asyncio.CancelledError:
                        logger.info(f"Previous TTS task was cancelled for {frontend_ws_id}")
                    except asyncio.TimeoutError:
                        logger.warning(f"Previous TTS task cancellation timed out for {frontend_ws_id}")
                    except Exception as e:
                        logger.error(f"Error cancelling previous TTS task for {frontend_ws_id}: {e}")  
        # Send utterance to frontend immediately
        await frontend_ws.send_json({
            "type": "utterance_end",
            "text": full_transcript,
            "message": "Processing your request..."
        })
        
        # Queue TTS processing as a background task (don't await it here!)
        tts_task = asyncio.create_task(
            process_utterance_with_tts(frontend_ws_id, full_transcript)
        )

        app.state.tts_tasks[frontend_ws_id] = tts_task
        logger.info(f"Queued TTS processing task for {frontend_ws_id}")

    except Exception as e:
        logger.error(f"Error handling Deepgram utterance end for {frontend_ws_id}: {e}")


@perf_monitor.timing_decorator("deepgram_connection_create")
def create_deepgram_connection(frontend_ws_id: str):
    """Create a Deepgram connection for real-time transcription"""
    try:
        # Create Deepgram client
        config = DeepgramClientOptions(options={"keepalive": "true"})
        deepgram = DeepgramClient(DEEPGRAM_API_KEY, config)
        
        # Create live transcription connection
        dg_connection = deepgram.listen.websocket.v("1")
        
        # Configure live transcription options
        options = LiveOptions(
            model="nova-2",
            language="hi",
            smart_format=True,
            interim_results=True,
            utterance_end_ms=1500,
        )
        
        logger.info(f"Starting Deepgram connection with options: {options}")
        
        # Get the main event loop reference
        main_loop = asyncio.get_event_loop()
        
        # Create thread-safe wrapper functions for event handlers
        def on_transcript(self, result, **kwargs):
            try:
                # Schedule coroutine on main event loop from thread
                future = asyncio.run_coroutine_threadsafe(
                    handle_deepgram_message(result, frontend_ws_id=frontend_ws_id), 
                    main_loop
                )
                # Don't wait for completion to avoid blocking
                logger.debug(f"Scheduled transcript handler for {frontend_ws_id}")
            except Exception as e:
                logger.error(f"Error scheduling transcript handler for {frontend_ws_id}: {e}")
        
        def on_error(self, error, **kwargs):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    handle_deepgram_error(error, frontend_ws_id=frontend_ws_id), 
                    main_loop
                )
                logger.debug(f"Scheduled error handler for {frontend_ws_id}")
            except Exception as e:
                logger.error(f"Error scheduling error handler for {frontend_ws_id}: {e}")

        def on_utterance_end(self, utterance_end, **kwargs):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    handle_deepgram_utterance_end(utterance_end, frontend_ws_id=frontend_ws_id), 
                    main_loop
                )
                logger.debug(f"Scheduled utterance end handler for {frontend_ws_id}")
            except Exception as e:
                logger.error(f"Error scheduling utterance end handler for {frontend_ws_id}: {e}")
                
        def on_open(self, open, **kwargs):
            logger.info(f"Deepgram connection opened for {frontend_ws_id}")
            
        def on_close(self, close, **kwargs):
            logger.warning(f"Deepgram connection closed for {frontend_ws_id}")
            
        # Set up event handlers
        dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        dg_connection.on(LiveTranscriptionEvents.Close, on_close)
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        dg_connection.on(LiveTranscriptionEvents.UtteranceEnd, on_utterance_end)

        # Start the connection (synchronous call, not awaitable)
        logger.info(f"Attempting to start Deepgram connection for {frontend_ws_id}")
        start_result = dg_connection.start(options)
        logger.info(f"Deepgram start result: {start_result} for {frontend_ws_id}")
        
        if start_result:
            logger.info(f"Deepgram connection started successfully for frontend {frontend_ws_id}")
            # Store the connection
            app.state.deepgram_connections[frontend_ws_id] = dg_connection
            return dg_connection
        else:
            logger.error(f"Failed to start Deepgram connection for frontend {frontend_ws_id}")
            return None
            
    except Exception as e:
        logger.error(f"Exception in create_deepgram_connection for {frontend_ws_id}: {e}", exc_info=True)
        return None
async def close_sarvam_connection(frontend_ws_id: str):
    """Properly close the Sarvam AI connection"""
    logger.info(f"Starting Sarvam connection cleanup for {frontend_ws_id}")
    
    try:
        # Close the context manager first (most important)
        if hasattr(app.state, 'sarvam_context_managers') and frontend_ws_id in app.state.sarvam_context_managers:
            context_manager = app.state.sarvam_context_managers[frontend_ws_id]
            logger.info(f"Found Sarvam context manager for {frontend_ws_id}, attempting to close...")
            try:
                # Add timeout to prevent hanging
                await asyncio.wait_for(
                    context_manager.__aexit__(None, None, None), 
                    timeout=5.0  # 5 second timeout
                )
                logger.info(f"Successfully closed Sarvam context manager for {frontend_ws_id}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout closing Sarvam context manager for {frontend_ws_id}")
            except Exception as e:
                logger.error(f"Error closing Sarvam context manager for {frontend_ws_id}: {e}")
            
            # Remove from state even if closing failed
            try:
                del app.state.sarvam_context_managers[frontend_ws_id]
                logger.info(f"Removed Sarvam context manager from state for {frontend_ws_id}")
            except KeyError:
                logger.warning(f"Sarvam context manager already removed for {frontend_ws_id}")
        else:
            logger.info(f"No Sarvam context manager found for {frontend_ws_id}")
        
        # Clean up the WebSocket connection reference
        if hasattr(app.state, 'sarvam_connections') and frontend_ws_id in app.state.sarvam_connections:
            try:
                del app.state.sarvam_connections[frontend_ws_id]
                logger.info(f"Cleaned up Sarvam WebSocket connection reference for {frontend_ws_id}")
            except KeyError:
                logger.warning(f"Sarvam connection already removed for {frontend_ws_id}")
        else:
            logger.info(f"No Sarvam connection found for {frontend_ws_id}")
        
        # Clean up client
        if hasattr(app.state, 'sarvam_clients') and frontend_ws_id in app.state.sarvam_clients:
            client = app.state.sarvam_clients[frontend_ws_id]
            logger.info(f"Found Sarvam client for {frontend_ws_id}, attempting to close...")
            try:
                if hasattr(client, 'close'):
                    await asyncio.wait_for(client.close(), timeout=3.0)  # 3 second timeout
                    logger.info(f"Successfully closed Sarvam client for {frontend_ws_id}")
                else:
                    logger.info(f"Sarvam client for {frontend_ws_id} has no close method")
            except asyncio.TimeoutError:
                logger.error(f"Timeout closing Sarvam client for {frontend_ws_id}")
            except Exception as e:
                logger.error(f"Error closing Sarvam client for {frontend_ws_id}: {e}")
            
            # Remove from state even if closing failed
            try:
                del app.state.sarvam_clients[frontend_ws_id]
                logger.info(f"Removed Sarvam client from state for {frontend_ws_id}")
            except KeyError:
                logger.warning(f"Sarvam client already removed for {frontend_ws_id}")
        else:
            logger.info(f"No Sarvam client found for {frontend_ws_id}")
            
    except Exception as e:
        logger.error(f"Error in Sarvam connection cleanup for {frontend_ws_id}: {e}")
    
    logger.info(f"Completed Sarvam connection cleanup for {frontend_ws_id}")



@perf_monitor.timing_decorator("sarvam_connection_create")
async def create_sarvam_connection(frontend_ws_id: str):
    """Create a Sarvam AI TTS connection"""
    try:
        logger.info(f"trying to create Sarvam AI connection for {frontend_ws_id}")
        client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
        logger.info(f"Sarvam AI client created for {frontend_ws_id}")
        # Get the context manager
        context_manager = client.text_to_speech_streaming.connect(model="bulbul:v2")
        logger.info(f"Sarvam AI context manager created for {frontend_ws_id}")
        # Manually enter the context manager
        ws_connection = await context_manager.__aenter__()
        logger.info(f"Sarvam AI WebSocket connection established for {frontend_ws_id}")
        # Configure the TTS settings
        await ws_connection.configure(target_language_code="hi-IN", speaker="manisha", enable_preprocessing=True)
        logger.info(f"TTS configured for {frontend_ws_id}")
        # Store the connection and context manager in the state map
        app.state.sarvam_connections[frontend_ws_id] = ws_connection
        app.state.sarvam_context_managers = getattr(app.state, 'sarvam_context_managers', {})
        app.state.sarvam_context_managers[frontend_ws_id] = context_manager
        app.state.sarvam_clients = getattr(app.state, 'sarvam_clients', {})
        app.state.sarvam_clients[frontend_ws_id] = client
        
        # Comment out ping task for now since send_sarvam_pings might not be implemented
        # ping_task = asyncio.create_task(send_sarvam_pings(ws_connection))
        logger.info(f"Created Sarvam AI connection for {frontend_ws_id}")
        return ws_connection
    
    except Exception as e:
        logger.error(f"Failed to create Sarvam AI connection for {frontend_ws_id}: {e}", exc_info=True)
        # Clean up any partial state
        try:
            if hasattr(app.state, 'sarvam_connections') and frontend_ws_id in app.state.sarvam_connections:
                del app.state.sarvam_connections[frontend_ws_id]
            if hasattr(app.state, 'sarvam_context_managers') and frontend_ws_id in app.state.sarvam_context_managers:
                del app.state.sarvam_context_managers[frontend_ws_id]
            if hasattr(app.state, 'sarvam_clients') and frontend_ws_id in app.state.sarvam_clients:
                del app.state.sarvam_clients[frontend_ws_id]
        except Exception as cleanup_error:
            logger.error(f"Error during Sarvam connection cleanup: {cleanup_error}")
        return None
async def async_text_generator(text: str):
    """Convert string to async generator for TTS"""
    yield text

@perf_monitor.timing_decorator("utterance_processing_full")
async def process_utterance_with_tts(frontend_ws_id: str, full_transcript: str):
    """Process utterance with LLM and TTS in background task"""
    try:
        current_task = asyncio.current_task()
        task_id = id(current_task)  # Memory address as unique ID
        
        logger.info(f"Starting LLM + TTS task {task_id} for {frontend_ws_id}")
        frontend_ws = app.state.frontend_ws_map.get(frontend_ws_id)
        sarvam_ws = app.state.sarvam_connections.get(frontend_ws_id)
        
        if not frontend_ws:
            logger.error(f"Frontend WS not found for TTS processing {frontend_ws_id}")
            return
            
        if not sarvam_ws:
            logger.error(f"Sarvam WS not found for TTS processing {frontend_ws_id}")
            await frontend_ws.send_json({
                "type": "error",
                "message": "TTS service not available"
            })
            return

        # Get LLM response
        logger.info(f"Getting LLM response for: '{full_transcript}'")
         # Your existing code with timing points
    
        utterance_start = time.time()
        print(f"🎯 Starting utterance processing: '{full_transcript}'")
    
        llm_start = time.time()
        llm_res = get_inference_stream(full_transcript, frontend_ws=frontend_ws)
        llm_time = time.time() - llm_start
        print(f"📝 LLM response time: {llm_time*1000:.1f}ms")

        # Send LLM response to frontend first
        await frontend_ws.send_json({
            "type": "llm_complete",
            "utterance": full_transcript
        })
        
        # Now process TTS with timeout
        logger.info(f"Starting TTS processing for {frontend_ws_id}")
        try:
            tts_start = time.time()
            # Add timeout to TTS processing
            audio_chunks = await asyncio.wait_for(
                tts_stream_from_text(
                    text_stream=llm_res, 
                    frontend_ws=frontend_ws, 
                    sarvam_ws=sarvam_ws,
                    task_id=task_id
                ),
                timeout=30.0  # 30 second timeout
            )
            logger.info(f"TTS completed for {frontend_ws_id}: {len(audio_chunks)} chunks")
            tts_time = time.time() - tts_start
            print(f"🎵 TTS processing time: {tts_time*1000:.1f}ms")
    
            total_time = time.time() - utterance_start
            print(f"🏁 Total utterance processing: {total_time*1000:.1f}ms")
            
        except asyncio.TimeoutError:
            logger.error(f"TTS processing timed out for {frontend_ws_id}")
            await frontend_ws.send_json({
                "type": "error",
                "message": "TTS processing timed out"
            })
        except Exception as tts_error:
            logger.error(f"TTS processing failed for {frontend_ws_id}: {tts_error}")
            await frontend_ws.send_json({
                "type": "error",
                "message": f"TTS processing failed: {str(tts_error)}"
            })
            
        # Send final completion message
        await frontend_ws.send_json({
            "type": "processing_complete",
            "utterance": full_transcript,
            # "llm": llm_res
        })
        
    except Exception as e:
        logger.error(f"Error in TTS processing task for {frontend_ws_id}: {e}")
        if frontend_ws_id in app.state.frontend_ws_map:
            frontend_ws = app.state.frontend_ws_map[frontend_ws_id]
            await frontend_ws.send_json({
                "type": "error",
                "message": f"Processing failed: {str(e)}"
            })


async def setup_connections_background(frontend_ws_id: str, websocket: WebSocket):
    """Background task to set up Deepgram and Sarvam connections"""
    try:
        logger.info(f"Starting background connection setup for {frontend_ws_id}")
        
        # Create Deepgram connection
        logger.info(f"Creating Deepgram connection for {frontend_ws_id}")
        dg_connection = create_deepgram_connection(frontend_ws_id)
        if not dg_connection:
            await websocket.send_json({
                "type": "error",
                "message": "Failed to connect to Deepgram."
            })
            logger.error(f"Failed to connect to Deepgram for frontend {frontend_ws_id}")
            return False
            
        # Create Sarvam connection
        logger.info(f"Creating Sarvam connection for {frontend_ws_id}")
        sarvam_connection = await create_sarvam_connection(frontend_ws_id)
        if not sarvam_connection:
            await websocket.send_json({
                "type": "error", 
                "message": "Failed to connect to Sarvam AI TTS."
            })
            logger.error(f"Failed to connect to Sarvam AI for frontend {frontend_ws_id}")
            # Clean up Deepgram if Sarvam fails
            if dg_connection:
                try:
                    dg_connection.finish()
                except:
                    pass
            return False
        
        # Send success message
        await websocket.send_json({
            "type": "connections_ready",
            "message": "All connections established successfully"
        })
        
        logger.info(f"Background connection setup completed for {frontend_ws_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error in background connection setup for {frontend_ws_id}: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Connection setup failed: {str(e)}"
        })
        return False

async def cleanup_connections_background(frontend_ws_id: str):
    """Background task to clean up connections"""
    try:
        logger.info(f"Starting background cleanup for {frontend_ws_id}")
        if hasattr(app.state, 'tts_tasks') and frontend_ws_id in app.state.tts_tasks:
            tts_task = app.state.tts_tasks[frontend_ws_id]
            if not tts_task.done():
                logger.info(f"Cancelling running TTS task for {frontend_ws_id}")
                tts_task.cancel()
                try:
                    await tts_task
                except asyncio.CancelledError:
                    pass
            del app.state.tts_tasks[frontend_ws_id]
        # Close Deepgram connection
        if frontend_ws_id in app.state.deepgram_connections:
            try:
                dg_connection = app.state.deepgram_connections[frontend_ws_id]
                dg_connection.finish()  # Synchronous method
                del app.state.deepgram_connections[frontend_ws_id]
                logger.info(f"Deepgram connection closed in background for {frontend_ws_id}")
            except Exception as e:
                logger.error(f"Error closing Deepgram connection in background for {frontend_ws_id}: {e}")

        # Close Sarvam AI connection with timeout
        try:
            await asyncio.wait_for(
                close_sarvam_connection(frontend_ws_id), 
                timeout=10.0
            )
            logger.info(f"Sarvam cleanup completed in background for {frontend_ws_id}")
        except asyncio.TimeoutError:
            logger.error(f"Sarvam cleanup timed out in background for {frontend_ws_id}")
            # Force cleanup
            for state_key in ['sarvam_connections', 'sarvam_context_managers', 'sarvam_clients']:
                if hasattr(app.state, state_key) and frontend_ws_id in getattr(app.state, state_key):
                    try:
                        del getattr(app.state, state_key)[frontend_ws_id]  # Fixed this line
                    except:
                        pass
        except Exception as e:
            logger.error(f"Error during Sarvam cleanup in background for {frontend_ws_id}: {e}")
        
        # Clean up state maps
        for state_key in ['frontend_ws_map', 'pending_transcripts']:
            if hasattr(app.state, state_key) and frontend_ws_id in getattr(app.state, state_key):
                try:
                    del getattr(app.state, state_key)[frontend_ws_id]
                    logger.info(f"Cleaned up {state_key} for {frontend_ws_id}")
                except:
                    pass
        
        # Remove the cleanup task reference
        if hasattr(app.state, 'connection_tasks') and frontend_ws_id in app.state.connection_tasks:
            del app.state.connection_tasks[frontend_ws_id]
            
        logger.info(f"Background cleanup completed for {frontend_ws_id}")
        
    except Exception as e:
        logger.error(f"Error in background cleanup for {frontend_ws_id}: {e}")

@app.get("/")
async def get():
   return "Backend up and running - WebSocket endpoint available at /ws/audio"

# --- WebSocket Endpoint for Audio Streaming ---
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Frontend WebSocket connection accepted at /ws/audio")
    
    # Create audio_recordings directory if it doesn't exist
    os.makedirs("audio_recordings", exist_ok=True)
    
    # Store the frontend WebSocket in a shared map so event handlers can access it
    frontend_ws_id = str(id(websocket)) # Unique ID for this frontend connection
    app.state.frontend_ws_map[frontend_ws_id] = websocket

    
    # Connection status flags
    connections_ready = False
    
    logger.info(f"Starting Deepgram real-time transcription for {frontend_ws_id}")
    try:
        # Check if API key is available
        if not DEEPGRAM_API_KEY or not SARVAM_API_KEY:
            await websocket.send_json({
                "type": "error",
                "message": "API keys not configured on backend."
            })
            return

        # Start background connection setup
        setup_task = asyncio.create_task(
            setup_connections_background(frontend_ws_id, websocket)
        )
        app.state.connection_tasks[frontend_ws_id] = setup_task
        
        # Wait for connections to be ready (with timeout)
        try:
            connections_ready = await asyncio.wait_for(setup_task, timeout=30.0)
        except asyncio.TimeoutError:
            logger.error(f"Connection setup timed out for {frontend_ws_id}")
            await websocket.send_json({
                "type": "error",
                "message": "Connection setup timed out"
            })
            return
        
        if not connections_ready:
            logger.error(f"Failed to establish connections for {frontend_ws_id}")
            return

        
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                audio_chunk = message["bytes"]
                # Log the size of the received audio chunk
                # logger.info(f"Received audio chunk: {len(audio_chunk)} bytes (WebM/Opus).")
                # Send audio directly to Deepgram (no buffering needed)
                # Get Deepgram connection
                dg_connection = app.state.deepgram_connections.get(frontend_ws_id)
                if dg_connection:
                    try:
                        logger.debug(f"Sending audio chunk to Deepgram for {frontend_ws_id}")
                        dg_connection.send(audio_chunk)
                        logger.debug(f"Sent {len(audio_chunk)} bytes to Deepgram for {frontend_ws_id}")
                    except Exception as e:
                        logger.error(f"Failed to send audio to Deepgram for {frontend_ws_id}: {e}")
                else:
                    logger.error(f"No Deepgram connection available for {frontend_ws_id}, cannot send audio.")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Deepgram connection not established."
                    })
                    continue
                
                # Send a simple acknowledgment back to the frontend
                # await websocket.send_json({"type": "ack", "message": f"Received {len(audio_chunk)} bytes of audio."})

            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    # Log any JSON message received (like the 'audio_end' signal)
                    logger.info(f"Received JSON message: {data}")
                    
                    if data.get('type') == 'audio_end':
                        # Send any remaining audio and close Deepgram connection
                        if dg_connection:
                            try:
                                dg_connection.finish()  # Remove await - this is a synchronous method
                                logger.info(f"Deepgram connection finished for {frontend_ws_id}")
                            except Exception as e:
                                logger.error(f"Failed to finish Deepgram connection for {frontend_ws_id}: {e}")
                        
                        logger.info(f"Audio stream ended for {frontend_ws_id}")
                    
                    elif data.get('type') == 'stop_transcription':
                        # Explicit request to stop transcription and close Deepgram connection
                        logger.info(f"Received stop_transcription request from {frontend_ws_id}")
                        
                        if dg_connection:
                            try:
                                dg_connection.finish()
                                logger.info(f"Deepgram connection explicitly closed for {frontend_ws_id}")
                                
                                # Clear from state maps
                                if frontend_ws_id in app.state.deepgram_connections:
                                    del app.state.deepgram_connections[frontend_ws_id]
                                
                                # Reset variable to None to prevent double cleanup
                                dg_connection = None
                                
                            except Exception as e:
                                logger.error(f"Failed to properly close Deepgram connection for {frontend_ws_id}: {e}")
                    
                    elif data.get('type') == 'start_transcription':
                        # Request to start/restart transcription
                        logger.info(f"Received start_transcription request from {frontend_ws_id}")
                        
                        # If no Deepgram connection exists, create one
                        if not dg_connection:
                            dg_connection = create_deepgram_connection(frontend_ws_id)
                            if dg_connection:
                                logger.info(f"New Deepgram connection established for {frontend_ws_id}")
                            else:
                                await websocket.send_json({
                                    "type": "error", 
                                    "message": "Failed to establish Deepgram connection"
                                })
                        else:
                            logger.info(f"Deepgram connection already active for {frontend_ws_id}")
                    
                    else:
                        # Handle other message types if needed
                        logger.info(f"Received unknown message type '{data.get('type')}' from {frontend_ws_id}")

                    await websocket.send_json({"type": "ack", "message": f"Received text message: {data.get('type', 'unknown')}"})
                except json.JSONDecodeError:
                    # Log if it's a non-JSON text message
                    logger.warning(f"Received non-JSON text message: '{message['text']}'")
                    await websocket.send_json({"type": "ack", "message": f"Received raw text: {message['text'][:50]}..."})

            else:
                # Log if an unexpected message type is received
                logger.warning(f"Received unknown message type: {message}")
                await websocket.send_json({"type": "ack", "message": "Unknown message type received."})

    except WebSocketDisconnect:
        logger.info(f"Frontend WebSocket disconnected gracefully for {frontend_ws_id}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in WebSocket for {frontend_ws_id}: {e}", exc_info=True)
    finally:
        logger.info(f"Starting cleanup for {frontend_ws_id}")

        # Start background cleanup task and don't wait for it
        cleanup_task = asyncio.create_task(
            cleanup_connections_background(frontend_ws_id)
        )
        
        # Remove from frontend map immediately
        if frontend_ws_id in app.state.frontend_ws_map:
            del app.state.frontend_ws_map[frontend_ws_id]
            logger.info(f"Removed frontend WebSocket from state map for {frontend_ws_id}")
        
        logger.info(f"WebSocket cleanup initiated for {frontend_ws_id} (background task running)")



# Add performance endpoint
@app.get("/performance/stats")
async def get_performance_stats():
    return {
        "summary": perf_monitor.get_performance_summary(),
        "recent_logs": perf_monitor.detailed_logs[-50:],  # Last 50 operations
        "timestamp": datetime.now().isoformat()
    }