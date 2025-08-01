# main.py
import asyncio
from asyncio import Queue
from typing import AsyncGenerator, Optional
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
from dataclasses import dataclass
from performance_monitor import perf_monitor

load_dotenv()
# Initialize FastAPI app
app = FastAPI()

    
app.state.frontend_ws_map = {} # Maps frontend_ws_id to the actual WebSocket object
app.state.deepgram_connections = {} # Maps frontend_ws_id to Deepgram connection
app.state.sarvam_connections = {} # Maps frontend_ws_id to sarvam AI TTS connection
app.state.sarvam_clients = {} # Maps frontend_ws_id to sarvam AI client
app.state.sarvam_context_managers = {} # Maps frontend_ws_id to sarvam AI context manager
# app.state.connection_tasks = {} 
app.state.tts_tasks = {}
app.state.tts_queues = {}  # Maps frontend_ws_id to request queue
app.state.tts_processors = {}  # Maps frontend_ws_id to processor task
app.state.sarvam_ping_tasks = {}  # Maps frontend_ws_id to ping task
current_utterance_transcript = []
utterance_lock = threading.Lock() 

@dataclass
class TTSRequest:
    user_input: str
    task_id: str
    frontend_ws_id: str
    cancelled: bool = False
    processing_start_time: Optional[float] = None

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
        
        if(sentence and is_final):
            if len(sentence.strip()) > 2:  # Significant speech detected
            # Cancel pending TTS requests
                # await cancel_pending_tts_requests(frontend_ws_id)
            
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
                            
                        await frontend_ws.send_json({
                            "type": 'audio_interrupt',
                            "detected_speech": sentence
                })
                    logger.info(f"🛑 INTERRUPTING: Detected speech '{sentence}' for {frontend_ws_id}")
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
        processing_start_time = time.time()
        if not frontend_ws:
            logger.error(f"Frontend WS not found for {frontend_ws_id}")
            return

        with utterance_lock:
            full_transcript = " ".join(current_utterance_transcript).strip()
            current_utterance_transcript = []

        if not full_transcript:
            logger.warning(f"Received empty utterance end for {frontend_ws_id}")
            return
 
        try:
            logger.info(f"Processing utterance end: '{full_transcript}' for {frontend_ws_id} at {time.time()*1000:.1f}ms")
            await frontend_ws.send_json({
                "type": "utterance_end",
                "text": full_transcript,
                "message": "Processing your request..."
            })
        except Exception:
            logger.warning(f"Could not send utterance_end message to {frontend_ws_id} - frontend may be disconnected")
            return
        
        # Queue TTS processing as a background task (don't await it here!)
        tts_task = asyncio.create_task(
            process_utterance_with_tts(frontend_ws_id, full_transcript,processing_start_time)
        )

        app.state.tts_tasks[frontend_ws_id] = tts_task
        logger.info(f"Created TTS + LLM processing task for {frontend_ws_id}")


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
            utterance_end_ms=1111
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
        # Cancel ping task first
        if hasattr(app.state, 'sarvam_ping_tasks') and frontend_ws_id in app.state.sarvam_ping_tasks:
            ping_task = app.state.sarvam_ping_tasks[frontend_ws_id]
            logger.info(f"Found Sarvam ping task for {frontend_ws_id}, attempting to cancel...")
            try:
                if not ping_task.done():
                    ping_task.cancel()
                    logger.info(f"Cancelled Sarvam ping task for {frontend_ws_id}")
                    try:
                        # Wait for the cancellation to complete with timeout
                        await asyncio.wait_for(ping_task, timeout=2.0)
                        logger.info(f"Sarvam ping task cancelled successfully for {frontend_ws_id}")
                    except asyncio.CancelledError:
                        logger.info(f"Sarvam ping task was cancelled for {frontend_ws_id}")
                    except asyncio.TimeoutError:
                        logger.warning(f"Sarvam ping task cancellation timed out for {frontend_ws_id}")
                else:
                    logger.info(f"Sarvam ping task already completed for {frontend_ws_id}")
            except Exception as e:
                logger.error(f"Error cancelling Sarvam ping task for {frontend_ws_id}: {e}")
            
            # Remove from state even if cancellation failed
            try:
                del app.state.sarvam_ping_tasks[frontend_ws_id]
                logger.info(f"Removed Sarvam ping task from state for {frontend_ws_id}")
            except KeyError:
                logger.warning(f"Sarvam ping task already removed for {frontend_ws_id}")
        else:
            logger.info(f"No Sarvam ping task found for {frontend_ws_id}")
        
        # Close the context manager (most important)
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
        ping_task = asyncio.create_task(send_sarvam_pings(ws_connection))
        app.state.sarvam_ping_tasks[frontend_ws_id] = ping_task
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
async def process_utterance_with_tts(frontend_ws_id: str, full_transcript: str,processing_start_time = None):
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
                    task_id=task_id,
                    tts_start=tts_start,
                    processing_start_time = processing_start_time
                ),
                timeout=30.0  # 30 second timeout
            )
            logger.info(f"TTS : {task_id} completed for {frontend_ws_id}: {len(audio_chunks)} chunks")
            tts_time = time.time() - tts_start
            print(f"🎵 TTS processing time: {tts_time*1000:.1f}ms")
    
            total_time = time.time() - utterance_start
            print(f"🏁 Total utterance processing: {total_time*1000:.1f}ms")
        
        except asyncio.CancelledError:
            # Handle cancellation properly (don't treat as error)
            logger.info(f"🛑 TTS task {task_id} was cancelled for {frontend_ws_id}")
            # Don't send error message for cancellation - it's expected behavior
            raise

        except asyncio.TimeoutError:
            logger.error(f"TTS processing timed out for {frontend_ws_id}")
            if frontend_ws and hasattr(frontend_ws, 'client_state') and frontend_ws.client_state.value == 1:
                await frontend_ws.send_json({
                    "type": "error",
                    "message": "TTS processing timed out"
                })
        except Exception as tts_error:
            logger.error(f"TTS processing failed for {frontend_ws_id}: {tts_error}")
            if frontend_ws and hasattr(frontend_ws, 'client_state') and frontend_ws.client_state.value == 1:
                await frontend_ws.send_json({
                    "type": "error",
                    "message": f"TTS processing failed: {str(tts_error)}"
                })
        
    except Exception as e:
        logger.error(f"Error in TTS processing task for {frontend_ws_id}: {e}")
        if frontend_ws_id in app.state.frontend_ws_map:
            frontend_ws = app.state.frontend_ws_map[frontend_ws_id]
            if frontend_ws and hasattr(frontend_ws, 'client_state') and frontend_ws.client_state.value == 1:
                await frontend_ws.send_json({
                    "type": "error",
                    "message": f"Processing failed: {str(e)}"
                })

async def cleanup_connections_direct(frontend_ws_id: str):
    """Direct cleanup in main thread - continue cleanup even if individual steps fail"""
    logger.info(f"Starting direct cleanup for {frontend_ws_id}")
    
    cleanup_results = {
        "tts_tasks": False,
        "tts_processors": False,
        "deepgram": False,
        "sarvam": False,
        "frontend_ws": False
    }
    
    # 1. Cancel any running TTS tasks (legacy) - INDEPENDENT CLEANUP
    try:
        if hasattr(app.state, 'tts_tasks') and frontend_ws_id in app.state.tts_tasks:
            tts_task = app.state.tts_tasks[frontend_ws_id]
            if not tts_task.done():
                logger.info(f"Cancelling running TTS task for {frontend_ws_id}")
                tts_task.cancel()
                try:
                    await asyncio.wait_for(tts_task, timeout=2.0)
                    logger.info(f"TTS task cancelled successfully for {frontend_ws_id}")
                except asyncio.CancelledError:
                    logger.info(f"TTS task was cancelled for {frontend_ws_id}")
                except asyncio.TimeoutError:
                    logger.warning(f"TTS task cancellation timed out for {frontend_ws_id}")
                except Exception as e:
                    logger.error(f"Error waiting for TTS task cancellation for {frontend_ws_id}: {e}")
            
            # Always remove from state, even if cancellation failed
            del app.state.tts_tasks[frontend_ws_id]
            logger.info(f"Removed TTS task from state for {frontend_ws_id}")
            cleanup_results["tts_tasks"] = True
    except Exception as e:
        logger.error(f"Error in TTS task cleanup for {frontend_ws_id}: {e}")
    
    # 2. Cancel TTS processor - INDEPENDENT CLEANUP
    try:
        if hasattr(app.state, 'tts_processors') and frontend_ws_id in app.state.tts_processors:
            processor_task = app.state.tts_processors[frontend_ws_id]
            if not processor_task.done():
                logger.info(f"Stopping TTS processor for {frontend_ws_id}")
                # Send shutdown signal
                try:
                    if frontend_ws_id in app.state.tts_queues:
                        await app.state.tts_queues[frontend_ws_id].put(None)
                except Exception as queue_error:
                    logger.error(f"Error sending shutdown signal to TTS queue for {frontend_ws_id}: {queue_error}")
                
                processor_task.cancel()
                try:
                    await asyncio.wait_for(processor_task, timeout=2.0)
                    logger.info(f"TTS processor stopped successfully for {frontend_ws_id}")
                except asyncio.CancelledError:
                    logger.info(f"TTS processor was cancelled for {frontend_ws_id}")
                except asyncio.TimeoutError:
                    logger.warning(f"TTS processor stop timed out for {frontend_ws_id}")
                except Exception as e:
                    logger.error(f"Error stopping TTS processor for {frontend_ws_id}: {e}")
            
            # Always clean up processor state
            if frontend_ws_id in app.state.tts_processors:
                del app.state.tts_processors[frontend_ws_id]
                logger.info(f"Removed TTS processor from state for {frontend_ws_id}")
            if frontend_ws_id in app.state.tts_queues:
                del app.state.tts_queues[frontend_ws_id]
                logger.info(f"Removed TTS queue from state for {frontend_ws_id}")
            cleanup_results["tts_processors"] = True
    except Exception as e:
        logger.error(f"Error in TTS processor cleanup for {frontend_ws_id}: {e}")
    
    # 3. Close Deepgram connection - INDEPENDENT CLEANUP
    try:
        if frontend_ws_id in app.state.deepgram_connections:
            dg_connection = app.state.deepgram_connections[frontend_ws_id]
            dg_connection.finish()  # Synchronous method
            del app.state.deepgram_connections[frontend_ws_id]
            logger.info(f"Deepgram connection closed for {frontend_ws_id}")
            cleanup_results["deepgram"] = True
    except Exception as e:
        logger.error(f"Error closing Deepgram connection for {frontend_ws_id}: {e}")

    # 4. Close Sarvam AI connection - INDEPENDENT CLEANUP
    try:
        await close_sarvam_connection(frontend_ws_id)
        logger.info(f"Sarvam cleanup completed for {frontend_ws_id}")
        cleanup_results["sarvam"] = True
    except Exception as e:
        logger.error(f"Error during Sarvam cleanup for {frontend_ws_id}: {e}")
    
    # 5. Remove from frontend map - INDEPENDENT CLEANUP
    try:
        if frontend_ws_id in app.state.frontend_ws_map:
            del app.state.frontend_ws_map[frontend_ws_id]
            logger.info(f"Removed frontend WebSocket from state map for {frontend_ws_id}")
            cleanup_results["frontend_ws"] = True
    except Exception as e:
        logger.error(f"Error removing frontend WS from state for {frontend_ws_id}: {e}")
    
    # Log cleanup summary
    successful_cleanups = sum(cleanup_results.values())
    total_cleanups = len(cleanup_results)
    logger.info(f"Direct cleanup completed for {frontend_ws_id}: {successful_cleanups}/{total_cleanups} steps successful")
    
    if successful_cleanups < total_cleanups:
        logger.warning(f"Some cleanup steps failed for {frontend_ws_id}: {cleanup_results}")  

@app.get("/")
async def get():
   return "Backend up and running - WebSocket endpoint available at /ws/audio"

# --- WebSocket Endpoint for Audio Streaming ---
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    setup_start = time.time()
    await websocket.accept()
    logger.info("Frontend WebSocket connection accepted at /ws/audio")
    
    # Create audio_recordings directory if it doesn't exist
    os.makedirs("audio_recordings", exist_ok=True)
    
    # Store the frontend WebSocket in a shared map so event handlers can access it
    frontend_ws_id = str(id(websocket)) # Unique ID for this frontend connection
    app.state.frontend_ws_map[frontend_ws_id] = websocket

    
    # logger.info(f"Starting Deepgram real-time transcription for {frontend_ws_id}")
    try:
    #     # Check if API key is available
        if not DEEPGRAM_API_KEY or not SARVAM_API_KEY:
            await websocket.send_json({
                "type": "error",
                "message": "API keys not configured on backend."
            })
            return

        # create deepgram and sarvam connections in main thread
        dg_start = time.time()
        dg_connection = create_deepgram_connection(frontend_ws_id)
        dg_time = time.time() - dg_start
        
        if not dg_connection:
            await websocket.send_json({"type": "error", "message": "Deepgram connection failed"})
            return
            
        sarvam_start = time.time()
        sarvam_connection = await create_sarvam_connection(frontend_ws_id)
        sarvam_time = time.time() - sarvam_start
        
        if not sarvam_connection:
            await websocket.send_json({"type": "error", "message": "Sarvam connection failed"})
            dg_connection.finish()
            del app.state.deepgram_connections[frontend_ws_id]
            return
        
        # Create persistent TTS processor (not using now)
        # await create_persistent_tts_processor(frontend_ws_id)
        # logger.info(f"Persistent TTS processor created for {frontend_ws_id}")

        setup_time = time.time() - setup_start
        logger.info(f"✅ Setup complete in {setup_time*1000:.1f}ms (DG: {dg_time*1000:.1f}ms, Sarvam: {sarvam_time*1000:.1f}ms)")
        
        await websocket.send_json({
            "type": "connections_ready",
            "setup_time_ms": round(setup_time * 1000, 1)
        })

        while True:
            message = await websocket.receive()
            message_type = message.get("type")
                
            if message_type == "websocket.disconnect":
                    # Handle disconnect message properly
                    disconnect_code = message.get("code", "unknown")
                    disconnect_reason = message.get("reason", "")
                    logger.info(f"WebSocket disconnect received for {frontend_ws_id} - Code: {disconnect_code}, Reason: '{disconnect_reason}'")
                    break  # Exit the loop to trigger cleanup
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
                        # Also close Sarvam connection when audio ends
                            try:
                                sarvam_ws = app.state.sarvam_connections.get(frontend_ws_id)
                                if sarvam_ws:
                                    # Flush any remaining TTS data and close connection
                                    await sarvam_ws.flush()
                                    logger.info(f"Flushed Sarvam connection for {frontend_ws_id}")
                                    
                            except Exception as e:
                                logger.error(f"Error handling Sarvam connection during audio_end for {frontend_ws_id}: {e}")

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

        
        await cleanup_connections_direct(frontend_ws_id)
        
        # Remove from frontend map immediately
        if frontend_ws_id in app.state.frontend_ws_map:
            del app.state.frontend_ws_map[frontend_ws_id]
            logger.info(f"Removed frontend WebSocket from state map for {frontend_ws_id}")
        



# Add performance endpoint
@app.get("/performance/stats")
async def get_performance_stats():
    return {
        "summary": perf_monitor.get_performance_summary(),
        "recent_logs": perf_monitor.detailed_logs[-50:],  # Last 50 operations
        "timestamp": datetime.now().isoformat()
    }