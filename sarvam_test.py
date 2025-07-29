import asyncio
import logging
import os
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI, AudioOutput
import base64

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')

async def monitor_sarvam_connection(ws_connection, monitor_duration=60):
    """Monitor Sarvam WebSocket connection state continuously"""
    logger.info("Starting Sarvam connection monitor...")
    start_time = asyncio.get_event_loop().time()
    
    while (asyncio.get_event_loop().time() - start_time) < monitor_duration:
        try:
            # Check connection state
            if hasattr(ws_connection, '_websocket'):
                ws_state = ws_connection._websocket.state if hasattr(ws_connection._websocket, 'state') else "unknown"
                ws_closed = ws_connection._websocket.closed if hasattr(ws_connection._websocket, 'closed') else "unknown"
                logger.info(f"WebSocket state: {ws_state}, closed: {ws_closed}")
            
            # Try to ping if available
            if hasattr(ws_connection, 'ping'):
                try:
                    await asyncio.wait_for(ws_connection.ping(), timeout=3.0)
                    logger.info("Ping successful")
                except Exception as ping_error:
                    logger.warning(f"Ping failed: {ping_error}")
            
            # Check available methods
            methods = [method for method in dir(ws_connection) if not method.startswith('_')]
            logger.debug(f"Available methods: {methods}")
            
            await asyncio.sleep(5)  # Check every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in connection monitor: {e}")
            break
    
    logger.info("Connection monitor ended")

async def test_sarvam_tts_isolated():
    """Test Sarvam TTS in isolation with detailed logging"""
    logger.info("Starting isolated Sarvam TTS test...")
    
    try:
        # Create client
        logger.info("Creating Sarvam client...")
        client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
        logger.info("Sarvam client created successfully")
        
        # Get context manager
        logger.info("Getting TTS streaming context manager...")
        context_manager = client.text_to_speech_streaming.connect(model="bulbul:v2")
        logger.info("Context manager created")
        
        # Enter context manager
        logger.info("Entering context manager...")
        ws_connection = await context_manager.__aenter__()
        logger.info(f"WebSocket connection established: {type(ws_connection)}")
        
        # Start monitoring task
        monitor_task = asyncio.create_task(monitor_sarvam_connection(ws_connection, monitor_duration=30))
        
        try:
            # Configure TTS
            logger.info("Configuring TTS...")
            await ws_connection.configure(target_language_code="hi-IN", speaker="anushka")
            logger.info("TTS configured successfully")
            
            # Test with simple text
            test_texts = [
                "नमस्ते",  # Simple greeting
                "आज मौसम कैसा है?",  # How's the weather today?
                "Hello world",  # English fallback
            ]
            
            for i, test_text in enumerate(test_texts):
                logger.info(f"\n=== Test {i+1}: Testing with text: '{test_text}' ===")
                
                try:
                    # Send text
                    logger.info(f"Sending text to convert: '{test_text}'")
                    await ws_connection.convert(test_text)
                    logger.info("Text sent successfully")
                    
                    # Flush
                    logger.info("Flushing TTS buffer...")
                    await ws_connection.flush()
                    logger.info("Buffer flushed successfully")
                    
                    # Listen for responses with timeout
                    logger.info("Listening for audio responses...")
                    audio_chunks_received = 0
                    
                    try:
                        async with asyncio.timeout(15.0):  # 15 second timeout per text
                            async for message in ws_connection:
                                logger.info(f"Received message type: {type(message)}")
                                
                                if isinstance(message, AudioOutput):
                                    audio_chunks_received += 1
                                    audio_size = len(message.data.audio) if hasattr(message.data, 'audio') else 0
                                    logger.info(f"Received audio chunk {audio_chunks_received}, size: {audio_size} chars (base64)")
                                    
                                    # Decode and check actual audio size
                                    try:
                                        audio_bytes = base64.b64decode(message.data.audio)
                                        logger.info(f"Decoded audio size: {len(audio_bytes)} bytes")
                                    except Exception as decode_error:
                                        logger.error(f"Failed to decode audio: {decode_error}")
                                    
                                    # Break after first chunk for this test
                                    if audio_chunks_received >= 3:  # Get a few chunks then move to next test
                                        logger.info(f"Received {audio_chunks_received} chunks, moving to next test")
                                        break
                                        
                                elif hasattr(message, 'type'):
                                    logger.info(f"Received message with type: {getattr(message, 'type', 'unknown')}")
                                    if 'end' in str(getattr(message, 'type', '')).lower():
                                        logger.info("Detected end message, breaking")
                                        break
                                else:
                                    logger.info(f"Received other message: {message}")
                    
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout waiting for audio response for text: '{test_text}'")
                    
                    logger.info(f"Test {i+1} completed. Audio chunks received: {audio_chunks_received}")
                    
                    # Small delay between tests
                    await asyncio.sleep(2)
                    
                except Exception as text_error:
                    logger.error(f"Error processing text '{test_text}': {text_error}")
                    continue
            
            logger.info("\n=== All tests completed ===")
            
        finally:
            # Cancel monitor task
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                logger.info("Monitor task cancelled")
            
            # Exit context manager
            logger.info("Exiting context manager...")
            try:
                await asyncio.wait_for(
                    context_manager.__aexit__(None, None, None),
                    timeout=5.0
                )
                logger.info("Context manager exited successfully")
            except Exception as exit_error:
                logger.error(f"Error exiting context manager: {exit_error}")
        
    except Exception as e:
        logger.error(f"Error in isolated test: {e}", exc_info=True)

async def test_multiple_connections():
    """Test multiple concurrent Sarvam connections"""
    logger.info("Testing multiple concurrent connections...")
    
    async def single_connection_test(connection_id):
        try:
            logger.info(f"Connection {connection_id}: Starting...")
            client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
            
            async with client.text_to_speech_streaming.connect(model="bulbul:v2") as ws:
                await ws.configure(target_language_code="hi-IN", speaker="anushka")
                logger.info(f"Connection {connection_id}: Configured")
                
                await ws.convert(f"टेस्ट नंबर {connection_id}")
                await ws.flush()
                logger.info(f"Connection {connection_id}: Text sent")
                
                chunk_count = 0
                async with asyncio.timeout(10.0):
                    async for message in ws:
                        if isinstance(message, AudioOutput):
                            chunk_count += 1
                            if chunk_count >= 2:  # Just get a couple chunks
                                break
                
                logger.info(f"Connection {connection_id}: Completed with {chunk_count} chunks")
                return chunk_count
                
        except Exception as e:
            logger.error(f"Connection {connection_id}: Error - {e}")
            return 0
    
    # Run 3 concurrent connections
    tasks = [single_connection_test(i) for i in range(1, 4)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    logger.info(f"Multiple connection test results: {results}")

if __name__ == "__main__":
    async def main():
        logger.info("Starting Sarvam WebSocket isolated tests...")
        
        # Test 1: Single connection with detailed monitoring
        await test_sarvam_tts_isolated()
        
        # Wait a bit between tests
        await asyncio.sleep(5)
        
        # Test 2: Multiple connections
        await test_multiple_connections()
        
        logger.info("All tests completed!")
    
    # Run the tests
    asyncio.run(main())