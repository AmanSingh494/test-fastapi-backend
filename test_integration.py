# In main.py - Add a task queue for TTS processing
app.state.tts_tasks = {}  # Maps frontend_ws_id to current TTS task

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
        
        # Send utterance to frontend immediately
        await frontend_ws.send_json({
            "type": "utterance_received",
            "text": full_transcript,
            "message": "Processing your request..."
        })
        
        # Queue TTS processing as a background task (don't await it here!)
        tts_task = asyncio.create_task(
            process_utterance_with_tts(frontend_ws_id, full_transcript)
        )
        
        # Store task reference (cancel previous if still running)
        if frontend_ws_id in app.state.tts_tasks:
            old_task = app.state.tts_tasks[frontend_ws_id]
            if not old_task.done():
                old_task.cancel()
        
        app.state.tts_tasks[frontend_ws_id] = tts_task
        logger.info(f"Queued TTS processing task for {frontend_ws_id}")

    except Exception as e:
        logger.error(f"Error handling Deepgram utterance end for {frontend_ws_id}: {e}")




# Initialize the TTS tasks map
app.state.tts_tasks = {}