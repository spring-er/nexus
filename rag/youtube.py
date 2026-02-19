"""
rag/youtube.py — YouTube Transcript Ingestion

This module pulls transcripts from YouTube videos and converts them
into chunks for the knowledge base. No Google API key is needed —
we use the youtube-transcript-api library which reads the transcript
data that YouTube makes publicly available.

HOW IT WORKS:
    1. Extract the video ID from a YouTube URL
    2. Fetch the transcript (auto-generated or manual captions)
    3. Combine transcript segments into full text
    4. Chunk the text using our standard splitter

WHY YOUTUBE TRANSCRIPTS?
    YouTube is a massive knowledge source. By ingesting video transcripts,
    users can ask questions about video content without watching the
    entire video. This is especially useful for lectures, tutorials,
    and talks.

LIMITATIONS:
    - Only works for videos with captions (most have auto-generated ones)
    - Auto-generated captions can have errors (no punctuation, wrong words)
    - Some videos disable transcript access
"""

import re

from youtube_transcript_api import YouTubeTranscriptApi

from rag.ingestion import TextChunk, ChunkMetadata, _split_text_into_chunks


def extract_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from various URL formats.

    YouTube URLs come in many forms:
        - https://www.youtube.com/watch?v=VIDEO_ID
        - https://youtu.be/VIDEO_ID
        - https://www.youtube.com/embed/VIDEO_ID
        - https://www.youtube.com/v/VIDEO_ID

    This function handles all of them using regex.

    Args:
        url: A YouTube video URL.

    Returns:
        The 11-character video ID string.

    Raises:
        ValueError: If the URL doesn't contain a valid YouTube video ID.
    """
    # Pattern matches YouTube video IDs in all common URL formats.
    # Video IDs are 11 characters: letters, numbers, hyphens, underscores.
    patterns = [
        r"(?:youtube\.com/watch\?v=)([\w-]{11})",    # Standard URL
        r"(?:youtu\.be/)([\w-]{11})",                  # Short URL
        r"(?:youtube\.com/embed/)([\w-]{11})",         # Embed URL
        r"(?:youtube\.com/v/)([\w-]{11})",             # Old embed URL
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError(
        f"Could not extract video ID from URL: {url}\n"
        f"Please provide a valid YouTube URL."
    )


def ingest_youtube(url: str) -> list[TextChunk]:
    """
    Ingest a YouTube video by fetching and chunking its transcript.

    Args:
        url: A YouTube video URL (any format).

    Returns:
        List of TextChunk objects from the video transcript.

    Raises:
        ValueError: If the URL is invalid or the video has no transcript.
    """
    # Step 1: Extract the video ID from the URL
    video_id = extract_video_id(url)

    # Step 2: Fetch the transcript
    # The API returns a list of segments, each with: text, start, duration
    # Example: [{"text": "Hello everyone", "start": 0.0, "duration": 2.5}, ...]
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try to get a manually created transcript first (higher quality),
        # fall back to auto-generated
        try:
            transcript = transcript_list.find_manually_created_transcript(
                ["en"]
            )
        except Exception:
            transcript = transcript_list.find_generated_transcript(["en"])

        segments = transcript.fetch()

    except Exception as e:
        raise ValueError(
            f"Could not fetch transcript for video {video_id}: {e}\n"
            f"The video may not have captions available."
        ) from e

    # Step 3: Combine segments into full text.
    # Each segment is a short phrase (2-5 seconds of speech).
    # We join them with spaces to create readable paragraphs.
    full_text = " ".join(
        segment.text for segment in segments
    )

    if not full_text.strip():
        raise ValueError(f"Transcript is empty for video {video_id}")

    # Step 4: Chunk the text using our standard splitter.
    title = f"YouTube: {video_id}"
    chunks = _split_text_into_chunks(
        text=full_text,
        source=url,
        source_type="youtube",
        title=title,
    )

    return chunks
