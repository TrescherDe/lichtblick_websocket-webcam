# main.py

import asyncio
import time
import cv2

from src.config import IMAGE_WIDTH, IMAGE_HEIGHT, VIDEO_PATH
from src.messages.mouse_position import get_mouse_messages
from src.messages.eyes import get_eyes_message
from src.messages.faces import get_faces_message
from src.messages.camera_info import get_camera_info_message
from src.messages.webcam import get_image_message
from src.channels import add_channels_to_server
from src.listener import FoxgloveListener
from src.gstreamer_source import GStreamerFileSource  # NEW

from foxglove_websocket import run_cancellable
from foxglove_websocket.server import FoxgloveServer


async def main():
    # --- GStreamer source instead of webcam ---
    source = GStreamerFileSource(VIDEO_PATH, IMAGE_WIDTH, IMAGE_HEIGHT)
    source.start()

    async with FoxgloveServer(
        "0.0.0.0",
        8765,
        "example server",
        capabilities=["clientPublish", "services"],
        supported_encodings=["json"],
    ) as server:
        server.set_listener(FoxgloveListener())
        channels = await add_channels_to_server(server)

        loop = asyncio.get_running_loop()

        while not source.done:
            # Get next frame from GStreamer in a thread so we don’t block asyncio
            frame = await loop.run_in_executor(None, source.get_frame_blocking)

            # At this point:
            # - frame is a numpy array (H, W, 3), BGR order
            # - Just like cv2.VideoCapture gave you before

            # If you want to force resize anyway (should already be IMAGE_WIDTH x IMAGE_HEIGHT from pipeline)
            frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))

            # Encode frame as JPEG (same as your previous code)
            ret, jpeg = cv2.imencode(".jpg", frame)
            if not ret:
                print("Warning: JPEG encoding failed, skipping frame")
                continue

            await asyncio.sleep(0.001)

            # Reuse same timestamp logic
            now = time.time()
            timestamp = {
                "sec": int(now),
                "nsec": int((now % 1) * 1e9),
            }

            # --- Build and send messages just like before ---
            image_message = get_image_message(jpeg, timestamp)
            await server.send_message(channels["image"], time.time_ns(), image_message)

            camera_info_message = get_camera_info_message(timestamp)
            await server.send_message(channels["camera_calibration"], time.time_ns(), camera_info_message)

            faces_message = get_faces_message(frame, timestamp)
            await server.send_message(channels["faces"], time.time_ns(), faces_message)

            eyes_message = get_eyes_message(frame, timestamp)
            await server.send_message(channels["eyes"], time.time_ns(), eyes_message)

            mouse_position_message, mouse_markers_message = get_mouse_messages(timestamp)
            await server.send_message(channels["mouse_position"], time.time_ns(), mouse_position_message)
            await server.send_message(channels["mouse_markers"], time.time_ns(), mouse_markers_message)

    # Cleanup
    source.stop()


if __name__ == "__main__":
    run_cancellable(main())
