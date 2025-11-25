# src/gstreamer_source.py

import threading
import queue

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GObject", "2.0")
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GObject, GstVideo

import numpy as np

Gst.init(None)


class GStreamerFileSource:
    """
    Simple GStreamer → numpy bridge using appsink.

    - Plays a local video file once (no loop).
    - Delivers raw BGR frames via a blocking queue.
    """

    def __init__(self, filename: str, width: int, height: int, queue_size: int = 10):
        self._filename = filename
        self._width = width
        self._height = height
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=queue_size)
        self._done = False

        pipeline_str = f"""
            filesrc location="{self._filename}" !
            decodebin !
            videoconvert !
            videoscale !
            video/x-raw,format=BGR,width={self._width},height={self._height} !
            appsink name=mysink emit-signals=true sync=false max-buffers=2 drop=true
        """

        self._pipeline: Gst.Pipeline = Gst.parse_launch(pipeline_str)
        self._appsink: Gst.Element = self._pipeline.get_by_name("mysink")
        self._appsink.connect("new-sample", self._on_new_sample)

        # Watch for EOS / errors
        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._loop = GObject.MainLoop()
        self._thread = threading.Thread(target=self._loop.run, daemon=True)

    def start(self):
        """Start the GStreamer pipeline and GLib main loop thread."""
        self._pipeline.set_state(Gst.State.PLAYING)
        self._thread.start()

    def stop(self):
        """Stop playback and the GLib loop."""
        self._pipeline.set_state(Gst.State.NULL)
        if self._loop.is_running():
            self._loop.quit()
        self._done = True

    @property
    def done(self) -> bool:
        return self._done

    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message):
        msg_type = message.type

        if msg_type == Gst.MessageType.EOS:
            # End of stream: mark done and stop
            self._done = True
            self.stop()

        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"GStreamer ERROR: {err}, debug: {debug}")
            self._done = True
            self.stop()

    def _on_new_sample(self, sink: Gst.Element):
        """Called by GStreamer when the appsink has a new frame."""
        sample = sink.emit("pull-sample")
        buf = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        width = struct.get_value("width")
        height = struct.get_value("height")

        # Get video meta to access stride (bytes per row)
        video_meta = GstVideo.buffer_get_video_meta(buf)
        if video_meta is not None:
            row_stride = video_meta.stride[0]   # bytes per row for plane 0
        else:
            # Fallback: infer stride from total size (may still work)
            ok, tmp_map = buf.map(Gst.MapFlags.READ)
            if not ok:
                return Gst.FlowReturn.ERROR
            row_stride = tmp_map.size // height
            buf.unmap(tmp_map)

        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR

        try:
            # raw BGR data in a padded layout
            data = np.frombuffer(mapinfo.data, dtype=np.uint8)

            expected_size = row_stride * height
            if data.size < expected_size:
                # Something’s wrong with our assumptions; bail out
                print(
                    f"Buffer smaller than expected: size={data.size}, "
                    f"row_stride={row_stride}, height={height}"
                )
                return Gst.FlowReturn.ERROR

            # First interpret it as (height, row_stride)
            frame_padded = data[: expected_size].reshape((height, row_stride))

            # Drop padding at the end of each row: keep only width*3 bytes
            valid_row_bytes = width * 3
            if valid_row_bytes > row_stride:
                print(
                    f"Row width*3 ({valid_row_bytes}) > row_stride ({row_stride}) – caps/format mismatch?"
                )
                return Gst.FlowReturn.ERROR

            frame_2d = frame_padded[:, :valid_row_bytes]

            # Now reshape into (height, width, 3)
            frame = frame_2d.reshape((height, width, 3))

            # Copy because GStreamer will reuse the buffer
            frame = frame.copy()

            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                # Drop frame if consumer is too slow
                pass

        finally:
            buf.unmap(mapinfo)

        return Gst.FlowReturn.OK

    def get_frame_blocking(self):
        """
        Blocking call: returns the next frame (numpy array BGR).

        Raises queue.Empty only if called after stop() with empty queue.
        """
        return self._queue.get()
