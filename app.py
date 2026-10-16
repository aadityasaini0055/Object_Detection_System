
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="Object Detection System",
    page_icon="🎯",
    layout="wide"
)

# Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

# Sidebar Navigation
st.sidebar.title("🎯 Object Detection")

if "page" not in st.session_state:
    st.session_state.page = "About Project"

if st.sidebar.button("🏠 About Project", use_container_width=True):
    st.session_state.page = "About Project"

if st.sidebar.button("🖼️ Image Detection", use_container_width=True):
    st.session_state.page = "Image Detection"

if st.sidebar.button("📹 Webcam Detection", use_container_width=True):
    st.session_state.page = "Webcam Detection"

if st.sidebar.button("🎬 Video Detection", use_container_width=True):
    st.session_state.page = "Video Detection"

page = st.session_state.page

# About Project
if page == "About Project":
    st.image("image.png")

    st.title("🎯 YOLOv8 Object Detection System")

    st.write("""
    This project detects objects using YOLOv8 and OpenCV.

    ### Features
    - Image Object Detection
    - Real-Time Webcam Detection
    - Video Object Detection

    ### Technologies
    Python, YOLOv8, OpenCV, Streamlit, Deep Learning
    """)

# Image Detection
elif page == "Image Detection":

    st.title("🖼️ Image Detection")

    uploaded_image = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        file_bytes = np.frombuffer(
            uploaded_image.getvalue(), np.uint8
        )
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("Invalid image file.")
        else:
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption="Uploaded Image",
                use_container_width=True
            )

            if st.button("🔍 Predict Image"):
                try:
                    model = load_model()
                    result = model.predict(
                        image, conf=0.25, verbose=False
                    )[0]

                    detected_image = result.plot()

                    st.subheader("Detection Result")
                    st.image(
                        cv2.cvtColor(
                            detected_image, cv2.COLOR_BGR2RGB
                        ),
                        use_container_width=True
                    )

                    st.success(
                        f"Objects detected: {len(result.boxes)}"
                    )

                    success, encoded = cv2.imencode(
                        ".jpg", detected_image
                    )

                    if success:
                        st.download_button(
                            "⬇️ Download Result",
                            data=encoded.tobytes(),
                            file_name="detected_image.jpg",
                            mime="image/jpeg"
                        )

                except Exception as e:
                    st.error(f"Detection Error: {e}")

# Webcam Detection
elif page == "Webcam Detection":

    st.title("📹 Live Webcam Detection")
    st.write("Start the camera to detect objects in real time.")

    try:
        model = load_model()

        class WebcamDetector(VideoProcessorBase):

            def recv(self, frame):
                image = frame.to_ndarray(format="bgr24")

                result = model.predict(
                    image, conf=0.25, verbose=False
                )[0]

                output = result.plot()

                return av.VideoFrame.from_ndarray(
                    output, format="bgr24"
                )

        webrtc_streamer(
            key="object-detection-webcam",
            video_processor_factory=WebcamDetector,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True
        )

        st.info("Click START to begin and STOP to finish.")

    except Exception as e:
        st.error(f"Webcam Error: {e}")

# Video Detection
elif page == "Video Detection":

    st.title("🎬 Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("🔍 Predict Video"):
            input_path = None
            output_path = None
            cap = None
            out = None

            try:
                model = load_model()

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as temp_input:
                    temp_input.write(uploaded_video.getvalue())
                    input_path = temp_input.name

                cap = cv2.VideoCapture(input_path)

                if not cap.isOpened():
                    st.error("Video could not be opened.")
                else:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 20.0

                    width = int(
                        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    )
                    height = int(
                        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    )

                    if width <= 0 or height <= 0:
                        st.error("Invalid video dimensions.")
                    else:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".mp4"
                        ) as temp_output:
                            output_path = temp_output.name

                        out = cv2.VideoWriter(
                            output_path,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (width, height)
                        )

                        if not out.isOpened():
                            st.error("Could not create output video.")
                        else:
                            progress = st.progress(0)
                            status = st.empty()

                            total_frames = int(
                                cap.get(cv2.CAP_PROP_FRAME_COUNT)
                            )
                            frame_count = 0

                            while True:
                                ret, frame = cap.read()

                                if not ret:
                                    break

                                result = model.predict(
                                    frame,
                                    conf=0.25,
                                    verbose=False
                                )[0]

                                out.write(result.plot())
                                frame_count += 1

                                if total_frames > 0:
                                    progress.progress(
                                        min(
                                            frame_count / total_frames,
                                            1.0
                                        )
                                    )

                            cap.release()
                            cap = None
                            out.release()
                            out = None

                            with open(output_path, "rb") as f:
                                st.session_state.video_result = f.read()

                            st.success(
                                f"Detection completed! "
                                f"Frames processed: {frame_count}"
                            )

            except Exception as e:
                st.error(f"Video Detection Error: {e}")

            finally:
                if cap is not None:
                    cap.release()

                if out is not None:
                    out.release()

                for path in (input_path, output_path):
                    if path and os.path.exists(path):
                        os.remove(path)

    if "video_result" in st.session_state:
        st.subheader("Detection Result")

        st.video(st.session_state.video_result)

        st.download_button(
            "⬇️ Download Detected Video",
            data=st.session_state.video_result,
            file_name="detected_video.mp4",
            mime="video/mp4"
        )
