import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml

from finger_symbol_recognition.classifier import EMNISTClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def prepare_canvas(
    canvas: np.ndarray, target_size: tuple[int, int] = (28, 28)
) -> np.ndarray | None:
    if canvas is None:
        return None

    gray_canvas = np.max(canvas, axis=2) if canvas.ndim == 3 else canvas
    _, thresh = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)

    points = cv2.findNonZero(thresh)
    if points is None:
        return None

    x, y, w, h = cv2.boundingRect(points)

    padding = 20
    H, W = canvas.shape[:2]
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(W, x + w + padding)
    y2 = min(H, y + h + padding)

    cropped = thresh[y1:y2, x1:x2]
    ch, cw = cropped.shape
    max_side = max(ch, cw)
    square = np.zeros((max_side, max_side), dtype=np.uint8)

    offset_x = (max_side - cw) // 2
    offset_y = (max_side - ch) // 2
    square[offset_y : offset_y + ch, offset_x : offset_x + cw] = cropped

    resized = cv2.resize(square, target_size, interpolation=cv2.INTER_AREA)
    return resized


class AirDrawingApp:
    def __init__(
        self,
        config_path: Path | str = PROJECT_ROOT / "config.yaml",
        hand_model_path: Path | str = PROJECT_ROOT / "models" / "hand_landmarker.task",
        classifier_model_path: Path | str = PROJECT_ROOT
        / "models"
        / "emnist_balanced.onnx",
    ):
        self.config_path = Path(config_path)
        self.hand_model_path = Path(hand_model_path)
        self.classifier_model_path = Path(classifier_model_path)

        self.load_config()
        self.classifier = EMNISTClassifier(self.classifier_model_path)

    def load_config(self) -> None:
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as cfg:
                config = yaml.safe_load(cfg) or {}
        else:
            config = {}

        cam_cfg = config.get("camera", {})
        self.camera_index = cam_cfg.get("index", config.get("camera_index", 0))
        self.camera_width = cam_cfg.get("width", 1280)
        self.camera_height = cam_cfg.get("height", 720)
        self.window_name = cam_cfg.get("window_name", "Air Drawing")

        draw_cfg = config.get("drawing", {})
        self.main_color = tuple(
            draw_cfg.get("main_color", config.get("main_color", [255, 0, 0]))
        )
        self.brush_thickness = draw_cfg.get(
            "brush_thickness", config.get("brush_thickness", 8)
        )
        self.text_block_color = tuple(
            draw_cfg.get(
                "text_block_color", config.get("text_block_color", [255, 255, 0])
            )
        )
        self.banner_alpha = draw_cfg.get("banner_alpha", 0.4)
        self.confidence_threshold = draw_cfg.get("confidence_threshold", 0.7)

        gest_cfg = config.get("gestures", {})
        self.draw_delay = gest_cfg.get("draw_delay", 0.5)
        self.swipe_distance_threshold = gest_cfg.get("swipe_distance_threshold", 0.12)
        self.swipe_ratio_threshold = gest_cfg.get("swipe_ratio_threshold", 1.3)
        self.swipe_time_window = gest_cfg.get("swipe_time_window", 0.4)
        self.swipe_cooldown = gest_cfg.get("swipe_cooldown", 0.8)

    @contextmanager
    def open_camera(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open camera at index {self.camera_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.camera_width, self.camera_height)

        try:
            yield cap
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run(self) -> None:
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.hand_model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        with (
            self.open_camera() as cap,
            HandLandmarker.create_from_options(options) as landmarker,
        ):
            canvas = None
            prev_x, prev_y = None, None
            draw_start_time = None
            holding_save = False
            holding_backspace = False
            last_timestamp_ms = 0
            text = ""

            palm_history = deque(maxlen=20)
            last_erase_time = 0.0
            erase_notification_until = 0.0
            backspace_notification_until = 0.0

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                if canvas is None:
                    canvas = np.zeros_like(frame)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                current_time_ms = int(time.time() * 1000)
                timestamp_ms = max(current_time_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms

                results = landmarker.detect_for_video(mp_image, timestamp_ms)

                eucl_dist = lambda hand1, hand2: np.sqrt(
                    (hand1.x - hand2.x) ** 2 + (hand1.y - hand2.y) ** 2
                )

                now = time.time()

                live_prep_canvas = prepare_canvas(canvas, target_size=(28, 28))
                top_predictions = None
                if live_prep_canvas is not None:
                    top_predictions = self.classifier.predict_top_k(
                        live_prep_canvas, k=3
                    )

                if results.hand_landmarks:
                    for hand in results.hand_landmarks:
                        is_draw = (
                            eucl_dist(hand[8], hand[0])
                            > 0.9 * eucl_dist(hand[6], hand[0])
                        ) and (
                            eucl_dist(hand[12], hand[0])
                            < 0.9 * eucl_dist(hand[8], hand[0])
                        )

                        is_backspace = (
                            (
                                eucl_dist(hand[20], hand[0])
                                > eucl_dist(hand[18], hand[0])
                            )
                            and (
                                eucl_dist(hand[8], hand[0])
                                < eucl_dist(hand[6], hand[0])
                            )
                            and (
                                eucl_dist(hand[12], hand[0])
                                < eucl_dist(hand[10], hand[0])
                            )
                            and (
                                eucl_dist(hand[16], hand[0])
                                < eucl_dist(hand[14], hand[0])
                            )
                        )

                        is_clear = (
                            (eucl_dist(hand[8], hand[0]) < eucl_dist(hand[6], hand[0]))
                            and (
                                eucl_dist(hand[12], hand[0])
                                < eucl_dist(hand[10], hand[0])
                            )
                            and (
                                eucl_dist(hand[16], hand[0])
                                < eucl_dist(hand[14], hand[0])
                            )
                            and (
                                eucl_dist(hand[20], hand[0])
                                < eucl_dist(hand[18], hand[0])
                            )
                        )

                        is_palm_open = (
                            (eucl_dist(hand[8], hand[0]) > eucl_dist(hand[6], hand[0]))
                            and (
                                eucl_dist(hand[12], hand[0])
                                > eucl_dist(hand[10], hand[0])
                            )
                            and (
                                eucl_dist(hand[16], hand[0])
                                > eucl_dist(hand[14], hand[0])
                            )
                            and (
                                eucl_dist(hand[20], hand[0])
                                > eucl_dist(hand[18], hand[0])
                            )
                        )

                        is_swipe_erased = False
                        if is_palm_open:
                            palm_center_x = (hand[0].x + hand[9].x) / 2.0
                            palm_center_y = (hand[0].y + hand[9].y) / 2.0
                            palm_history.append((now, palm_center_x, palm_center_y))

                            if len(palm_history) >= 4 and (
                                now - last_erase_time > self.swipe_cooldown
                            ):
                                valid_points = [
                                    p
                                    for p in palm_history
                                    if now - p[0] <= self.swipe_time_window
                                ]
                                if len(valid_points) >= 3:
                                    dx = palm_center_x - valid_points[0][1]
                                    dy = palm_center_y - valid_points[0][2]

                                    if abs(dx) > self.swipe_distance_threshold and abs(
                                        dx
                                    ) > self.swipe_ratio_threshold * abs(dy):
                                        text = ""
                                        canvas = np.zeros_like(frame)
                                        last_erase_time = now
                                        erase_notification_until = now + 1.2
                                        palm_history.clear()
                                        holding_save = True
                                        is_swipe_erased = True
                        else:
                            palm_history.clear()

                        if is_draw and not is_palm_open:
                            holding_save = False
                            holding_backspace = False
                            if draw_start_time is None:
                                draw_start_time = now

                            if (now - draw_start_time) >= self.draw_delay:
                                it_x, it_y = int(hand[8].x * w), int(hand[8].y * h)
                                if prev_x is not None:
                                    cv2.line(
                                        canvas,
                                        (prev_x, prev_y),
                                        (it_x, it_y),
                                        color=self.main_color,
                                        thickness=self.brush_thickness,
                                    )
                                prev_x, prev_y = it_x, it_y
                            else:
                                prev_x, prev_y = None, None
                        elif is_backspace:
                            draw_start_time = None
                            if not holding_backspace:
                                if len(text) > 0:
                                    text = text[:-1]
                                    backspace_notification_until = now + 1.0
                                holding_backspace = True
                            holding_save = False
                            prev_x, prev_y = None, None
                        elif is_clear:
                            draw_start_time = None
                            canvas = np.zeros_like(frame)
                            prev_x, prev_y = None, None
                            holding_save = False
                            holding_backspace = False
                        elif is_palm_open and not is_swipe_erased:
                            draw_start_time = None
                            holding_backspace = False
                            if not holding_save:
                                if (
                                    top_predictions is not None
                                    and len(top_predictions) > 0
                                ):
                                    best_char, best_conf = top_predictions[0]
                                    if best_conf > self.confidence_threshold:
                                        text += best_char
                                canvas = np.zeros_like(frame)
                                holding_save = True
                            prev_x, prev_y = None, None
                        else:
                            draw_start_time = None
                            holding_save = False
                            holding_backspace = False
                            prev_x, prev_y = None, None
                else:
                    draw_start_time = None
                    holding_save = False
                    holding_backspace = False
                    prev_x, prev_y = None, None
                    palm_history.clear()

                canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                result_frame = cv2.add(frame, canvas_bgr)

                if len(text) > 0:
                    overlay = result_frame.copy()
                    banner_color_bgr = (
                        self.text_block_color[2],
                        self.text_block_color[1],
                        self.text_block_color[0],
                    )
                    cv2.rectangle(overlay, (25, 25), (w - 25, 85), banner_color_bgr, -1)

                    result_frame = cv2.addWeighted(
                        overlay,
                        self.banner_alpha,
                        result_frame,
                        1.0 - self.banner_alpha,
                        0,
                    )

                    cv2.putText(
                        result_frame,
                        text,
                        (35, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if now < backspace_notification_until:
                    overlay = result_frame.copy()
                    cv2.rectangle(
                        overlay, (25, h - 70), (330, h - 20), (0, 140, 255), -1
                    )
                    result_frame = cv2.addWeighted(overlay, 0.6, result_frame, 0.4, 0)
                    cv2.putText(
                        result_frame,
                        "Backspace [-1 Char]",
                        (35, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                elif now < erase_notification_until:
                    overlay = result_frame.copy()
                    cv2.rectangle(overlay, (25, h - 70), (320, h - 20), (0, 0, 200), -1)
                    result_frame = cv2.addWeighted(overlay, 0.6, result_frame, 0.4, 0)
                    cv2.putText(
                        result_frame,
                        "Text Cleared [Swipe]",
                        (35, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if live_prep_canvas is not None and top_predictions is not None:
                    box_w, box_h = 240, 110
                    bx1, by1 = w - box_w - 25, h - box_h - 25
                    bx2, by2 = w - 25, h - 25

                    overlay = result_frame.copy()
                    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
                    result_frame = cv2.addWeighted(overlay, 0.75, result_frame, 0.25, 0)
                    cv2.rectangle(result_frame, (bx1, by1), (bx2, by2), (90, 90, 90), 1)

                    thumb_size = 80
                    thumb = cv2.resize(
                        live_prep_canvas,
                        (thumb_size, thumb_size),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
                    tx1, ty1 = bx1 + 15, by1 + 15
                    result_frame[ty1 : ty1 + thumb_size, tx1 : tx1 + thumb_size] = (
                        thumb_bgr
                    )
                    cv2.rectangle(
                        result_frame,
                        (tx1, ty1),
                        (tx1 + thumb_size, ty1 + thumb_size),
                        (180, 180, 180),
                        1,
                    )

                    for rank, (char_name, prob) in enumerate(top_predictions):
                        line_y = ty1 + 20 + rank * 24
                        color = (0, 255, 120) if rank == 0 else (200, 200, 200)
                        cv2.putText(
                            result_frame,
                            f"{rank + 1}. [{char_name}] {prob * 100:.0f}%",
                            (tx1 + thumb_size + 15, line_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            1,
                            cv2.LINE_AA,
                        )

                cv2.imshow(self.window_name, result_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    canvas = np.zeros_like(frame)
                    text = ""
