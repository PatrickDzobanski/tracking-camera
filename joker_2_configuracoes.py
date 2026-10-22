import cv2
import numpy as np
import time
import sys
import csv
import threading
from pathlib import Path

try:
    import serial
except ImportError:
    serial = None


class LatestFrameCapture:
    def __init__(self, video):
        self.video = video
        self.lock = threading.Lock()
        self.frame_available = threading.Event()
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.latest_frame = None
        self.latest_timestamp = None
        self.latest_sequence = 0
        self.consumed_sequence = 0
        self.captured_frames = 0
        self.dropped_frames = 0
        self.capture_error = None
        self.capture_failed = False
        self.thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True,
        )

    def _capture_loop(self):
        try:
            while not self.stop_event.is_set():
                success, frame = self.video.read()
                captured_at = time.perf_counter()

                if not success:
                    with self.lock:
                        self.capture_failed = True
                    return

                with self.lock:
                    if self.latest_sequence > self.consumed_sequence:
                        self.dropped_frames += 1
                    self.latest_frame = frame
                    self.latest_timestamp = captured_at
                    self.latest_sequence += 1
                    self.captured_frames += 1
                    self.frame_available.set()
        except cv2.error as exc:
            with self.lock:
                self.capture_error = str(exc)
        finally:
            self.finished_event.set()
            self.frame_available.set()

    def start(self):
        self.thread.start()

    def get_latest(self):
        with self.lock:
            if (
                self.latest_frame is None
                or self.latest_sequence <= self.consumed_sequence
            ):
                self.frame_available.clear()
                return None

            latest = (
                self.latest_frame,
                self.latest_timestamp,
                self.latest_sequence,
            )
            self.consumed_sequence = self.latest_sequence
            self.frame_available.clear()
            return latest

    def get_statistics(self):
        with self.lock:
            return self.captured_frames, self.dropped_frames

    def stop(self):
        self.stop_event.set()
        self.frame_available.set()


# 1. CONFIGURATION
CAMERA = 1

PROJECT_DIR = Path(__file__).resolve().parent.parent
REFERENCE_IMAGE_PATH =  "img.jpg"

ENABLE_COMPLEMENTARY_ZOOM = True
ZOOM_FACTORS = (3, 4)
# The one-way timeout allows 15 s for scanning, 6 s for return, and 1 s margin.
# Arduino firmware must finish returning before it accepts SEARCHING; Python
# cannot observe the motor position over this one-way connection.
VERIFYING_TIMEOUT_SECONDS = 22.0
VERIFYING_MIN_TRACKING_SECONDS = 2.5
VERIFYING_CONFIRMATION_INTERVAL_SECONDS = 0.5
VERIFYING_REQUIRED_DETECTIONS = 3
VERIFYING_COOLDOWN_SECONDS = 5.0
SEARCHING_HEARTBEAT_SECONDS = 5.0
ZOOM_MARGIN = 30
# Set to True to re-enable benchmark execution and its CSV columns.
ENABLE_ZOOM_BENCHMARK = False
SAVE_ZOOM_CSV = False
# Set to True to re-enable adaptive HSV instead of the fixed hsv.csv limits.
USE_ADAPTIVE_HSV = False
# Minimum contour area in original-image pixels squared for zoom searches.
ZOOM_MIN_AREA_PX2 = 50

dead_zone = 40
MIN_AREA = 300

# Geometric fallback: use minAreaRect only if the contour
# reasonably fills its bounding rectangle.
MIN_FALLBACK_RECTANGULARITY = 0.55

# Require confirmation before starting the Kalman filter to avoid
# locking onto a one-frame false positive.
CONFIRMATIONS_TO_TRACK = 3
MAX_CONFIRMATION_DISTANCE_PX = 80

# During TRACKING, limit the final search to an ROI around the
# Kalman prediction. During SEARCHING, scan the entire frame.
TRACKING_ROI_RADIUS = 220

SHAPE_TOLERANCE = 0.25

ROBUSTNESS_SCORE_THRESHOLD = 0.50

TEMPLATE_ROBUSTNESS_WEIGHT = 0.35
PROPORTION_ROBUSTNESS_WEIGHT = 0.10
RECTANGULARITY_WEIGHT = 0.10
MATCH_SHAPES_WEIGHT = 0.15
NORMALIZED_RGB_WEIGHT = 0.15
TEXTURE_WEIGHT = 0.15

MATCH_SHAPES_FACTOR = 4.0

NORMALIZED_RGB_SIGMA = 0.20
TEXTURE_SIGMA = 0.25

# Margin around the BGR ROI used to find a contour independently
# of the HSV mask. The candidate points locate the ROI, but the
# contour is extracted again from grayscale/Canny.
INDEPENDENT_CONTOUR_MARGIN = 0.18

FINAL_ROBUSTNESS_WEIGHT = 0.70
TEMPORAL_WEIGHT = 0.30

MAX_MISSED_FRAMES = 5

# NIS gating threshold for the two-dimensional (center_x, center_y) measurement.
# 13.82 is approximately the 99.9th percentile of a chi-squared
# distribution with two degrees of freedom. Higher measurements
# are considered inconsistent with the prediction.
NIS_GATE = 13.82

HSV_CSV_PATH = Path(__file__).resolve().parent / "hsv.csv"

with HSV_CSV_PATH.open(
    "r", encoding="utf-8-sig", newline=""
) as file:
    hsv_values = next(csv.DictReader(file))

H_MIN = int(hsv_values["hmin"])
H_MAX = int(hsv_values["hmax"])

S_MIN = int(hsv_values["smin"])
S_MAX = int(hsv_values["smax"])

V_MIN = int(hsv_values["vmin"])
REFERENCE_V_MIN = V_MIN
V_MAX = int(hsv_values["vmax"])

INITIAL_V_MIN = 100

V_MIN_LOWER_LIMIT = 30
V_MIN_UPPER_LIMIT = 170
# Ambient brightness provides an initial value:
# V_base = AMBIENT_LIGHT_FACTOR * median(image V channel).
AMBIENT_LIGHT_FACTOR = 0.55

LIGHT_ALPHA = 0.10

# The controller tests current V +/- DELTA_V on the SAME frame.
# This prevents area changes from being confused with object
# movement between frames.
DELTA_V = 5

ADJUST_V_EVERY = 5

# Change V only if the alternative improves the adjustment score
# by at least this amount. This helps prevent oscillation (85 <-> 90).
MIN_V_SWITCH_GAIN = 0.015

# Use MORE permissive filters while searching for the best V
# because partial segmentation may have imperfect shape or appearance.
MIN_ADJUSTMENT_AREA = 80
ADJUSTMENT_SHAPE_TOLERANCE = 0.60

# Score used ONLY to choose V_MIN.
# Area contributes without dominating, preventing a lower V that
# captures the background from automatically being considered better.
ADJUSTMENT_VISUAL_WEIGHT = 0.60
ADJUSTMENT_SHAPE_WEIGHT = 0.25
ADJUSTMENT_AREA_WEIGHT = 0.15

# Once the Kalman filter has acquired the target, limit the V search
# to a region around its prediction to reduce false detections and cost.
V_ADJUSTMENT_ROI_RADIUS = 180

# GaussianBlur kernel dimensions must be positive odd numbers.

# --- DETECTION AND NORMALIZATION ---
NORMALIZED_HEIGHT_FALLBACK = 300
MIN_NORMALIZED_HEIGHT = 100
MAX_NORMALIZED_HEIGHT = 500
POLYGON_APPROXIMATION_FACTOR = 0.03
NORMALIZED_WIDTH = 300

# --- KALMAN ---
MEASUREMENT_SIGMA = 5.0
MIN_ACCELERATION_SIGMA = 50.0
MAX_ACCELERATION_SIGMA = 500.0
PROCESS_NOISE_ALPHA = 0.2
MIN_DT = 0.001
MIN_NIS_ADAPTATION = 2.0
MAX_DT = 0.2
MAX_NIS_ADAPTATION = 9.21
# Initial position uncertainty, in pixels².
INITIAL_POSITION_VARIANCE = 25
# Initial velocity uncertainty, in (pixels/s)².
INITIAL_VELOCITY_VARIANCE = 1000

# --- RGB AND TEXTURE COMPARISON ---
MIN_RECTANGULARITY_SCALE = 0.2
MIN_RGB_SUM = 30.0
TEXTURE_STDDEV_SCALE = 128.0
TEXTURE_LAPLACIAN_SCALE = 255.0
TEXTURE_CONTRAST_SCALE = 64.0
TEXTURE_IMAGE_SIZE = (160, 160)
TEXTURE_INTERPOLATION = cv2.INTER_AREA
LAPLACIAN_KERNEL_SIZE = 3

# --- NUMERICAL TOLERANCES ---
DESCRIPTOR_EPSILON = 1e-06
GEOMETRY_EPSILON = 1e-09

DEBUG_Y_POSITION = 30
MIN_DEBUG_X_MARGIN = 10
CANDIDATE_TEXT_X_OFFSET = 100
CANDIDATE_TEXT_Y_OFFSET = 105
DEBUG_PANEL_WIDTH = 430
LOG_EVERY_N_FRAMES = 30
MAX_V_SUMMARY_LENGTH = 65
DEBUG_TEXT_SPACING = 24
CANDIDATE_TEXT_SPACING = 21
REFERENCE_WINDOW = 'Reference original'
REFERENCE_MASK_WINDOW = 'Reference mask'
NORMALIZED_REFERENCE_WINDOW = 'Normalized reference'
TRACKER_WINDOW = 'Robust tracker + Kalman'
HSV_MASK_WINDOW = 'Adaptive HSV mask'
EXIT_KEY = 'q'
NORMALIZED_CANDIDATE_WINDOW = 'Normalized candidate'
KEY_WAIT_MS = 1
CAPTURE_DIAGNOSTICS_INTERVAL_SECONDS = 1.0

# --- EDGES AND CONTOURS ---
LOW_CANNY_FACTOR = 0.66
HIGH_CANNY_FACTOR = 1.33
MIN_CANNY_INCREMENT = 30
MIN_CONTOUR_AREA_FRACTION = 0.01
MIN_CENTRALITY = 0.2
CENTRALITY_PENALTY_FACTOR = 1.5
CONTOUR_KERNEL_SIZE = (3, 3)
CONTOUR_CLOSE_OPERATION = cv2.MORPH_CLOSE
CONTOUR_CLOSE_ITERATIONS = 2
CONTOUR_DILATION_ITERATIONS = 1
CONTOUR_RETRIEVAL_MODE = cv2.RETR_EXTERNAL
CONTOUR_APPROXIMATION_METHOD = cv2.CHAIN_APPROX_SIMPLE

# --- V ADJUSTMENT: ADDITIONAL WEIGHTS ---
PRELIMINARY_VISUAL_WEIGHT = 0.7
PRELIMINARY_SHAPE_WEIGHT = 0.3
CURRENT_V_CENTER_WEIGHT = 0.8
AMBIENT_V_CENTER_WEIGHT = 0.2

# --- HSV MASK ---
# CLOSE fills gaps before OPEN removes small noise, matching jokerTESTE.PY.
HSV_CLOSE_KERNEL_SIZE = (7, 7)
HSV_OPEN_KERNEL_SIZE = (3, 3)

CONTOUR_BLUR_SIZE = (5, 5)
CONTOUR_BLUR_SIGMA = 0
TEXTURE_BLUR_SIZE = (3, 3)
TEXTURE_BLUR_SIGMA = 0
CONTRAST_BLUR_SIZE = (0, 0)
CONTRAST_BLUR_SIGMA = 3.0
TEMPLATE_BLUR_SIZE = (3, 3)
TEMPLATE_BLUR_SIGMA = 0

TEMPLATE_MATCH_METHOD = cv2.TM_CCOEFF_NORMED
MATCH_SHAPES_METHOD = cv2.CONTOURS_MATCH_I1

WINDOWS_CAMERA_BACKEND = cv2.CAP_DSHOW

GREEN = (0, 255, 0)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
V_ADJUSTMENT_ROI_COLOR = (180, 0, 180)
WHITE = (255, 255, 255)
ORANGE = (0, 165, 255)

DRAWING_THICKNESS_2 = 2
DRAWING_THICKNESS_3 = 3
MARKER_FILL = -1
MARKER_RADIUS_7 = 7
MARKER_RADIUS_6 = 6
MARKER_RADIUS_8 = 8
DRAWING_THICKNESS_1 = 1
MARKER_RADIUS_9 = 9
MARKER_RADIUS_10 = 10

TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE_0_55 = 0.55
TEXT_THICKNESS_2 = 2
TEXT_SCALE_0_7 = 0.7
TEXT_SCALE_0_48 = 0.48
TEXT_THICKNESS_1 = 1
TEXT_SCALE_0_5 = 0.5
TEXT_SCALE_0_43 = 0.43
TEXT_SCALE_0_6 = 0.6

TEXT_POSITION_35 = (20, 35)
TEXT_POSITION_65 = (20, 65)
TEXT_POSITION_95 = (20, 95)
TEXT_POSITION_125 = (20, 125)
TEXT_POSITION_155 = (20, 155)
TEXT_POSITION_185 = (20, 185)
TEXT_POSITION_215 = (20, 215)
TEXT_POSITION_245 = (20, 245)
TEXT_POSITION_305 = (20, 305)
TEXT_POSITION_275 = (20, 275)

KERNEL_HSV_CLOSE = np.ones(HSV_CLOSE_KERNEL_SIZE, np.uint8)
KERNEL_HSV_OPEN = np.ones(HSV_OPEN_KERNEL_SIZE, np.uint8)

# 2. UART COMMUNICATION

SERIAL_PORT = "COM6"
SERIAL_BAUDRATE = 115200


class UARTSerialController:
    """Send tracking status to an Arduino over a serial connection."""

    def __init__(
        self,
        port=SERIAL_PORT,
        baudrate=SERIAL_BAUDRATE,
        timeout=0.1,
        write_timeout=0.1,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.serial_port = None
        self.connection_attempted = False

    @property
    def is_connected(self):
        return self.serial_port is not None and self.serial_port.is_open

    def open(self):
        if self.is_connected:
            return True
        if self.connection_attempted:
            return False
        self.connection_attempted = True
        if serial is None:
            return False
        try:
            self.serial_port = serial.Serial(
                self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                write_timeout=self.write_timeout,
            )
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            return True
        except (serial.SerialException, OSError) as exc:
            print(f"[UART] Could not open {self.port}: {exc}")
            try:
                if self.serial_port is not None and self.serial_port.is_open:
                    self.serial_port.close()
            except (serial.SerialException, OSError, ValueError) as close_exc:
                print(f"[UART] Could not close {self.port}: {close_exc}")
            self.serial_port = None
            return False

    def send(self, message):
        if not self.is_connected:
            return False
        try:
            payload = f"{message}\r\n".encode("ascii", errors="ignore")
            self.serial_port.write(payload)
            self.serial_port.flush()
            return True
        except (serial.SerialException, OSError, ValueError) as exc:
            print(f"[UART] Connection lost on {self.port}: {exc}")
            try:
                if self.serial_port is not None and self.serial_port.is_open:
                    self.serial_port.close()
            except (serial.SerialException, OSError, ValueError) as close_exc:
                print(f"[UART] Could not close {self.port}: {close_exc}")
            self.serial_port = None
            return False

    def send_tracking_state(self, state, error_x=0, error_y=0):
        if state == "TRACKING":
            payload = f"TRACKING,{int(error_x)},{int(error_y)}"
        elif state == "VERIFYING":
            payload = "VERIFYING"
        elif state == "SEARCHING":
            payload = "SEARCHING,0,0"
        else:
            raise ValueError(f"Estado UART invalido: {state}")
        return self.send(payload)
    
    def close(self):
        if self.serial_port is not None:
            try:
                if self.serial_port.is_open:
                    self.serial_port.close()
            except (serial.SerialException, OSError, ValueError) as exc:
                print(f"[UART] Could not close {self.port}: {exc}")
            finally:
                self.serial_port = None


def resolve_arduino_state(
    current_state,
    tracking_active,
    has_candidate,
    candidate_center,
    now,
    verification_started_at,
    verification_samples,
    verification_timeout,
    cooldown_until,
    candidate_cleared_after_failure,
):
    if verification_started_at is not None:
        if now - verification_started_at >= verification_timeout:
            return (
                "SEARCHING",
                None,
                (),
                now + VERIFYING_COOLDOWN_SECONDS,
                False,
                True,
            )

        samples = list(verification_samples)
        if has_candidate and candidate_center is not None:
            should_sample = (
                not samples
                or now - samples[-1][0]
                >= VERIFYING_CONFIRMATION_INTERVAL_SECONDS
            )
            if should_sample:
                center = np.asarray(candidate_center, dtype=np.float32)
                if (
                    not samples
                    or np.linalg.norm(center - samples[-1][1])
                    <= MAX_CONFIRMATION_DISTANCE_PX
                ):
                    samples.append((now, center))
                else:
                    samples = [(now, center)]

        verification_elapsed = now - verification_started_at
        if (
            tracking_active
            and has_candidate
            and verification_elapsed >= VERIFYING_MIN_TRACKING_SECONDS
            and len(samples) >= VERIFYING_REQUIRED_DETECTIONS
        ):
            return "TRACKING", None, (), None, True, False

        return (
            "VERIFYING",
            verification_started_at,
            tuple(samples),
            cooldown_until,
            candidate_cleared_after_failure,
            False,
        )

    if cooldown_until is not None:
        if now < cooldown_until:
            return (
                "SEARCHING",
                None,
                (),
                cooldown_until,
                candidate_cleared_after_failure,
                False,
            )
        if not candidate_cleared_after_failure:
            if not has_candidate:
                candidate_cleared_after_failure = True
            return (
                "SEARCHING",
                None,
                (),
                cooldown_until,
                candidate_cleared_after_failure,
                False,
            )
        cooldown_until = None

    if has_candidate and current_state != "TRACKING":
        center = np.asarray(candidate_center, dtype=np.float32)
        return "VERIFYING", now, ((now, center),), cooldown_until, True, False
    if tracking_active:
        return "TRACKING", None, (), cooldown_until, True, False
    return "SEARCHING", None, (), cooldown_until, True, False


def should_send_uart_state(
    state,
    last_sent_state,
    last_searching_sent_at,
    now,
):
    if state == "TRACKING":
        return True
    if state == "VERIFYING":
        return state != last_sent_state
    if state == "SEARCHING":
        return (
            state != last_sent_state
            or last_searching_sent_at is None
            or now - last_searching_sent_at >= SEARCHING_HEARTBEAT_SECONDS
        )
    raise ValueError(f"Estado UART invalido: {state}")


# 2. KALMAN FILTER
# State: [center_x, center_y, vx, vy], with position in pixels and velocity in pixels/s.

class PredictiveKalmanTracker:

    def __init__(
        self,
        measurement_sigma=MEASUREMENT_SIGMA,
        min_acceleration_sigma=MIN_ACCELERATION_SIGMA,
        max_acceleration_sigma=MAX_ACCELERATION_SIGMA,
        max_missed_frames=MAX_MISSED_FRAMES,
        nis_gate=NIS_GATE,
        confirmations_to_start=CONFIRMATIONS_TO_TRACK,
        max_confirmation_distance=MAX_CONFIRMATION_DISTANCE_PX,
    ):

        self.kalman = cv2.KalmanFilter(4, 2)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        self.kalman.measurementMatrix = self.H

        self.measurement_sigma = measurement_sigma

        self.R = np.array([
            [measurement_sigma ** 2, 0],
            [0, measurement_sigma ** 2],
        ], dtype=np.float32)

        self.kalman.measurementNoiseCov = self.R.copy()

        self.P0 = np.array([
            [INITIAL_POSITION_VARIANCE, 0, 0, 0],
            [0, INITIAL_POSITION_VARIANCE, 0, 0],
            [0, 0, INITIAL_VELOCITY_VARIANCE, 0],
            [0, 0, 0, INITIAL_VELOCITY_VARIANCE],
        ], dtype=np.float32)

        self.kalman.errorCovPost = self.P0.copy()

        # Adaptive Q
        # acceleration_sigma represents the allowed deviation from the
        # constant-velocity model.
        self.min_acceleration_sigma = min_acceleration_sigma
        self.max_acceleration_sigma = max_acceleration_sigma
        self.acceleration_sigma = min_acceleration_sigma

        self.process_noise_alpha = PROCESS_NOISE_ALPHA

        self.initialized = False
        self.missed_frames = 0
        self.max_missed_frames = max_missed_frames
        self.last_prediction = None
        self.last_nis = 0.0
        self.nis_gate = float(nis_gate)

        # Initial confirmation before tracking starts.
        self.confirmations_to_start = max(
            1,
            int(confirmations_to_start),
        )
        self.max_confirmation_distance = max(
            1.0,
            float(max_confirmation_distance),
        )
        self.initial_confirmations = 0
        self.confirmation_center = None

    def update_model(self, delta_time):

        delta_time = max(MIN_DT, min(float(delta_time), MAX_DT))

        self.kalman.transitionMatrix = np.array([
            [1, 0, delta_time, 0],
            [0, 1, 0, delta_time],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        dt2 = delta_time ** 2
        dt3 = delta_time ** 3
        dt4 = delta_time ** 4

        q = self.acceleration_sigma ** 2

        Q = q * np.array([
            [dt4 / 4, 0,         dt3 / 2, 0],
            [0,         dt4 / 4, 0,         dt3 / 2],
            [dt3 / 2,   0,       dt2,      0],
            [0,         dt3 / 2, 0,         dt2],
        ], dtype=np.float32)

        self.kalman.processNoiseCov = Q

    def initialize(self, center_x, center_y):

        self.kalman.statePost = np.array([
            [np.float32(center_x)],
            [np.float32(center_y)],
            [0],
            [0],
        ], dtype=np.float32)

        self.kalman.errorCovPost = self.P0.copy()

        self.initialized = True
        self.missed_frames = 0
        self.acceleration_sigma = self.min_acceleration_sigma
        self.last_nis = 0.0

        self.last_prediction = {
            "center_x": float(center_x),
            "center_y": float(center_y),
            "vx": 0.0,
            "vy": 0.0,
        }

    def predict(self, delta_time):

        if not self.initialized:
            return None

        self.update_model(delta_time)

        prediction = self.kalman.predict()

        result = {
            "center_x": float(prediction[0, 0]),
            "center_y": float(prediction[1, 0]),
            "vx": float(prediction[2, 0]),
            "vy": float(prediction[3, 0]),
        }

        self.last_prediction = result

        return result

    def calculate_candidate_nis(self, center_x, center_y):

        if not self.initialized or self.last_prediction is None:
            return 0.0

        z = np.array([
            [np.float32(center_x)],
            [np.float32(center_y)],
        ], dtype=np.float32)

        innovation = (
            z
            -
            self.H @ self.kalman.statePre
        )

        S = (
            self.H
            @ self.kalman.errorCovPre
            @ self.H.T
            +
            self.R
        )

        try:
            nis = float(
                (
                    innovation.T
                    @ np.linalg.solve(S, innovation)
                )[0, 0]
            )

        except np.linalg.LinAlgError:
            return float("inf")

        return max(0.0, nis)

    def temporal_score(self, center_x, center_y):

        if self.last_prediction is None:
            return 1.0, 0.0

        nis = self.calculate_candidate_nis(
            center_x,
            center_y,
        )

        if not np.isfinite(nis):
            return 0.0, nis

        score = np.exp(
            -0.5 * nis
        )

        return float(score), float(nis)

    def correct(self, center_x, center_y):

        # Before initializing the Kalman filter, require the same
        # candidate for several consecutive frames to avoid false locks.
        if not self.initialized:

            current_center = np.array([
                float(center_x),
                float(center_y),
            ], dtype=np.float32)

            if self.confirmation_center is None:
                self.initial_confirmations = 1
            else:
                distance = float(
                    np.linalg.norm(
                        current_center - self.confirmation_center
                    )
                )

                if distance <= self.max_confirmation_distance:
                    self.initial_confirmations += 1
                else:
                    self.initial_confirmations = 1

            self.confirmation_center = current_center

            if (
                self.initial_confirmations
                <
                self.confirmations_to_start
            ):
                return {
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "vx": 0.0,
                    "vy": 0.0,
                    "nis": 0.0,
                    "acceleration_sigma": self.acceleration_sigma,
                    "measurement_accepted": True,
                    "tracking_active": False,
                    "confirming": True,
                    "confirmations": self.initial_confirmations,
                }

            self.initialize(center_x, center_y)
            self.cancel_initial_confirmation()

            return {
                "center_x": float(center_x),
                "center_y": float(center_y),
                "vx": 0.0,
                "vy": 0.0,
                "nis": 0.0,
                "acceleration_sigma": self.acceleration_sigma,
                "measurement_accepted": True,
                "tracking_active": True,
                "confirming": False,
                "confirmations": self.confirmations_to_start,
            }

        z = np.array([
            [np.float32(center_x)],
            [np.float32(center_y)],
        ], dtype=np.float32)

        # Innovation = measurement - prediction
        innovation = (
            z
            -
            self.H @ self.kalman.statePre
        )

        # S = innovation covariance
        S = (
            self.H
            @ self.kalman.errorCovPre
            @ self.H.T
            +
            self.R
        )

        # NIS - Normalized Innovation Squared
        # Measures how far the measurement is from the prediction,
        # accounting for the filter's uncertainty.
        try:
            nis = float(
                (
                    innovation.T
                    @ np.linalg.solve(S, innovation)
                )[0, 0]
            )

        except np.linalg.LinAlgError:
            nis = float("inf")

        self.last_nis = nis

        if (not np.isfinite(nis)) or nis > self.nis_gate:

            self.missed_frames += 1

            predicted_state = self.kalman.statePre.copy()

            if self.missed_frames > self.max_missed_frames:

                predicted_x = float(predicted_state[0, 0])
                predicted_y = float(predicted_state[1, 0])
                predicted_vx = float(predicted_state[2, 0])
                predicted_vy = float(predicted_state[3, 0])

                self.reset()

                return {
                    "center_x": predicted_x,
                    "center_y": predicted_y,
                    "vx": predicted_vx,
                    "vy": predicted_vy,
                    "nis": nis,
                    "acceleration_sigma": self.acceleration_sigma,
                    "measurement_accepted": False,
                    "tracking_active": False,
                }

            self.kalman.statePost = predicted_state
            self.kalman.errorCovPost = self.kalman.errorCovPre.copy()

            return {
                "center_x": float(self.kalman.statePost[0, 0]),
                "center_y": float(self.kalman.statePost[1, 0]),
                "vx": float(self.kalman.statePost[2, 0]),
                "vy": float(self.kalman.statePost[3, 0]),
                "nis": nis,
                "acceleration_sigma": self.acceleration_sigma,
                "measurement_accepted": False,
                "tracking_active": True,
            }

        # Only a measurement that PASSED gating can change acceleration_sigma/Q.
        # A rejected false positive must not change the motion model.
        if nis <= MIN_NIS_ADAPTATION:
            target_sigma = self.min_acceleration_sigma

        elif nis >= MAX_NIS_ADAPTATION:
            target_sigma = self.max_acceleration_sigma

        else:
            factor = (
                (nis - MIN_NIS_ADAPTATION)
                /
                (MAX_NIS_ADAPTATION - MIN_NIS_ADAPTATION)
            )

            target_sigma = (
                self.min_acceleration_sigma
                +
                factor
                *
                (
                    self.max_acceleration_sigma
                    -
                    self.min_acceleration_sigma
                )
            )

        self.acceleration_sigma = (
            (1 - self.process_noise_alpha) * self.acceleration_sigma
            +
            self.process_noise_alpha * target_sigma
        )

        state = self.kalman.correct(z)

        self.missed_frames = 0

        return {
            "center_x": float(state[0, 0]),
            "center_y": float(state[1, 0]),
            "vx": float(state[2, 0]),
            "vy": float(state[3, 0]),
            "nis": nis,
            "acceleration_sigma": self.acceleration_sigma,
            "measurement_accepted": True,
            "tracking_active": True,
        }

    def cancel_initial_confirmation(self):
        """Cancel an incomplete confirmation while searching."""
        self.initial_confirmations = 0
        self.confirmation_center = None

    def accept_prediction_without_measurement(self):

        if not self.initialized:
            return False

        self.missed_frames += 1

        if self.missed_frames > self.max_missed_frames:
            self.reset()
            return False

        self.kalman.statePost = self.kalman.statePre.copy()
        self.kalman.errorCovPost = self.kalman.errorCovPre.copy()

        return True

    def reset(self):

        self.initialized = False
        self.missed_frames = 0
        self.last_prediction = None
        self.last_nis = 0.0
        self.acceleration_sigma = self.min_acceleration_sigma
        self.cancel_initial_confirmation()

# 3. HSV SEGMENTATION

def create_hsv_mask(hsv, vmin, zoom_factor=1):
    low = np.array([H_MIN, S_MIN, int(vmin)], dtype=np.uint8)
    high = np.array([H_MAX, S_MAX, V_MAX], dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)

    if zoom_factor == 1:
        kernel_close = KERNEL_HSV_CLOSE
        kernel_open = KERNEL_HSV_OPEN
    else:
        kernel_close = np.ones(
            scale_odd_kernel(HSV_CLOSE_KERNEL_SIZE, zoom_factor), np.uint8
        )
        kernel_open = np.ones(
            scale_odd_kernel(HSV_OPEN_KERNEL_SIZE, zoom_factor), np.uint8
        )

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    return mask

def create_mask(image, vmin=REFERENCE_V_MIN):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    return create_hsv_mask(
        hsv,
        vmin,
    )

# 4. ADAPTIVE HSV

class AdaptiveVController:

    def __init__(self):

        self.current_v_min = int(INITIAL_V_MIN)
        self.filtered_brightness = None
        self.v_base = int(INITIAL_V_MIN)

        self.last_mode = "INITIAL"
        self.last_area = 0.0
        self.last_visual_score = 0.0
        self.last_shape_score = 0.0
        self.last_area_score = 0.0
        self.last_adjustment_score = 0.0
        self.last_tests = []
        self.last_roi = None

    def update_brightness(self, hsv):

        v_channel = hsv[:, :, 2]

        # The median is more robust to windows, lamps, and reflections
        # than a simple average of the entire image.
        current_brightness = float(
            np.median(v_channel)
        )

        if self.filtered_brightness is None:
            self.filtered_brightness = current_brightness
        else:
            self.filtered_brightness = (
                LIGHT_ALPHA * current_brightness
                +
                (1.0 - LIGHT_ALPHA)
                * self.filtered_brightness
            )

        self.v_base = int(
            np.clip(
                AMBIENT_LIGHT_FACTOR
                * self.filtered_brightness,
                V_MIN_LOWER_LIMIT,
                V_MIN_UPPER_LIMIT,
            )
        )

        return current_brightness

    def limit_mask_roi(self, mask, prediction):

        if prediction is None:
            self.last_roi = None
            return mask

        height, width = mask.shape[:2]

        center_x = int(round(prediction["center_x"]))
        center_y = int(round(prediction["center_y"]))

        x1 = max(0, center_x - V_ADJUSTMENT_ROI_RADIUS)
        y1 = max(0, center_y - V_ADJUSTMENT_ROI_RADIUS)
        x2 = min(width, center_x + V_ADJUSTMENT_ROI_RADIUS)
        y2 = min(height, center_y + V_ADJUSTMENT_ROI_RADIUS)

        if x2 <= x1 or y2 <= y1:
            self.last_roi = None
            return mask

        roi_mask = np.zeros_like(mask)
        roi_mask[y1:y2, x1:x2] = 255

        self.last_roi = (
            x1,
            y1,
            x2,
            y2,
        )

        return cv2.bitwise_and(
            mask,
            roi_mask,
        )

    def evaluate_v_min(
        self,
        image,
        hsv,
        vmin,
        prediction,
        processed_reference,
        reference_aspect_ratio,
        normalized_width,
        normalized_height,
    ):

        full_mask = create_hsv_mask(
            hsv,
            vmin,
        )

        evaluation_mask = self.limit_mask_roi(
            full_mask,
            prediction,
        )

        objects = find_quadrilaterals(
            evaluation_mask,
            min_area=MIN_ADJUSTMENT_AREA,
        )

        best = None

        for object in objects:

            points = order_points(
                object["points"]
            )

            aspect_ratio = calculate_aspect_ratio(
                points
            )

            if reference_aspect_ratio > 0:
                shape_error = abs(
                    aspect_ratio
                    -
                    reference_aspect_ratio
                ) / reference_aspect_ratio
            else:
                shape_error = 999.0

            # V adjustment intentionally uses a wider shape gate
            # than final recognition.
            if shape_error > ADJUSTMENT_SHAPE_TOLERANCE:
                continue

            shape_score = float(
                np.clip(
                    1.0
                    -
                    shape_error / ADJUSTMENT_SHAPE_TOLERANCE,
                    0.0,
                    1.0,
                )
            )

            (
                visual_score,
                _,
                _,
            ) = best_normalization(
                image,
                points,
                processed_reference,
                normalized_width,
                normalized_height,
            )

            # First, choose the candidate with the best combination
            # of appearance and shape. Area is added later when
            # comparing the tested V values.
            preliminary_score = (
                PRELIMINARY_VISUAL_WEIGHT * visual_score
                +
                PRELIMINARY_SHAPE_WEIGHT * shape_score
            )

            candidate = {
                "vmin": int(vmin),
                "mask": full_mask,
                "area": float(object["area"]),
                "visual_score": float(visual_score),
                "shape_score": float(shape_score),
                "preliminary_score": float(preliminary_score),
                "points": points,
            }

            if (
                best is None
                or
                candidate["preliminary_score"]
                > best["preliminary_score"]
            ):
                best = candidate

        if best is None:
            return {
                "vmin": int(vmin),
                "mask": full_mask,
                "area": 0.0,
                "visual_score": 0.0,
                "shape_score": 0.0,
                "preliminary_score": 0.0,
                "area_score": 0.0,
                "adjustment_score": 0.0,
                "points": None,
            }

        best["area_score"] = 0.0
        best["adjustment_score"] = 0.0

        return best

    def update(
        self,
        image,
        hsv,
        frame_number,
        prediction,
        processed_reference,
        reference_aspect_ratio,
        normalized_width,
        normalized_height,
    ):

        current_brightness = self.update_brightness(
            hsv
        )

        if frame_number == 1:
            self.current_v_min = self.v_base

        should_adjust = (
            frame_number == 1
            or
            frame_number % ADJUST_V_EVERY == 0
        )

        if not should_adjust:

            mask = create_hsv_mask(
                hsv,
                self.current_v_min,
            )

            return {
                "mask": mask,
                "vmin": self.current_v_min,
                "current_brightness": current_brightness,
                "filtered_brightness": self.filtered_brightness,
                "v_base": self.v_base,
                "adjusted": False,
                "mode": self.last_mode,
                "area": self.last_area,
                "visual_score": self.last_visual_score,
                "shape_score": self.last_shape_score,
                "area_score": self.last_area_score,
                "adjustment_score": self.last_adjustment_score,
                "tests": self.last_tests,
                "roi": self.last_roi,
            }

        center = int(round(
            CURRENT_V_CENTER_WEIGHT * self.current_v_min
            +
            AMBIENT_V_CENTER_WEIGHT * self.v_base
        ))

        test_values = sorted(set([
            int(self.current_v_min),
            int(np.clip(
                center - DELTA_V,
                V_MIN_LOWER_LIMIT,
                V_MIN_UPPER_LIMIT,
            )),
            int(np.clip(
                center,
                V_MIN_LOWER_LIMIT,
                V_MIN_UPPER_LIMIT,
            )),
            int(np.clip(
                center + DELTA_V,
                V_MIN_LOWER_LIMIT,
                V_MIN_UPPER_LIMIT,
            )),
            int(self.v_base),
        ]))

        results = []

        for test_v in test_values:

            result = self.evaluate_v_min(
                image,
                hsv,
                test_v,
                prediction,
                processed_reference,
                reference_aspect_ratio,
                normalized_width,
                normalized_height,
            )

            results.append(result)



        max_area = max(
            [r["area"] for r in results]
            + [1.0]
        )

        for result in results:

            result["area_score"] = float(
                np.clip(
                    result["area"] / max_area,
                    0.0,
                    1.0,
                )
            )

            result["adjustment_score"] = (
                ADJUSTMENT_VISUAL_WEIGHT
                * result["visual_score"]
                +
                ADJUSTMENT_SHAPE_WEIGHT
                * result["shape_score"]
                +
                ADJUSTMENT_AREA_WEIGHT
                * result["area_score"]
            )

        best = max(
            results,
            key=lambda r: r["adjustment_score"],
        )

        current = min(
            results,
            key=lambda r: abs(
                r["vmin"] - self.current_v_min
            ),
        )

        improvement = (
            best["adjustment_score"]
            -
            current["adjustment_score"]
        )

        should_switch = (
            (
                current["area"] <= 0
                and
                best["area"] > 0
            )
            or
            improvement >= MIN_V_SWITCH_GAIN
        )

        if should_switch:
            self.current_v_min = int(
                best["vmin"]
            )
            selected = best
            self.last_mode = "ADJUSTED"
        else:
            if current["vmin"] == self.current_v_min:
                selected = current
            else:
                selected = self.evaluate_v_min(
                    image,
                    hsv,
                    self.current_v_min,
                    prediction,
                    processed_reference,
                    reference_aspect_ratio,
                    normalized_width,
                    normalized_height,
                )

                selected["area_score"] = float(
                    np.clip(
                        selected["area"] / max_area,
                        0.0,
                        1.0,
                    )
                )
                selected["adjustment_score"] = (
                    ADJUSTMENT_VISUAL_WEIGHT * selected["visual_score"]
                    +
                    ADJUSTMENT_SHAPE_WEIGHT * selected["shape_score"]
                    +
                    ADJUSTMENT_AREA_WEIGHT * selected["area_score"]
                )

            self.last_mode = "UNCHANGED"

        self.last_area = selected["area"]
        self.last_visual_score = selected["visual_score"]
        self.last_shape_score = selected["shape_score"]
        self.last_area_score = selected["area_score"]
        self.last_adjustment_score = selected["adjustment_score"]

        self.last_tests = [
            {
                "v": r["vmin"],
                "area": r["area"],
                "J": r["adjustment_score"],
            }
            for r in results
        ]

        return {
            "mask": selected["mask"],
            "vmin": self.current_v_min,
            "current_brightness": current_brightness,
            "filtered_brightness": self.filtered_brightness,
            "v_base": self.v_base,
            "adjusted": True,
            "mode": self.last_mode,
            "area": self.last_area,
            "visual_score": self.last_visual_score,
            "shape_score": self.last_shape_score,
            "area_score": self.last_area_score,
            "adjustment_score": self.last_adjustment_score,
            "tests": self.last_tests,
            "roi": self.last_roi,
        }

# 5. GEOMETRY AND APPEARANCE

def order_points(points):

    points = np.array(
        points,
        dtype=np.float32,
    )

    result = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    channel_sum = points.sum(axis=1)

    difference = np.diff(
        points,
        axis=1,
    ).flatten()

    result[0] = points[np.argmin(channel_sum)]
    result[2] = points[np.argmax(channel_sum)]
    result[1] = points[np.argmin(difference)]
    result[3] = points[np.argmax(difference)]

    return result

def distance(p1, p2):

    return np.linalg.norm(
        np.array(p1)
        -
        np.array(p2)
    )

def calculate_aspect_ratio(points):

    pts = order_points(points)

    top_left, top_right, bottom_right, bottom_left = pts

    width_1 = distance(top_left, top_right)
    width_2 = distance(bottom_left, bottom_right)

    height_1 = distance(top_left, bottom_left)
    height_2 = distance(top_right, bottom_right)

    width = (
        width_1 + width_2
    ) / 2

    height = (
        height_1 + height_2
    ) / 2

    if width == 0 or height == 0:
        return 0

    larger = max(width, height)
    smaller = min(width, height)

    return larger / smaller

def crop_roi_from_points(
    image,
    points,
    margin=INDEPENDENT_CONTOUR_MARGIN,
):
    """Crop a BGR ROI around the candidate's four points."""

    pts = np.asarray(
        points,
        dtype=np.int32,
    )

    x, y, w, h = cv2.boundingRect(pts)

    if w <= 0 or h <= 0:
        return None

    margin_x = int(round(w * margin))
    margin_y = int(round(h * margin))

    height, width = image.shape[:2]

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(width, x + w + margin_x)
    y2 = min(height, y + h + margin_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2].copy()

def extract_independent_contour(
    image,
    points,
    processing_scale=1,
):
    """
    Extract a contour from BGR, grayscale, and Canny without using HSV.

    Candidate points define the ROI to avoid processing the entire frame.
    """

    roi = crop_roi_from_points(
        image,
        points,
    )

    if roi is None or roi.size == 0:
        return None

    grayscale = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY,
    )

    grayscale = cv2.equalizeHist(grayscale)

    if processing_scale == 1:
        contour_blur_size = CONTOUR_BLUR_SIZE
        contour_blur_sigma = CONTOUR_BLUR_SIGMA
        contour_kernel_size = CONTOUR_KERNEL_SIZE
    else:
        contour_blur_size = scale_odd_kernel(
            CONTOUR_BLUR_SIZE,
            processing_scale,
        )
        contour_blur_sigma = CONTOUR_BLUR_SIGMA * processing_scale
        contour_kernel_size = scale_odd_kernel(
            CONTOUR_KERNEL_SIZE,
            processing_scale,
        )

    gray_blur = cv2.GaussianBlur(
        grayscale,
        contour_blur_size,
        contour_blur_sigma,
    )

    median_value = float(np.median(gray_blur))

    lower_threshold = int(
        max(0, LOW_CANNY_FACTOR * median_value)
    )

    upper_threshold = int(
        min(255, HIGH_CANNY_FACTOR * median_value)
    )

    if upper_threshold <= lower_threshold:
        upper_threshold = min(
            255,
            lower_threshold + MIN_CANNY_INCREMENT,
        )

    edges = cv2.Canny(
        gray_blur,
        lower_threshold,
        upper_threshold,
    )

    contour_kernel = np.ones(
        contour_kernel_size,
        np.uint8,
    )

    edges = cv2.morphologyEx(
        edges,
        CONTOUR_CLOSE_OPERATION,
        contour_kernel,
        iterations=CONTOUR_CLOSE_ITERATIONS,
    )

    edges = cv2.dilate(
        edges,
        contour_kernel,
        iterations=CONTOUR_DILATION_ITERATIONS,
    )

    contours, _ = cv2.findContours(
        edges,
        CONTOUR_RETRIEVAL_MODE,
        CONTOUR_APPROXIMATION_METHOD,
    )

    if not contours:
        return None

    roi_height, roi_width = grayscale.shape[:2]
    roi_area = float(
        roi_height * roi_width
    )

    roi_center_x = roi_width / 2.0
    roi_center_y = roi_height / 2.0
    diagonal_roi = max(
        1.0,
        np.hypot(roi_width, roi_height),
    )

    best_contour = None
    best_score = -1.0

    for contour in contours:

        area = float(
            cv2.contourArea(contour)
        )

        if area < MIN_CONTOUR_AREA_FRACTION * roi_area:
            continue

        moments = cv2.moments(contour)

        if abs(moments["m00"]) > GEOMETRY_EPSILON:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
        else:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w / 2.0
            center_y = y + h / 2.0

        center_distance = np.hypot(
            center_x - roi_center_x,
            center_y - roi_center_y,
        ) / diagonal_roi

        centrality = float(
            np.clip(
                1.0 - CENTRALITY_PENALTY_FACTOR * center_distance,
                MIN_CENTRALITY,
                1.0,
            )
        )

        score = area * centrality

        if score > best_score:
            best_score = score
            best_contour = contour

    return best_contour

def calculate_rectangularity(contour):
    """
    Rectangularity is contour area divided by minimum rotated rectangle area.
    A value of 1.0 means the contour fills the rectangle. This is independent
    of HSV color.
    """

    if contour is None or len(contour) < 3:
        return None

    area = float(
        cv2.contourArea(contour)
    )

    rectangle = cv2.minAreaRect(contour)
    width, height = rectangle[1]

    rectangle_area = float(
        width * height
    )

    if area <= 0 or rectangle_area <= GEOMETRY_EPSILON:
        return None

    return float(
        np.clip(
            area / rectangle_area,
            0.0,
            1.0,
        )
    )

def rectangularity_score(
    reference_image,
    candidate,
):
    """Compare rectangularity values and return a score from 0 to 1."""

    if reference_image is None or candidate is None:
        return None

    scale = max(
        abs(float(reference_image)),
        MIN_RECTANGULARITY_SCALE,
    )

    relative_error = abs(
        float(candidate)
        -
        float(reference_image)
    ) / scale

    return float(
        np.clip(
            1.0 - relative_error,
            0.0,
            1.0,
        )
    )

def compare_match_shapes(
    reference_contour,
    candidate_contour,
):
    """
    Compare Hu moments using cv2.matchShapes.

    Returns:
        shape_error: lower is better
        shape_score: 0 to 1, higher is better
    """

    if (
        reference_contour is None
        or
        candidate_contour is None
    ):
        return None, None

    try:
        error = float(
            cv2.matchShapes(
                reference_contour,
                candidate_contour,
                MATCH_SHAPES_METHOD,
                0.0,
            )
        )
    except cv2.error:
        return None, None

    if not np.isfinite(error):
        return None, None

    error = max(0.0, error)

    score = float(
        np.exp(
            -MATCH_SHAPES_FACTOR * error
        )
    )

    return error, score

def calculate_normalized_rgb(image):
    """
    Calculate a per-pixel normalized [r, g, b] signature:

        r = R / (R+G+B)
        g = G / (R+G+B)
        b = B / (R+G+B)

    This reduces the effect of approximately multiplicative brightness
    changes. HSV is not used.
    """

    if image is None or image.size == 0:
        return None

    frame = image.astype(
        np.float32
    )

    b, g, r = cv2.split(frame)

    channel_sum = r + g + b

    valid_pixels = channel_sum > MIN_RGB_SUM

    if not np.any(valid_pixels):
        return None

    safe_sum = channel_sum + DESCRIPTOR_EPSILON

    rn = r / safe_sum
    gn = g / safe_sum
    bn = b / safe_sum

    signature = np.array([
        np.median(rn[valid_pixels]),
        np.median(gn[valid_pixels]),
        np.median(bn[valid_pixels]),
    ], dtype=np.float32)

    return signature

def normalized_rgb_score(
    reference_signature,
    candidate_signature,
):
    """Compare normalized RGB signatures and return a score from 0 to 1."""

    if (
        reference_signature is None
        or
        candidate_signature is None
    ):
        return None

    rgb_distance = float(
        np.linalg.norm(
            reference_signature
            -
            candidate_signature
        )
    )

    sigma = max(
        DESCRIPTOR_EPSILON,
        float(NORMALIZED_RGB_SIGMA),
    )

    score = np.exp(
        -0.5
        *
        (rgb_distance / sigma) ** 2
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )

def calculate_texture_descriptor(image):
    """
    Lightweight grayscale texture descriptor.

    Components:
      1) intensity standard deviation
      2) Laplacian energy
      3) edge density
      4) local contrast

    Equalization reduces some dependence on illumination.
    """

    if image is None or image.size == 0:
        return None

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    grayscale = cv2.resize(
        grayscale,
        TEXTURE_IMAGE_SIZE,
        interpolation=TEXTURE_INTERPOLATION,
    )

    grayscale = cv2.equalizeHist(grayscale)

    gray_blur = cv2.GaussianBlur(
        grayscale,
        TEXTURE_BLUR_SIZE,
        TEXTURE_BLUR_SIGMA,
    )

    deviation = float(
        np.std(grayscale)
    ) / TEXTURE_STDDEV_SCALE

    laplaciano = cv2.Laplacian(
        gray_blur,
        cv2.CV_32F,
        ksize=LAPLACIAN_KERNEL_SIZE,
    )

    laplacian_energy = float(
        np.mean(
            np.abs(laplaciano)
        )
    ) / TEXTURE_LAPLACIAN_SCALE

    median_value = float(
        np.median(gray_blur)
    )

    lower_threshold = int(
        max(0, LOW_CANNY_FACTOR * median_value)
    )
    upper_threshold = int(
        min(255, HIGH_CANNY_FACTOR * median_value)
    )

    if upper_threshold <= lower_threshold:
        upper_threshold = min(
            255,
            lower_threshold + MIN_CANNY_INCREMENT,
        )

    edges = cv2.Canny(
        gray_blur,
        lower_threshold,
        upper_threshold,
    )

    edge_density = float(
        np.count_nonzero(edges)
        /
        edges.size
    )

    smoothed = cv2.GaussianBlur(
        grayscale.astype(np.float32),
        CONTRAST_BLUR_SIZE,
        CONTRAST_BLUR_SIGMA,
    )

    local_contrast = float(
        np.mean(
            np.abs(
                grayscale.astype(np.float32)
                -
                smoothed
            )
        )
    ) / TEXTURE_CONTRAST_SCALE

    descriptor = np.array([
        np.clip(deviation, 0.0, 1.0),
        np.clip(laplacian_energy, 0.0, 1.0),
        np.clip(edge_density, 0.0, 1.0),
        np.clip(local_contrast, 0.0, 1.0),
    ], dtype=np.float32)

    return descriptor

def calculate_texture_score(
    reference_descriptor,
    candidate_descriptor,
):
    """Compare texture descriptors and return a score from 0 to 1."""

    if (
        reference_descriptor is None
        or
        candidate_descriptor is None
    ):
        return None

    distance = float(
        np.linalg.norm(
            reference_descriptor
            -
            candidate_descriptor
        )
        /
        np.sqrt(len(reference_descriptor))
    )

    sigma = max(
        DESCRIPTOR_EPSILON,
        float(TEXTURE_SIGMA),
    )

    score = np.exp(
        -0.5
        *
        (distance / sigma) ** 2
    )

    return float(
        np.clip(score, 0.0, 1.0)
    )

def combine_robustness_scores(scores):
    """
    Weighted mean of the available scores.

    If an independent contour is unavailable in a frame, the
    rectangularity and matchShapes descriptors do not force the
    object's score to zero.
    """

    weights = {
        "template": TEMPLATE_ROBUSTNESS_WEIGHT,
        "aspect_ratio": PROPORTION_ROBUSTNESS_WEIGHT,
        "rectangularity": RECTANGULARITY_WEIGHT,
        "shape": MATCH_SHAPES_WEIGHT,
        "rgb": NORMALIZED_RGB_WEIGHT,
        "texture": TEXTURE_WEIGHT,
    }

    channel_sum = 0.0
    weight_sum = 0.0

    for name, weight in weights.items():

        value = scores.get(name)

        if value is None:
            continue

        if not np.isfinite(value):
            continue

        value = float(
            np.clip(value, 0.0, 1.0)
        )

        channel_sum += weight * value
        weight_sum += weight

    if weight_sum <= 0:
        return 0.0

    return float(
        channel_sum / weight_sum
    )

# 6. NORMALIZATION AND CANDIDATES

def correct_perspective(
    image,
    points,
    output_width,
    output_height,
):

    points = np.array(
        points,
        dtype=np.float32,
    )

    destination = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1],
    ], dtype=np.float32)

    transform_matrix = cv2.getPerspectiveTransform(
        points,
        destination,
    )

    result = cv2.warpPerspective(
        image,
        transform_matrix,
        (
            output_width,
            output_height,
        ),
    )

    return result

def prepare_image(image):

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    grayscale = cv2.GaussianBlur(
        grayscale,
        TEMPLATE_BLUR_SIZE,
        TEMPLATE_BLUR_SIGMA,
    )

    grayscale = cv2.equalizeHist(grayscale)

    return grayscale

def compare_images(
    reference_image,
    candidate,
):

    result = cv2.matchTemplate(
        candidate,
        reference_image,
        TEMPLATE_MATCH_METHOD,
    )

    score = float(result[0][0])

    score = max(
        0.0,
        min(score, 1.0),
    )

    return score

def best_normalization(
    image,
    points,
    processed_reference,
    output_width,
    output_height,
):

    points = order_points(points)

    best_score = -1
    best_image = None
    best_rotation = 0

    for i in range(4):

        rotated_points = np.roll(
            points,
            -i,
            axis=0,
        )

        normalized = correct_perspective(
            image,
            rotated_points,
            output_width,
            output_height,
        )

        processed = prepare_image(
            normalized
        )

        score = compare_images(
            processed_reference,
            processed,
        )

        if score > best_score:
            best_score = score
            best_image = normalized
            best_rotation = i * 90

    return (
        best_score,
        best_image,
        best_rotation,
    )

def find_quadrilaterals(mask, min_area=None):

    contours, _ = cv2.findContours(
        mask,
        CONTOUR_RETRIEVAL_MODE,
        CONTOUR_APPROXIMATION_METHOD,
    )

    objects = []

    if min_area is None:
        min_area = MIN_AREA

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        approximation = cv2.approxPolyDP(
            contour,
            POLYGON_APPROXIMATION_FACTOR * perimeter,
            True,
        )

        if len(approximation) == 4:

            points = approximation.reshape(
                4,
                2,
            )

            objects.append({
                "contour": contour,
                "points": points,
                "area": area,
            })

        else:

            rectangle = cv2.minAreaRect(
                contour
            )

            rect_width, rect_height = rectangle[1]
            rectangle_area = float(rect_width * rect_height)

            if rectangle_area <= GEOMETRY_EPSILON:
                continue

            fill_ratio = float(
                np.clip(
                    area / rectangle_area,
                    0.0,
                    1.0,
                )
            )

            if fill_ratio < MIN_FALLBACK_RECTANGULARITY:
                continue

            box = cv2.boxPoints(
                rectangle
            )

            box = np.int32(box)

            objects.append({
                "contour": contour,
                "points": box,
                "area": area,
                "fallback_rectangularity": fill_ratio,
            })

    return objects

def scale_odd_kernel(kernel_size, scale):
    return tuple(
        max(
            1,
            int(round(size * scale))
            + (int(round(size * scale)) % 2 == 0),
        )
        for size in kernel_size
    )

def create_zoom_hsv_mask(hsv, vmin, zoom_factor):
    return create_hsv_mask(hsv, vmin, zoom_factor=zoom_factor)

def expand_zoom_roi(bounds, image_shape, margin):
    x1, y1, x2, y2 = bounds
    image_height, image_width = image_shape[:2]

    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(image_width, x2 + margin),
        min(image_height, y2 + margin),
    )

def evaluate_zoom_roi(
    frame,
    roi_bounds,
    vmin,
    prediction,
    tracker,
    zoom_factor,
):
    x1, y1, x2, y2 = roi_bounds
    if x2 <= x1 or y2 <= y1:
        return None
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    zoom_size = (
        roi.shape[1] * zoom_factor,
        roi.shape[0] * zoom_factor,
    )
    zoom_frame = cv2.resize(
        roi,
        zoom_size,
        interpolation=cv2.INTER_CUBIC,
    )
    zoom_hsv = cv2.cvtColor(
        zoom_frame,
        cv2.COLOR_BGR2HSV,
    )
    zoom_mask = create_zoom_hsv_mask(
        zoom_hsv,
        vmin,
        zoom_factor,
    )
    zoom_objects = find_quadrilaterals(
        zoom_mask,
        min_area=ZOOM_MIN_AREA_PX2 * zoom_factor ** 2,
    )

    valid_candidates = []
    for zoom_object in zoom_objects:
        points = order_points(zoom_object["points"])
        current_aspect_ratio = calculate_aspect_ratio(points)
        if REFERENCE_ASPECT_RATIO > 0:
            shape_error = abs(
                current_aspect_ratio - REFERENCE_ASPECT_RATIO
            ) / REFERENCE_ASPECT_RATIO
        else:
            shape_error = 999

        valid_shape = shape_error <= SHAPE_TOLERANCE
        if not valid_shape:
            continue

        aspect_ratio_score = float(
            np.clip(
                1.0 - shape_error / SHAPE_TOLERANCE,
                0.0,
                1.0,
            )
        )
        (
            visual_score,
            normalized_image,
            rotation,
        ) = best_normalization(
            zoom_frame,
            points,
            processed_reference,
            NORMALIZED_WIDTH,
            NORMALIZED_HEIGHT,
        )
        independent_contour = extract_independent_contour(
            zoom_frame,
            points,
            processing_scale=zoom_factor,
        )
        current_rectangularity = calculate_rectangularity(
            independent_contour
        )
        rectangularity = rectangularity_score(
            REFERENCE_RECTANGULARITY,
            current_rectangularity,
        )
        _, shape_score = compare_match_shapes(
            REFERENCE_INDEPENDENT_CONTOUR,
            independent_contour,
        )
        current_rgb = calculate_normalized_rgb(normalized_image)
        rgb_score = normalized_rgb_score(REFERENCE_RGB, current_rgb)
        current_texture = calculate_texture_descriptor(normalized_image)
        texture_score = calculate_texture_score(
            REFERENCE_TEXTURE,
            current_texture,
        )
        robustness_score = combine_robustness_scores({
            "template": visual_score,
            "aspect_ratio": aspect_ratio_score,
            "rectangularity": rectangularity,
            "shape": shape_score,
            "rgb": rgb_score,
            "texture": texture_score,
        })

        if robustness_score < ROBUSTNESS_SCORE_THRESHOLD:
            continue

        center_x = x1 + float(np.mean(points[:, 0])) / zoom_factor
        center_y = y1 + float(np.mean(points[:, 1])) / zoom_factor
        temporal_score = 1.0
        candidate_nis = 0.0
        if prediction is not None:
            temporal_score, candidate_nis = tracker.temporal_score(
                center_x,
                center_y,
            )
            if (
                not np.isfinite(candidate_nis)
                or candidate_nis > tracker.nis_gate
            ):
                continue
            total_score = (
                FINAL_ROBUSTNESS_WEIGHT * robustness_score
                + TEMPORAL_WEIGHT * temporal_score
            )
        else:
            total_score = robustness_score

        box_x, box_y, box_width, box_height = cv2.boundingRect(
            zoom_object["contour"]
        )
        original_points = points.astype(np.float32) / zoom_factor
        original_points[:, 0] += x1
        original_points[:, 1] += y1
        valid_candidates.append({
            "score": robustness_score,
            "robustness_score": robustness_score,
            "visual_score": visual_score,
            "rectangularity_score": rectangularity,
            "match_shapes_score": shape_score,
            "rgb_score": rgb_score,
            "texture_score": texture_score,
            "temporal_score": temporal_score,
            "candidate_nis": candidate_nis,
            "total_score": total_score,
            "rotation": rotation,
            "normalized": normalized_image,
            "points": original_points,
            "zoom_points": points,
            "center": (center_x, center_y),
            "area": (box_width * box_height) / zoom_factor ** 2,
            "bounds": (
                x1 + box_x / zoom_factor,
                y1 + box_y / zoom_factor,
                x1 + (box_x + box_width) / zoom_factor,
                y1 + (box_y + box_height) / zoom_factor,
            ),
        })

    if not valid_candidates:
        return None

    return max(
        valid_candidates,
        key=lambda candidate: candidate["total_score"],
    )

def get_zoom_roi_bounds(
    objects,
    frame_shape,
    prediction,
    tracker,
    last_target_size,
):
    if objects:
        roi_bounds = []
        for detected_object in objects:
            x, y, width, height = cv2.boundingRect(
                detected_object["contour"]
            )
            roi_bounds.append(
                expand_zoom_roi(
                    (x, y, x + width, y + height),
                    frame_shape,
                    ZOOM_MARGIN,
                )
            )
        return roi_bounds, "hsv_contour"

    prediction_is_valid = (
        prediction is not None
        and tracker.initialized
        and tracker.missed_frames < tracker.max_missed_frames
        and all(
            np.isfinite(prediction[key])
            for key in ("center_x", "center_y", "vx", "vy")
        )
    )
    if not prediction_is_valid:
        return [], "NOT_AVAILABLE"

    if last_target_size is None:
        width = max(
            20,
            int(round(np.sqrt(MIN_AREA * REFERENCE_ASPECT_RATIO))),
        )
        height = max(
            20,
            int(round(np.sqrt(MIN_AREA / REFERENCE_ASPECT_RATIO))),
        )
    else:
        width, height = last_target_size

    center_x = int(round(prediction["center_x"]))
    center_y = int(round(prediction["center_y"]))
    bounds = expand_zoom_roi(
        (
            center_x - width // 2,
            center_y - height // 2,
            center_x + (width + 1) // 2,
            center_y + (height + 1) // 2,
        ),
        frame_shape,
        ZOOM_MARGIN,
    )
    return [bounds], "kalman_prediction"


def evaluate_zoom_rois(
    frame,
    roi_bounds,
    vmin,
    prediction,
    tracker,
    zoom_factor,
):
    candidates = []
    for bounds in roi_bounds:
        candidate = evaluate_zoom_roi(
            frame,
            bounds,
            vmin,
            prediction,
            tracker,
            zoom_factor,
        )
        if candidate is not None:
            candidates.append(candidate)
    candidates = deduplicate_zoom_candidates(candidates)
    return (
        max(candidates, key=lambda item: item["total_score"])
        if candidates else None
    )


def deduplicate_zoom_candidates(candidates):
    unique_candidates = []
    for candidate in sorted(
        candidates,
        key=lambda item: item["total_score"],
        reverse=True,
    ):
        candidate_x1, candidate_y1, candidate_x2, candidate_y2 = (
            candidate["bounds"]
        )
        candidate_area = max(
            0.0,
            candidate_x2 - candidate_x1,
        ) * max(
            0.0,
            candidate_y2 - candidate_y1,
        )
        duplicate = False

        for existing in unique_candidates:
            existing_x1, existing_y1, existing_x2, existing_y2 = (
                existing["bounds"]
            )
            intersection_width = max(
                0.0,
                min(candidate_x2, existing_x2)
                - max(candidate_x1, existing_x1),
            )
            intersection_height = max(
                0.0,
                min(candidate_y2, existing_y2)
                - max(candidate_y1, existing_y1),
            )
            intersection = intersection_width * intersection_height
            existing_area = max(
                0.0,
                existing_x2 - existing_x1,
            ) * max(
                0.0,
                existing_y2 - existing_y1,
            )
            union = candidate_area + existing_area - intersection
            if union > 0 and intersection / union >= 0.5:
                duplicate = True
                break

        if not duplicate:
            unique_candidates.append(candidate)

    return unique_candidates

def classify_zoom_size(area, frame_area):
    area_ratio = area / frame_area if frame_area else 0.0
    if area_ratio <= 0.01:
        return "small"
    if area_ratio <= 0.05:
        return "medium"
    return "large"

# 7. INITIALIZATION

print("\n======================================")
print("[1] LOADING REFERENCE")
print("======================================")

reference_image = cv2.imread(
    str(REFERENCE_IMAGE_PATH)
)

if reference_image is None:
    print(
        "[ERROR] Reference image not found:",
        REFERENCE_IMAGE_PATH,
    )
    exit()

print("[OK] Reference image loaded.")
print(
    "[INFO] Size:",
    reference_image.shape,
)

print("\n[2] Creating reference mask...")

reference_mask = create_mask(
    reference_image
)

reference_objects = find_quadrilaterals(
    reference_mask
)

if not reference_objects:
    print(
        "[ERROR] No target found "
        "in the reference image."
    )
    exit()

reference_object = max(
    reference_objects,
    key=lambda x: x["area"],
)

reference_points = order_points(
    reference_object["points"]
)

REFERENCE_ASPECT_RATIO = calculate_aspect_ratio(
    reference_points
)

print(
    "[OK] Reference aspect ratio:",
    round(REFERENCE_ASPECT_RATIO, 3),
)

top_left, top_right, bottom_right, bottom_left = reference_points

reference_width = (
    distance(top_left, top_right)
    +
    distance(bottom_left, bottom_right)
) / 2

reference_height = (
    distance(top_left, bottom_left)
    +
    distance(top_right, bottom_right)
) / 2

if reference_width > 0:
    NORMALIZED_HEIGHT = int(
        NORMALIZED_WIDTH
        *
        reference_height
        /
        reference_width
    )
else:
    NORMALIZED_HEIGHT = NORMALIZED_HEIGHT_FALLBACK

NORMALIZED_HEIGHT = max(
    MIN_NORMALIZED_HEIGHT,
    min(
        NORMALIZED_HEIGHT,
        MAX_NORMALIZED_HEIGHT,
    ),
)

print(
    "[INFO] Normalized size:",
    NORMALIZED_WIDTH,
    "x",
    NORMALIZED_HEIGHT,
)

normalized_reference = correct_perspective(
    reference_image,
    reference_points,
    NORMALIZED_WIDTH,
    NORMALIZED_HEIGHT,
)

processed_reference = prepare_image(
    normalized_reference
)

REFERENCE_INDEPENDENT_CONTOUR = extract_independent_contour(
    reference_image,
    reference_points,
)

REFERENCE_RECTANGULARITY = calculate_rectangularity(
    REFERENCE_INDEPENDENT_CONTOUR
)

REFERENCE_RGB = calculate_normalized_rgb(
    normalized_reference
)

REFERENCE_TEXTURE = calculate_texture_descriptor(
    normalized_reference
)

print("[OK] Reference normalized.")

print(
    "[ROBUSTNESS] Reference rectangularity:",
    "N/A"
    if REFERENCE_RECTANGULARITY is None
    else f"{REFERENCE_RECTANGULARITY:.3f}",
)

print(
    "[ROBUSTNESS] Reference normalized RGB:",
    "N/A"
    if REFERENCE_RGB is None
    else np.round(REFERENCE_RGB, 3),
)

print(
    "[ROBUSTNESS] Reference texture:",
    "N/A"
    if REFERENCE_TEXTURE is None
    else np.round(REFERENCE_TEXTURE, 3),
)

cv2.imshow(
    REFERENCE_WINDOW,
    reference_image,
)

cv2.imshow(
    REFERENCE_MASK_WINDOW,
    reference_mask,
)

cv2.imshow(
    NORMALIZED_REFERENCE_WINDOW,
    normalized_reference,
)

print("\n[3] Opening camera...")

if sys.platform.startswith("win"):
    video = cv2.VideoCapture(
        CAMERA,
        WINDOWS_CAMERA_BACKEND,
    )
else:
    video = cv2.VideoCapture(CAMERA)

if not video.isOpened():
    print(
        "[ERROR] Could not open camera."
    )
    exit()

print("[OK] Camera opened.")
print("[4] Starting detection...\n")

tracker = PredictiveKalmanTracker(
    measurement_sigma=MEASUREMENT_SIGMA,
    min_acceleration_sigma=MIN_ACCELERATION_SIGMA,
    max_acceleration_sigma=MAX_ACCELERATION_SIGMA,
    max_missed_frames=MAX_MISSED_FRAMES,
    nis_gate=NIS_GATE,
    confirmations_to_start=CONFIRMATIONS_TO_TRACK,
    max_confirmation_distance=MAX_CONFIRMATION_DISTANCE_PX,
)

v_controller = AdaptiveVController()
############################################
uart_controller = UARTSerialController()
############################################################
previous_time = None
frame_number = 0
processed_frame_count = 0
capture_worker = LatestFrameCapture(video)
capture_started = False
camera_released = False
arduino_state = "SEARCHING"
last_uart_state = None
last_uart_connected = False
verification_started_at = None
verification_samples = ()
verification_cooldown_until = None
verification_candidate_cleared = True
last_searching_sent_at = None
zoom_csv_file = None
zoom_csv_writer = None
zoom_csv_path = None
zoom_last_target_size = None
zoom_summary = {
    "frames": 0,
    "original_recognized": 0,
    "benchmark_3x_recognized": 0,
    "benchmark_4x_recognized": 0,
    "benchmark_3x_frames": 0,
    "benchmark_4x_frames": 0,
    "exclusive_3x": 0,
    "exclusive_4x": 0,
    "kalman_rejected_zoom_candidates": 0,
    "real_loop_seconds": 0.0,
    "original_seconds": 0.0,
    "operational_3x_seconds": 0.0,
    "operational_4x_seconds": 0.0,
    "benchmark_3x_seconds": 0.0,
    "benchmark_4x_seconds": 0.0,
    "operational_total_seconds": 0.0,
    "benchmark_total_seconds": 0.0,
}

# 8. MAIN LOOP

try:
    use_arduino = input(
        "Deseja utilizar a comunicação serial com o Arduino? [s/N]: "
    ).strip().lower() in ("s", "sim", "y", "yes")

    if use_arduino:
        if serial is None:
            raise RuntimeError(
                "PySerial nao esta instalado neste Python."
            )

        if not uart_controller.open():
            raise RuntimeError(
                f"Nao foi possivel conectar ao Arduino em "
                f"{uart_controller.port}. Verifique a conexao."
            )

        print("[UART] Arduino conectado.")
        print(f"[UART] Porta: {uart_controller.port}")
        print(f"[UART] Baudrate: {uart_controller.baudrate}")

        time.sleep(2.5)

        confirmation = input(
            "[Arduino] Plataforma alinhada? Definir zero logico? [s/N]: "
        ).strip().lower()

        if confirmation not in ("s", "sim", "y", "yes"):
            raise RuntimeError(
                "Inicializacao cancelada: zero nao autorizado."
            )

        if not uart_controller.send("Z"):
            raise RuntimeError(
                "Falha ao enviar Z para o Arduino."
            )

        print("[Arduino] Comando Z enviado.")
    else:
        print("[UART] Comunicacao serial desativada.")

    if SAVE_ZOOM_CSV:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_directory = Path(__file__).resolve().parent
        suffix = 1
        while zoom_csv_file is None:
            suffix_text = "" if suffix == 1 else f"_{suffix - 1}"
            candidate_path = (
                csv_directory
                / f"zoom_test_results_{timestamp}{suffix_text}.csv"
            )
            try:
                zoom_csv_file = candidate_path.open(
                    "x",
                    encoding="utf-8",
                    newline="",
                )
            except FileExistsError:
                suffix += 1
            else:
                zoom_csv_path = candidate_path
        zoom_csv_fieldnames = [
            "frame",
            "timestamp",
            "original_result",
            "original_score",
            "original_template_score",
            "original_processing_ms",
            "zoom_3x_result",
            "zoom_3x_score",
            "zoom_3x_processing_ms",
            "zoom_4x_result",
            "zoom_4x_score",
            "zoom_4x_processing_ms",
            "chosen_method",
            "consecutive_missed_frames",
            "object_area_px2",
            "tracking_state",
            "operational_total_ms",
            "operational_fps",
            "frame_processing_ms",
            "frame_processing_fps",
            "real_loop_ms",
            "real_loop_fps",
            "false_positive_confirmed",
            "kalman_rejected_zoom_candidate",
            "zoom_roi_source",
        ]
        if ENABLE_ZOOM_BENCHMARK:
            zoom_csv_fieldnames[18:18] = [
                "benchmark_3x_result",
                "benchmark_3x_score",
                "benchmark_3x_processing_ms",
                "benchmark_4x_result",
                "benchmark_4x_score",
                "benchmark_4x_processing_ms",
                "benchmark_extra_ms",
                "benchmark_total_ms",
                "benchmark_fps",
            ]
            zoom_csv_fieldnames[31:31] = [
                "benchmark_comparison",
                "method_disagreement",
            ]
        zoom_csv_writer = csv.DictWriter(
            zoom_csv_file,
            fieldnames=zoom_csv_fieldnames,
        )
        zoom_csv_writer.writeheader()
        print(f"[ZOOM TEST] CSV: {zoom_csv_path}")

    capture_worker.start()
    capture_started = True
    diagnostics_started = time.perf_counter()
    last_diagnostics_time = diagnostics_started
    last_capture_count, _ = capture_worker.get_statistics()
    last_processed_count = processed_frame_count
    frame_latency_ms = 0.0

    while True:
        loop_started = time.perf_counter()

        if not capture_worker.frame_available.wait(timeout=0.5):
            if capture_worker.finished_event.is_set():
                if capture_worker.capture_error is not None:
                    print(
                        "[ERROR] Camera capture failed: "
                        f"{capture_worker.capture_error}"
                    )
                elif capture_worker.capture_failed:
                    print("[ERROR] Failed to capture frame.")
                break
            continue

        latest_frame = capture_worker.get_latest()
        if latest_frame is None:
            if capture_worker.finished_event.is_set():
                if capture_worker.capture_error is not None:
                    print(
                        "[ERROR] Camera capture failed: "
                        f"{capture_worker.capture_error}"
                    )
                elif capture_worker.capture_failed:
                    print("[ERROR] Failed to capture frame.")
                break
            continue

        frame, frame_capture_time, _ = latest_frame
        frame_processing_started = time.perf_counter()
        frame_latency_ms = max(
            0.0,
            (frame_processing_started - frame_capture_time) * 1000.0,
        )
        frame_timestamp = (
            time.strftime("%Y-%m-%dT%H:%M:%S")
            + f".{int((time.time() % 1) * 1000):03d}"
        )
        frame_number += 1
        display_image = frame.copy()

        video_height, video_width = frame.shape[:2]

        if previous_time is None:
            delta_time = MIN_DT
        else:
            delta_time = max(0.0, frame_capture_time - previous_time)
        previous_time = frame_capture_time

        prediction = tracker.predict(delta_time)

        camera_center_x = video_width // 2
        camera_center_y = video_height // 2

        original_processing_started = time.perf_counter()

        hsv_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV,
        )

        if USE_ADAPTIVE_HSV:
            v_info = v_controller.update(
                image=frame,
                hsv=hsv_frame,
                frame_number=frame_number,
                prediction=prediction,
                processed_reference=processed_reference,
                reference_aspect_ratio=REFERENCE_ASPECT_RATIO,
                normalized_width=NORMALIZED_WIDTH,
                normalized_height=NORMALIZED_HEIGHT,
            )
            mask = v_info["mask"]
            current_v_min = v_info["vmin"]
        else:
            current_v_min = V_MIN
            mask = create_hsv_mask(
                hsv_frame,
                current_v_min,
            )

        search_mask = mask
        tracking_roi = None

        if prediction is not None and tracker.initialized:
            roi_center_x = int(round(prediction["center_x"]))
            roi_center_y = int(round(prediction["center_y"]))

            x1 = max(0, roi_center_x - TRACKING_ROI_RADIUS)
            y1 = max(0, roi_center_y - TRACKING_ROI_RADIUS)
            x2 = min(video_width, roi_center_x + TRACKING_ROI_RADIUS)
            y2 = min(video_height, roi_center_y + TRACKING_ROI_RADIUS)

            if x2 > x1 and y2 > y1:
                tracking_roi = (x1, y1, x2, y2)
                roi_mask = np.zeros_like(mask)
                roi_mask[y1:y2, x1:x2] = 255
                search_mask = cv2.bitwise_and(mask, roi_mask)

        objects = find_quadrilaterals(
            search_mask
        )

        valid_candidates = []

        for index, object in enumerate(objects):

            points = order_points(
                object["points"]
            )

            current_aspect_ratio = calculate_aspect_ratio(
                points
            )

            if REFERENCE_ASPECT_RATIO > 0:
                shape_error = abs(
                    current_aspect_ratio
                    -
                    REFERENCE_ASPECT_RATIO
                ) / REFERENCE_ASPECT_RATIO
            else:
                shape_error = 999

            valid_shape = (
                shape_error
                <=
                SHAPE_TOLERANCE
            )

            if valid_shape:
                aspect_ratio_score = float(
                    np.clip(
                        1.0
                        -
                        shape_error / SHAPE_TOLERANCE,
                        0.0,
                        1.0,
                    )
                )
            else:
                aspect_ratio_score = 0.0

            if valid_shape:

                (
                    visual_score,
                    normalized_image,
                    rotation,
                ) = best_normalization(
                    frame,
                    points,
                    processed_reference,
                    NORMALIZED_WIDTH,
                    NORMALIZED_HEIGHT,
                )

                independent_contour = extract_independent_contour(
                    frame,
                    points,
                )

                current_rectangularity = calculate_rectangularity(
                    independent_contour
                )
                
                nota_retangularidade = rectangularity_score(
                    REFERENCE_RECTANGULARITY,
                    current_rectangularity,
                )

                (
                    match_shapes_error,
                    match_shapes_score,
                ) = compare_match_shapes(
                    REFERENCE_INDEPENDENT_CONTOUR,
                    independent_contour,
                )

                current_rgb = calculate_normalized_rgb(
                    normalized_image
                )

                rgb_score = normalized_rgb_score(
                    REFERENCE_RGB,
                    current_rgb,
                )

                current_texture = calculate_texture_descriptor(
                    normalized_image
                )

                texture_score = calculate_texture_score(
                    REFERENCE_TEXTURE,
                    current_texture,
                )

                robustness_score = combine_robustness_scores({
                    "template": visual_score,
                    "aspect_ratio": aspect_ratio_score,
                    "rectangularity": nota_retangularidade,
                    "shape": match_shapes_score,
                    "rgb": rgb_score,
                    "texture": texture_score,
                })

            else:
                visual_score = 0.0
                normalized_image = None
                rotation = 0
                independent_contour = None
                current_rectangularity = None
                nota_retangularidade = None
                match_shapes_error = None
                match_shapes_score = None
                current_rgb = None
                rgb_score = None
                current_texture = None
                texture_score = None
                robustness_score = 0.0

            is_valid = (
                valid_shape
                and
                robustness_score >= ROBUSTNESS_SCORE_THRESHOLD
            )

            center_x = int(
                np.mean(
                    points[:, 0]
                )
            )

            center_y = int(
                np.mean(
                    points[:, 1]
                )
            )

            valid_temporal = True

            if is_valid and prediction is not None:

                (
                    temporal_score,
                    candidate_nis,
                ) = tracker.temporal_score(
                    center_x,
                    center_y,
                )

                valid_temporal = (
                    np.isfinite(candidate_nis)
                    and
                    candidate_nis <= tracker.nis_gate
                )

                if valid_temporal:
                    total_score = (
                        FINAL_ROBUSTNESS_WEIGHT * robustness_score
                        +
                        TEMPORAL_WEIGHT * temporal_score
                    )
                else:
                    total_score = 0.0

            else:
                temporal_score = 1.0
                candidate_nis = 0.0
                total_score = robustness_score if is_valid else 0.0

            is_final_valid = (
                is_valid
                and
                valid_temporal
            )

            if is_final_valid:

                color = GREEN
                status = "VALIDO"

                valid_candidates.append({
                    "score": robustness_score,
                    "robustness_score": robustness_score,
                    "visual_score": visual_score,
                    "aspect_ratio_score": aspect_ratio_score,
                    "rectangularity_score": nota_retangularidade,
                    "match_shapes_score": match_shapes_score,
                    "match_shapes_error": match_shapes_error,
                    "rgb_score": rgb_score,
                    "texture_score": texture_score,
                    "temporal_score": temporal_score,
                    "total_score": total_score,
                    "candidate_nis": candidate_nis,
                    "center": (center_x, center_y),
                    "points": points,
                    "normalized": normalized_image,
                    "rotation": rotation,
                    "shape_error": shape_error,
                })

            elif is_valid and prediction is not None and not valid_temporal:

                color = ORANGE
                status = "REJEITADO NIS"

            else:
                color = RED
                status = "REJEITADO"

            ptsDraw = np.int32(points)

            cv2.polylines(
                display_image,
                [ptsDraw],
                True,
                color,
                DRAWING_THICKNESS_3,
            )

            for point in ptsDraw:
                cv2.circle(
                    display_image,
                    tuple(point),
                    MARKER_RADIUS_7,
                    MAGENTA,
                    MARKER_FILL,
                )

            cv2.circle(
                display_image,
                (center_x, center_y),
                MARKER_RADIUS_7,
                color,
                MARKER_FILL,
            )

            text_x = center_x - CANDIDATE_TEXT_X_OFFSET
            text_y = center_y - CANDIDATE_TEXT_Y_OFFSET

            cv2.putText(
                display_image,
                status,
                (text_x, text_y),
                TEXT_FONT,
                TEXT_SCALE_0_55,
                color,
                TEXT_THICKNESS_2,
            )


            def fmt_score(value):
                if value is None or not np.isfinite(value):
                    return "N/A"
                return f"{value * 100:.0f}%"

            candidate_lines = [
                f"Prop:{aspect_ratio_score * 100:.0f}% Tmpl:{visual_score * 100:.0f}%",
                f"Rect:{fmt_score(nota_retangularidade)} Shape:{fmt_score(match_shapes_score)}",
                f"RGB:{fmt_score(rgb_score)} Tex:{fmt_score(texture_score)}",
                f"Robustness:{robustness_score * 100:.1f}%",
            ]

            if is_valid and prediction is not None:
                candidate_lines.append(
                    f"Temp:{temporal_score * 100:.0f}% Total:{total_score * 100:.1f}%"
                )
            else:
                candidate_lines.append(
                    f"Rot:{rotation} graus"
                )

            for candidate_line_index, candidate_line in enumerate(candidate_lines):
                cv2.putText(
                    display_image,
                    candidate_line,
                    (text_x, text_y + CANDIDATE_TEXT_SPACING * (candidate_line_index + 1)),
                    TEXT_FONT,
                    TEXT_SCALE_0_43,
                    color,
                    TEXT_THICKNESS_1,
                )

            if frame_number % LOG_EVERY_N_FRAMES == 0:
                print(
                    f"Candidate {index} | "
                    f"Aspect ratio={aspect_ratio_score:.3f} | "
                    f"Template={visual_score:.3f} | "
                    f"Rect={nota_retangularidade if nota_retangularidade is not None else -1:.3f} | "
                    f"Shape={match_shapes_score if match_shapes_score is not None else -1:.3f} | "
                    f"RGB={rgb_score if rgb_score is not None else -1:.3f} | "
                    f"Texture={texture_score if texture_score is not None else -1:.3f} | "
                    f"Robustness={robustness_score:.3f} | "
                    f"Temporal={temporal_score:.3f} | "
                    f"NIScand={candidate_nis:.2f} | "
                    f"Total={total_score:.3f} | "
                    f"Rotation={rotation} | "
                    f"{status}"
                )

        original_processing_seconds = (
            time.perf_counter() - original_processing_started
        )
        original_best = (
            max(valid_candidates, key=lambda item: item["total_score"])
            if valid_candidates else None
        )
        original_recognized = original_best is not None
        zoom_roi_objects = objects
        if (
            not zoom_roi_objects
            and (ENABLE_COMPLEMENTARY_ZOOM or ENABLE_ZOOM_BENCHMARK)
        ):
            zoom_roi_objects = find_quadrilaterals(
                search_mask,
                min_area=ZOOM_MIN_AREA_PX2,
            )
        zoom_roi_bounds, zoom_roi_source = get_zoom_roi_bounds(
            zoom_roi_objects,
            frame.shape,
            prediction,
            tracker,
            zoom_last_target_size,
        )
        operational_zoom_results = {
            factor: None for factor in ZOOM_FACTORS
        }
        operational_zoom_seconds = {
            factor: None for factor in ZOOM_FACTORS
        }
        benchmark_zoom_results = {
            factor: None for factor in ZOOM_FACTORS
        }
        benchmark_zoom_seconds = {
            factor: None for factor in ZOOM_FACTORS
        }
        chosen_method = "ORIGINAL" if original_recognized else "SEARCHING"
        tracking_candidates = valid_candidates

        if (
            not original_recognized
            and ENABLE_COMPLEMENTARY_ZOOM
            and zoom_roi_bounds
        ):
            for factor in ZOOM_FACTORS:
                started = time.perf_counter()
                candidate = evaluate_zoom_rois(
                    frame,
                    zoom_roi_bounds,
                    current_v_min,
                    prediction,
                    tracker,
                    factor,
                )
                operational_zoom_seconds[factor] = (
                    time.perf_counter() - started
                )
                operational_zoom_results[factor] = candidate
                if candidate is not None:
                    tracking_candidates = [candidate]
                    chosen_method = f"ZOOM_{factor}X"
                    _, _, target_width, target_height = cv2.boundingRect(
                        np.int32(candidate["points"])
                    )
                    zoom_last_target_size = (
                        max(1, target_width),
                        max(1, target_height),
                    )
                    break

        if original_best is not None:
            _, _, target_width, target_height = cv2.boundingRect(
                np.int32(original_best["points"])
            )
            zoom_last_target_size = (
                max(1, target_width),
                max(1, target_height),
            )

        if ENABLE_ZOOM_BENCHMARK and zoom_roi_bounds:
            for factor in ZOOM_FACTORS:
                if operational_zoom_seconds[factor] is not None:
                    benchmark_zoom_results[factor] = (
                        operational_zoom_results[factor]
                    )
                    benchmark_zoom_seconds[factor] = (
                        operational_zoom_seconds[factor]
                    )
                    continue
                started = time.perf_counter()
                benchmark_zoom_results[factor] = evaluate_zoom_rois(
                    frame,
                    zoom_roi_bounds,
                    current_v_min,
                    prediction,
                    tracker,
                    factor,
                )
                benchmark_zoom_seconds[factor] = (
                    time.perf_counter() - started
                )

        benchmark_3x = benchmark_zoom_results.get(3)
        benchmark_4x = benchmark_zoom_results.get(4)
        comparison = (
            "original={} zoom_3x={} zoom_4x={}".format(
                int(original_recognized),
                int(benchmark_3x is not None),
                int(benchmark_4x is not None),
            )
            if ENABLE_ZOOM_BENCHMARK else "NOT_EXECUTED"
        )
        object_area_px2 = None
        measurement_accepted = None
        candidate_center = None
        if tracking_candidates:

            # Use the visual score before initialization and the combined score afterward.
            best = max(
                tracking_candidates,
                key=lambda c: c["total_score"],
            )

            measured_x, measured_y = (
                int(round(best["center"][0])),
                int(round(best["center"][1])),
            )
            candidate_center = (measured_x, measured_y)
            if "area" in best:
                object_area_px2 = best["area"]
            else:
                _, _, object_width, object_height = cv2.boundingRect(
                    np.int32(best["points"])
                )
                object_area_px2 = object_width * object_height

            score = best["score"]
            target_robustness_score = best["robustness_score"]
            target_visual_score = best["visual_score"]
            target_rectangularity_score = best["rectangularity_score"]
            target_shape_score = best["match_shapes_score"]
            target_rgb_score = best["rgb_score"]
            target_texture_score = best["texture_score"]
            temporal_score = best["temporal_score"]
            total_score = best["total_score"]
            normalized_image = best["normalized"]
            rotation = best["rotation"]

            kalman_state = tracker.correct(
                measured_x,
                measured_y,
            )

            kalman_vx = kalman_state["vx"]
            kalman_vy = kalman_state["vy"]
            nis = kalman_state["nis"]
            acceleration_sigma = kalman_state["acceleration_sigma"]
            measurement_accepted = kalman_state["measurement_accepted"]
            tracking_active = kalman_state["tracking_active"]
            confirming_target = bool(
                kalman_state.get("confirming", False)
            )
            confirmation_count = int(
                kalman_state.get("confirmations", 0)
            )

            # correct() never returns a rejected measurement.
            # When tracking has ended, do not generate an error
            # or a motor command.
            kalman_x = int(
                round(kalman_state["center_x"])
            )

            kalman_y = int(
                round(kalman_state["center_y"])
            )

            if tracking_active:

                error_x = kalman_x - camera_center_x
                error_y = kalman_y - camera_center_y

                movement_x = "CENTERED"
                movement_y = "CENTERED"

                if error_x < -dead_zone:
                    movement_x = "MOVE LEFT"

                elif error_x > dead_zone:
                    movement_x = "MOVE RIGHT"

                if error_y < -dead_zone:
                    movement_y = "MOVE UP"

                elif error_y > dead_zone:
                    movement_y = "MOVE DOWN"

            else:

                # Tracking has ended: do not produce a control error
                # or a directional command.
                error_x = 0
                error_y = 0
                movement_x = "SEARCHING"
                movement_y = "SEARCHING"

            cv2.circle(
                display_image,
                (measured_x, measured_y),
                MARKER_RADIUS_6,
                GREEN,
                MARKER_FILL,
            )

            cv2.circle(
                display_image,
                (kalman_x, kalman_y),
                MARKER_RADIUS_8,
                MAGENTA,
                DRAWING_THICKNESS_2,
            )

            if prediction is not None:

                predicted_x = int(round(prediction["center_x"]))
                predicted_y = int(round(prediction["center_y"]))

                cv2.circle(
                    display_image,
                    (predicted_x, predicted_y),
                    MARKER_RADIUS_9,
                    YELLOW,
                    DRAWING_THICKNESS_2,
                )

            cv2.line(
                display_image,
                (camera_center_x, camera_center_y),
                (kalman_x, kalman_y),
                RED,
                DRAWING_THICKNESS_2,
            )

            cv2.putText(
                display_image,
                f"TARGET robustness {target_robustness_score * 100:.1f}%",
                TEXT_POSITION_35,
                TEXT_FONT,
                TEXT_SCALE_0_7,
                YELLOW,
                TEXT_THICKNESS_2,
            )

            cv2.putText(
                display_image,
                f"Template {target_visual_score * 100:.1f}%  Temporal {temporal_score * 100:.1f}%  Total {total_score * 100:.1f}%",
                TEXT_POSITION_65,
                TEXT_FONT,
                TEXT_SCALE_0_55,
                YELLOW,
                TEXT_THICKNESS_2,
            )

            cv2.putText(
                display_image,
                (
                    f"Rect {target_rectangularity_score * 100:.0f}% "
                    if target_rectangularity_score is not None else "Rect N/A "
                )
                +
                (
                    f"Shape {target_shape_score * 100:.0f}% "
                    if target_shape_score is not None else "Shape N/A "
                )
                +
                (
                    f"RGB {target_rgb_score * 100:.0f}% "
                    if target_rgb_score is not None else "RGB N/A "
                )
                +
                (
                    f"Tex {target_texture_score * 100:.0f}%"
                    if target_texture_score is not None else "Tex N/A"
                ),
                TEXT_POSITION_95,
                TEXT_FONT,
                TEXT_SCALE_0_48,
                YELLOW,
                TEXT_THICKNESS_1,
            )

            cv2.putText(
                display_image,
                f"Kalman: ({kalman_x}, {kalman_y})",
                TEXT_POSITION_125,
                TEXT_FONT,
                TEXT_SCALE_0_55,
                MAGENTA,
                TEXT_THICKNESS_2,
            )

            cv2.putText(
                display_image,
                f"Vx: {kalman_vx:.1f} px/s  Vy: {kalman_vy:.1f} px/s",
                TEXT_POSITION_155,
                TEXT_FONT,
                TEXT_SCALE_0_55,
                MAGENTA,
                TEXT_THICKNESS_2,
            )

            cv2.putText(
                display_image,
                f"Error X: {error_x}  Error Y: {error_y}",
                TEXT_POSITION_185,
                TEXT_FONT,
                TEXT_SCALE_0_55,
                GREEN,
                TEXT_THICKNESS_2,
            )

            cv2.putText(
                display_image,
                f"NIS: {nis:.2f}  acceleration_sigma: {acceleration_sigma:.1f}",
                TEXT_POSITION_215,
                TEXT_FONT,
                TEXT_SCALE_0_55,
                CYAN,
                TEXT_THICKNESS_2,
            )

            if confirming_target:
                measurement_status = (
                    f"VERIFYING TARGET "
                    f"{confirmation_count}/{CONFIRMATIONS_TO_TRACK}"
                )
            elif not tracking_active:
                measurement_status = (
                    "VERIFYING TARGET"
                    if verification_started_at is not None
                    else "TRACKING ENDED - SEARCHING"
                )
            elif measurement_accepted:
                measurement_status = "MEASUREMENT ACCEPTED"
            else:
                measurement_status = "MEASUREMENT REJECTED BY NIS"

            cv2.putText(
                display_image,
                measurement_status,
                TEXT_POSITION_245,
                TEXT_FONT,
                TEXT_SCALE_0_5,
                GREEN if measurement_accepted else RED,
                TEXT_THICKNESS_2,
            )

            cv2.putText(
                display_image,
                movement_x,
                TEXT_POSITION_305,
                TEXT_FONT,
                TEXT_SCALE_0_55,
                BLUE,
                TEXT_THICKNESS_2,
            )

            cv2.putText(
                display_image,
                movement_y,
                TEXT_POSITION_275,
                TEXT_FONT,
                TEXT_SCALE_0_55,
                BLUE,
                TEXT_THICKNESS_2,
            )

            if normalized_image is not None:
                cv2.imshow(
                    NORMALIZED_CANDIDATE_WINDOW,
                    normalized_image,
                )

        else:

            tracking_active = False
            error_x = 0
            error_y = 0
            frames_lost = tracker.missed_frames

            prediction_valid = (
                prediction is not None
                and all(
                    np.isfinite(prediction[key])
                    for key in ("center_x", "center_y", "vx", "vy")
                )
            )

            if prediction_valid and tracker.initialized:
                still_tracking = (
                    tracker.accept_prediction_without_measurement()
                )

                if still_tracking:

                    tracking_active = True
                    frames_lost = tracker.missed_frames
                    kalman_x = int(round(prediction["center_x"]))
                    kalman_y = int(round(prediction["center_y"]))

                    kalman_vx = prediction["vx"]
                    kalman_vy = prediction["vy"]

                    error_x = kalman_x - camera_center_x
                    error_y = kalman_y - camera_center_y

                    cv2.circle(
                        display_image,
                        (kalman_x, kalman_y),
                        MARKER_RADIUS_10,
                        YELLOW,
                        DRAWING_THICKNESS_2,
                    )

                    cv2.line(
                        display_image,
                        (camera_center_x, camera_center_y),
                        (kalman_x, kalman_y),
                        YELLOW,
                        DRAWING_THICKNESS_2,
                    )

                    cv2.putText(
                        display_image,
                        "TARGET NOT DETECTED - KALMAN PREDICTION",
                        TEXT_POSITION_35,
                        TEXT_FONT,
                        TEXT_SCALE_0_6,
                        YELLOW,
                        TEXT_THICKNESS_2,
                    )

                    cv2.putText(
                        display_image,
                        f"Prediction: ({kalman_x}, {kalman_y})",
                        TEXT_POSITION_65,
                        TEXT_FONT,
                        TEXT_SCALE_0_6,
                        YELLOW,
                        TEXT_THICKNESS_2,
                    )

                    cv2.putText(
                        display_image,
                        f"Vx: {kalman_vx:.1f} px/s  Vy: {kalman_vy:.1f} px/s",
                        TEXT_POSITION_95,
                        TEXT_FONT,
                        TEXT_SCALE_0_55,
                        YELLOW,
                        TEXT_THICKNESS_2,
                    )

                    cv2.putText(
                        display_image,
                        (
                            f"Frames missed: "
                            f"{tracker.missed_frames}/"
                            f"{tracker.max_missed_frames}"
                        ),
                        TEXT_POSITION_125,
                        TEXT_FONT,
                        TEXT_SCALE_0_55,
                        YELLOW,
                        TEXT_THICKNESS_2,
                    )

                    cv2.putText(
                        display_image,
                        f"Error X: {error_x}  Error Y: {error_y}",
                        TEXT_POSITION_155,
                        TEXT_FONT,
                        TEXT_SCALE_0_55,
                        YELLOW,
                        TEXT_THICKNESS_2,
                    )

                else:
                    frames_lost = tracker.max_missed_frames + 1

                    cv2.putText(
                        display_image,
                        (
                            "VERIFYING TARGET"
                            if verification_started_at is not None
                            else "SEARCHING - TARGET LOST"
                        ),
                        TEXT_POSITION_35,
                        TEXT_FONT,
                        TEXT_SCALE_0_7,
                        RED,
                        TEXT_THICKNESS_2,
                    )

            else:

                tracker.cancel_initial_confirmation()
                frames_lost += 1

                cv2.putText(
                    display_image,
                    (
                        "VERIFYING TARGET"
                        if verification_started_at is not None
                        else "SEARCHING"
                    ),
                    TEXT_POSITION_35,
                    TEXT_FONT,
                    TEXT_SCALE_0_7,
                    RED,
                    TEXT_THICKNESS_2,
                )

        now = time.perf_counter()
        (
            arduino_state,
            verification_started_at,
            verification_samples,
            verification_cooldown_until,
            verification_candidate_cleared,
            verification_expired,
        ) = resolve_arduino_state(
            current_state=arduino_state,
            tracking_active=tracking_active,
            has_candidate=bool(tracking_candidates),
            candidate_center=candidate_center,
            now=now,
            verification_started_at=verification_started_at,
            verification_samples=verification_samples,
            verification_timeout=VERIFYING_TIMEOUT_SECONDS,
            cooldown_until=verification_cooldown_until,
            candidate_cleared_after_failure=verification_candidate_cleared,
        )
        if verification_expired:
            tracker.reset()

        uart_connected = uart_controller.is_connected
        if uart_connected and not last_uart_connected:
            last_uart_state = None
            last_searching_sent_at = None
        state_changed = arduino_state != last_uart_state
        should_send_state = should_send_uart_state(
            arduino_state,
            last_uart_state,
            last_searching_sent_at,
            now,
        )
        if uart_connected and should_send_state:
            if uart_controller.send_tracking_state(
                state=arduino_state,
                error_x=error_x,
                error_y=error_y,
            ):
                last_uart_state = arduino_state
                sent_at = time.perf_counter()
                if arduino_state == "SEARCHING":
                    last_searching_sent_at = sent_at
                elif arduino_state == "VERIFYING" and state_changed:
                    verification_started_at = sent_at
                    verification_samples = ()
        last_uart_connected = uart_controller.is_connected

        print(
            f"[TRACKING] state={arduino_state} "
            f"frames_lost={tracker.missed_frames} "
            f"error_x={error_x} error_y={error_y}"
        )

        tracking_state = arduino_state
        if not tracking_candidates and tracking_active:
            chosen_method = "KALMAN"
        elif not tracking_candidates and arduino_state == "SEARCHING":
            chosen_method = "SEARCHING"

        if object_area_px2 is None:
            benchmark_candidate = (
                benchmark_zoom_results.get(3)
                or benchmark_zoom_results.get(4)
            )
            if benchmark_candidate is not None:
                object_area_px2 = benchmark_candidate["area"]
            elif objects:
                largest_object = max(
                    objects,
                    key=lambda detected: cv2.contourArea(
                        detected["contour"]
                    ),
                )
                _, _, object_width, object_height = cv2.boundingRect(
                    largest_object["contour"]
                )
                object_area_px2 = object_width * object_height

        operational_zoom_total = sum(
            seconds
            for seconds in operational_zoom_seconds.values()
            if seconds is not None
        )
        operational_total_seconds = (
            original_processing_seconds + operational_zoom_total
        )
        benchmark_extra_seconds = sum(
            seconds
            for factor, seconds in benchmark_zoom_seconds.items()
            if (
                seconds is not None
                and operational_zoom_seconds[factor] is None
            )
        )
        benchmark_total_seconds = (
            operational_total_seconds + benchmark_extra_seconds
        )
        frame_processing_seconds = (
            time.perf_counter() - frame_processing_started
        )
        benchmark_executed = any(
            seconds is not None
            for seconds in benchmark_zoom_seconds.values()
        )
        benchmark_comparison = (
            comparison if benchmark_executed else "NOT_EXECUTED"
        )
        method_disagreements = []
        if benchmark_executed:
            for factor in ZOOM_FACTORS:
                if benchmark_zoom_seconds[factor] is None:
                    continue
                zoom_recognized = (
                    benchmark_zoom_results[factor] is not None
                )
                if zoom_recognized != original_recognized:
                    method_disagreements.append(
                        f"ORIGINAL_vs_ZOOM_{factor}X"
                    )
            if (
                benchmark_zoom_seconds.get(3) is not None
                and benchmark_zoom_seconds.get(4) is not None
                and (benchmark_zoom_results.get(3) is not None)
                != (benchmark_zoom_results.get(4) is not None)
            ):
                method_disagreements.append("ZOOM_3X_vs_ZOOM_4X")
        method_disagreement = (
            ";".join(method_disagreements)
            if method_disagreements
            else "NO" if benchmark_executed
            else "NOT_EXECUTED"
        )
        kalman_rejected_zoom_candidate = (
            chosen_method.startswith("ZOOM_")
            and measurement_accepted is False
        )
        false_positive_confirmed = "NOT_VERIFIABLE"

        zoom_summary["frames"] += 1
        zoom_summary["original_recognized"] += int(original_recognized)
        zoom_summary["original_seconds"] += original_processing_seconds
        zoom_summary["operational_total_seconds"] += (
            operational_total_seconds
        )
        zoom_summary["benchmark_total_seconds"] += benchmark_total_seconds
        zoom_summary["kalman_rejected_zoom_candidates"] += int(
            kalman_rejected_zoom_candidate
        )

        if benchmark_executed:
            zoom_summary["benchmark_3x_frames"] += int(
                benchmark_zoom_seconds.get(3) is not None
            )
            zoom_summary["benchmark_4x_frames"] += int(
                benchmark_zoom_seconds.get(4) is not None
            )
            zoom_summary["benchmark_3x_recognized"] += int(
                benchmark_zoom_results.get(3) is not None
            )
            zoom_summary["benchmark_4x_recognized"] += int(
                benchmark_zoom_results.get(4) is not None
            )
            zoom_summary["benchmark_3x_seconds"] += (
                benchmark_zoom_seconds.get(3) or 0.0
            )
            zoom_summary["benchmark_4x_seconds"] += (
                benchmark_zoom_seconds.get(4) or 0.0
            )
            zoom_summary["exclusive_3x"] += int(
                not original_recognized
                and benchmark_zoom_results.get(3) is not None
                and benchmark_zoom_results.get(4) is None
            )
            zoom_summary["exclusive_4x"] += int(
                not original_recognized
                and benchmark_zoom_results.get(4) is not None
                and benchmark_zoom_results.get(3) is None
            )

        zoom_summary["operational_3x_seconds"] += (
            operational_zoom_seconds.get(3) or 0.0
        )
        zoom_summary["operational_4x_seconds"] += (
            operational_zoom_seconds.get(4) or 0.0
        )

        zoom_csv_row = None
        if zoom_csv_writer is not None:
            original_score = (
                original_best["robustness_score"]
                if original_best is not None else ""
            )
            original_template_score = (
                original_best["visual_score"]
                if original_best is not None else ""
            )
            row = {
                "frame": frame_number,
                "timestamp": frame_timestamp,
                "original_result": (
                    "DETECTED" if original_recognized else "NOT_DETECTED"
                ),
                "original_score": original_score,
                "original_template_score": original_template_score,
                "original_processing_ms": (
                    original_processing_seconds * 1000.0
                ),
                "chosen_method": chosen_method,
                "consecutive_missed_frames": tracker.missed_frames,
                "object_area_px2": (
                    object_area_px2 if object_area_px2 is not None else ""
                ),
                "tracking_state": tracking_state,
                "operational_total_ms": (
                    operational_total_seconds * 1000.0
                ),
                "operational_fps": (
                    1.0 / operational_total_seconds
                    if operational_total_seconds > 0 else 0.0
                ),
                "frame_processing_ms": (
                    frame_processing_seconds * 1000.0
                ),
                "frame_processing_fps": (
                    1.0 / frame_processing_seconds
                    if frame_processing_seconds > 0 else 0.0
                ),
                "real_loop_ms": "",
                "real_loop_fps": "",
                "false_positive_confirmed": false_positive_confirmed,
                "kalman_rejected_zoom_candidate": (
                    "YES" if kalman_rejected_zoom_candidate
                    else "NO" if chosen_method.startswith("ZOOM_")
                    else "NOT_APPLICABLE"
                ),
                "zoom_roi_source": zoom_roi_source,
            }
            if ENABLE_ZOOM_BENCHMARK:
                row.update({
                    "benchmark_extra_ms": (
                        benchmark_extra_seconds * 1000.0
                    ),
                    "benchmark_total_ms": (
                        benchmark_total_seconds * 1000.0
                    ),
                    "benchmark_fps": (
                        1.0 / benchmark_total_seconds
                        if benchmark_total_seconds > 0 else 0.0
                    ),
                    "benchmark_comparison": benchmark_comparison,
                    "method_disagreement": method_disagreement,
                })
            for factor in ZOOM_FACTORS:
                operation_seconds = operational_zoom_seconds[factor]
                operation_candidate = operational_zoom_results[factor]
                benchmark_seconds = benchmark_zoom_seconds[factor]
                benchmark_candidate = benchmark_zoom_results[factor]
                prefix = f"zoom_{factor}x"
                row[f"{prefix}_result"] = (
                    "NOT_EXECUTED"
                    if operation_seconds is None
                    else "DETECTED"
                    if operation_candidate is not None
                    else "NOT_DETECTED"
                )
                row[f"{prefix}_score"] = (
                    operation_candidate["robustness_score"]
                    if operation_candidate is not None else ""
                )
                row[f"{prefix}_processing_ms"] = (
                    operation_seconds * 1000.0
                    if operation_seconds is not None else "NOT_EXECUTED"
                )
                if ENABLE_ZOOM_BENCHMARK:
                    benchmark_prefix = f"benchmark_{factor}x"
                    row[f"{benchmark_prefix}_result"] = (
                        "NOT_EXECUTED"
                        if benchmark_seconds is None
                        else "DETECTED"
                        if benchmark_candidate is not None
                        else "NOT_DETECTED"
                    )
                    row[f"{benchmark_prefix}_score"] = (
                        benchmark_candidate["robustness_score"]
                        if benchmark_candidate is not None else ""
                    )
                    row[f"{benchmark_prefix}_processing_ms"] = (
                        benchmark_seconds * 1000.0
                        if benchmark_seconds is not None else "NOT_EXECUTED"
                    )
            zoom_csv_row = row

        v_info_x = max(MIN_DEBUG_X_MARGIN, video_width - DEBUG_PANEL_WIDTH)
        v_info_y = DEBUG_Y_POSITION

        system_state = arduino_state

        v_lines = [
            f"STATE: {system_state}",
            (
                f"HSV ADAPTIVE | {v_info['mode']}"
                if USE_ADAPTIVE_HSV
                else "HSV FIXED | hsv.csv"
            ),
            f"H: {H_MIN}-{H_MAX}  S: {S_MIN}-{S_MAX}",
            f"V: {current_v_min}-{V_MAX}",
        ]

        for line_index, text_line in enumerate(v_lines):
            cv2.putText(
                display_image,
                text_line,
                (v_info_x, v_info_y + DEBUG_TEXT_SPACING * line_index),
                TEXT_FONT,
                TEXT_SCALE_0_48,
                WHITE,
                TEXT_THICKNESS_1,
            )

        if frame_number % LOG_EVERY_N_FRAMES == 0:
            print(
                f"[HSV {'ADAPT' if USE_ADAPTIVE_HSV else 'FIXED'}] "
                f"H={H_MIN}-{H_MAX} S={S_MIN}-{S_MAX} "
                f"V={current_v_min}-{V_MAX}"
            )

        if tracking_roi is not None:
            x1t, y1t, x2t, y2t = tracking_roi
            cv2.rectangle(
                display_image,
                (x1t, y1t),
                (x2t, y2t),
                MAGENTA,
                DRAWING_THICKNESS_1,
            )

        cv2.line(
            display_image,
            (camera_center_x, 0),
            (camera_center_x, video_height),
            GREEN,
            DRAWING_THICKNESS_2,
        )

        cv2.line(
            display_image,
            (0, camera_center_y),
            (video_width, camera_center_y),
            GREEN,
            DRAWING_THICKNESS_2,
        )

        cv2.rectangle(
            display_image,
            (
                camera_center_x - dead_zone,
                camera_center_y - dead_zone,
            ),
            (
                camera_center_x + dead_zone,
                camera_center_y + dead_zone,
            ),
            CYAN,
            DRAWING_THICKNESS_2,
        )

        cv2.imshow(
            TRACKER_WINDOW,
            display_image,
        )

        cv2.imshow(
            HSV_MASK_WINDOW,
            mask,
        )

        exit_requested = (
            cv2.waitKey(KEY_WAIT_MS) & 0xFF == ord(EXIT_KEY)
        )
        processed_frame_count += 1
        diagnostics_now = time.perf_counter()
        diagnostics_elapsed = diagnostics_now - last_diagnostics_time
        if diagnostics_elapsed >= CAPTURE_DIAGNOSTICS_INTERVAL_SECONDS:
            capture_count, dropped_count = capture_worker.get_statistics()
            capture_fps = (
                capture_count - last_capture_count
            ) / diagnostics_elapsed
            processing_fps = (
                processed_frame_count - last_processed_count
            ) / diagnostics_elapsed
            print(
                f"[PERF] capture={capture_fps:.1f} FPS "
                f"processing={processing_fps:.1f} FPS "
                f"latency={frame_latency_ms:.1f} ms "
                f"dropped={dropped_count}"
            )
            last_diagnostics_time = diagnostics_now
            last_capture_count = capture_count
            last_processed_count = processed_frame_count

        real_loop_seconds = time.perf_counter() - loop_started
        zoom_summary["real_loop_seconds"] += real_loop_seconds
        if zoom_csv_row is not None:
            zoom_csv_row["real_loop_ms"] = real_loop_seconds * 1000.0
            zoom_csv_row["real_loop_fps"] = (
                1.0 / real_loop_seconds
                if real_loop_seconds > 0 else 0.0
            )
            zoom_csv_writer.writerow(zoom_csv_row)

        if exit_requested:
            break

finally:
    if uart_controller.is_connected and not uart_controller.send("STOP"):
        print("[UART] STOP nao enviado: conexao com o Arduino perdida.")
    uart_controller.close()
    if capture_started:
        capture_worker.stop()
        capture_worker.thread.join(timeout=2.0)
        if capture_worker.thread.is_alive():
            print(
                "[WARN] Camera capture thread is still blocked; "
                "releasing the camera to unblock it."
            )
            video.release()
            camera_released = True
            capture_worker.thread.join(timeout=2.0)
        if capture_worker.thread.is_alive():
            print("[ERROR] Camera capture thread did not stop.")
    if not camera_released:
        video.release()
    if zoom_csv_file is not None:
        zoom_csv_file.close()
    captured_count, dropped_count = capture_worker.get_statistics()
    print(
        f"[PERF] captured={captured_count} frames "
        f"processed={processed_frame_count} "
        f"dropped={dropped_count}"
    )
    cv2.destroyAllWindows()

if (
    (
        ENABLE_COMPLEMENTARY_ZOOM
        or ENABLE_ZOOM_BENCHMARK
        or SAVE_ZOOM_CSV
    )
    and zoom_summary["frames"] > 0
):
    measured_frames = zoom_summary["frames"]
    original_rate = (
        100.0 * zoom_summary["original_recognized"] / measured_frames
    )
    zoom_3x_rate = (
        100.0 * zoom_summary["benchmark_3x_recognized"]
        / zoom_summary["benchmark_3x_frames"]
        if zoom_summary["benchmark_3x_frames"] else None
    )
    zoom_4x_rate = (
        100.0 * zoom_summary["benchmark_4x_recognized"]
        / zoom_summary["benchmark_4x_frames"]
        if zoom_summary["benchmark_4x_frames"] else None
    )
    operational_fps = (
        measured_frames / zoom_summary["operational_total_seconds"]
        if zoom_summary["operational_total_seconds"] > 0 else 0.0
    )
    benchmark_fps = (
        measured_frames / zoom_summary["benchmark_total_seconds"]
        if zoom_summary["benchmark_total_seconds"] > 0 else 0.0
    )
    mean_original_ms = (
        1000.0 * zoom_summary["original_seconds"] / measured_frames
    )
    mean_3x_ms = (
        1000.0 * zoom_summary["benchmark_3x_seconds"]
        / zoom_summary["benchmark_3x_frames"]
        if zoom_summary["benchmark_3x_frames"] else None
    )
    mean_4x_ms = (
        1000.0 * zoom_summary["benchmark_4x_seconds"]
        / zoom_summary["benchmark_4x_frames"]
        if zoom_summary["benchmark_4x_frames"] else None
    )
    zoom_3x_rate_text = (
        f"{zoom_3x_rate:.1f}%" if zoom_3x_rate is not None else "N/A"
    )
    zoom_4x_rate_text = (
        f"{zoom_4x_rate:.1f}%" if zoom_4x_rate is not None else "N/A"
    )
    mean_3x_text = f"{mean_3x_ms:.2f}" if mean_3x_ms is not None else "N/A"
    mean_4x_text = f"{mean_4x_ms:.2f}" if mean_4x_ms is not None else "N/A"
    print("\n[ZOOM] Performance summary")
    print(
        f"Recognition rates (of evaluated frames): "
        f"original={original_rate:.1f}% "
        f"zoom_3x={zoom_3x_rate_text} "
        f"zoom_4x={zoom_4x_rate_text}"
    )
    if zoom_summary["benchmark_3x_frames"] or zoom_summary["benchmark_4x_frames"]:
        print(
            f"Exclusive benchmark recoveries: "
            f"3x={zoom_summary['exclusive_3x']} "
            f"4x={zoom_summary['exclusive_4x']}"
        )
    else:
        print("[ZOOM] Exclusive benchmark recoveries: N/A (benchmark not run)")
    if zoom_summary["benchmark_3x_frames"] or zoom_summary["benchmark_4x_frames"]:
        print(
            "Zoom candidates rejected by Kalman "
            "(not proof of false positives): "
            f"{zoom_summary['kalman_rejected_zoom_candidates']}"
        )
    else:
        print("[ZOOM] Kalman-rejected zoom candidates: 0")
    print(
        f"Mean recognition time (original/3x/4x): "
        f"{mean_original_ms:.2f}/{mean_3x_text}/{mean_4x_text} ms"
    )
    if ENABLE_ZOOM_BENCHMARK:
        print(
            f"Operational mean FPS={operational_fps:.2f}; "
            f"benchmark-inclusive FPS={benchmark_fps:.2f}"
        )
    else:
        print(
            f"Operational mean FPS={operational_fps:.2f}; "
            "benchmark disabled"
        )
    real_loop_fps = (
        measured_frames / zoom_summary["real_loop_seconds"]
        if zoom_summary["real_loop_seconds"] > 0 else 0.0
    )
    print(f"Real full-loop FPS (capture, processing, display): {real_loop_fps:.2f}")
    if zoom_csv_path is not None:
        print(f"[ZOOM] CSV saved: {zoom_csv_path}")

print("\n[OK] Program finished.")
