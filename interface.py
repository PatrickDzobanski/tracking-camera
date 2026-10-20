import csv
import io
import sys
import tkinter as tk
import cv2
import numpy as np
from tkinter import messagebox, ttk
from enviar_raspberry import enviar_arquivos
from PIL import Image, ImageTk
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

CAMERA = 0
OUTPUT_DIR = Path(__file__).resolve().parent
IMAGE_FILENAME = "img.jpg"
CSV_FILENAME = "hsv.csv"
JPEG_QUALITY = 95

# Initial values for yellow; adjust the sliders for your target.
# OpenCV ranges: H is 0-179; S and V are 0-255.
INITIAL_HSV = {
    "hmin": 18,
    "smin": 118,
    "vmin": 120,
    "hmax": 27,
    "smax": 255,
    "vmax": 255,
}

PREVIEW_WIDTH = 480
PREVIEW_HEIGHT = 300
UPDATE_INTERVAL_MS = 30
MAX_CAMERA_FAILURES = 5

# HSV processing and file output


def validate_hsv(values):
    """Validate channel ranges without silently changing selected values."""
    for channel, limit in (("h", 179), ("s", 255), ("v", 255)):
        minimum = int(values[f"{channel}min"])
        maximum = int(values[f"{channel}max"])
        if not 0 <= minimum <= maximum <= limit:
            raise ValueError(
                f"{channel.upper()}: use minimum <= maximum, between 0 and {limit}."
            )


def segment(image, values):
    """Apply HSV thresholding and return the selected image and mask."""
    validate_hsv(values)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(
        [values["hmin"], values["smin"], values["vmin"]], dtype=np.uint8
    )
    upper = np.array(
        [values["hmax"], values["smax"], values["vmax"]], dtype=np.uint8
    )
    mask = cv2.inRange(hsv, lower, upper)
    selected = cv2.bitwise_and(image, image, mask=mask)
    return selected, mask


def save_files(image, values, output_dir):
    """Save the captured image and a row containing its six HSV thresholds."""
    validate_hsv(values)
    if image is None or image.size == 0:
        raise ValueError("Capture a photo before saving.")

    success, jpeg = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not success:
        raise OSError("Could not encode the JPEG image.")

    fields = ["hmin", "smin", "vmin", "hmax", "smax", "vmax"]
    csv_content = io.StringIO(newline="")
    writer = csv.DictWriter(csv_content, fieldnames=fields)
    writer.writeheader()
    writer.writerow({field: int(values[field]) for field in fields})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / IMAGE_FILENAME
    csv_path = output_dir / CSV_FILENAME

    image_path.write_bytes(jpeg.tobytes())
    csv_path.write_text(csv_content.getvalue(), encoding="utf-8", newline="")
    return image_path, csv_path


class HSVInterface:
    def __init__(self, window):
        self.window = window
        self.video = None
        self.current_frame = None
        self.captured_photo = None
        self.camera_failures = 0
        self.closed = False
        self.scheduled_update = None
        self.preview_mode = tk.StringVar(value="selected_color")
        self.hsv_vars = {
            name: tk.IntVar(value=value) for name, value in INITIAL_HSV.items()
        }
        self.status = tk.StringVar(value="Position the target and capture a photo.")
        self.state = tk.StringVar(value="LIVE CAMERA")

        self.build_interface()
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<KeyPress>", self.handle_key)

        try:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            self.video = cv2.VideoCapture(CAMERA, backend)
            if not self.video.isOpened() and sys.platform.startswith("win"):
                self.video.release()
                self.video = cv2.VideoCapture(CAMERA)
            if not self.video.isOpened():
                raise RuntimeError(f"Could not open camera {CAMERA}.")
            self.update()
        except (RuntimeError, cv2.error) as error:
            messagebox.showerror("Camera", str(error), parent=self.window)
            self.close()

    def build_interface(self):
        self.window.title("Capture Photo and Adjust HSV")
        self.window.resizable(False, False)
        panel = ttk.Frame(self.window, padding=14)
        panel.pack(fill="both", expand=True)

        ttk.Label(panel, text="Photo + HSV Thresholds", font=("Arial", 17, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 4)
        )
        ttk.Label(panel, textvariable=self.state).grid(row=0, column=1, sticky="e")
        ttk.Label(panel, text="1. Capture a photo   2. Adjust HSV   3. Save the files").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )

        ttk.Label(panel, text="Camera / Original Photo").grid(row=2, column=0, sticky="w")
        preview_modes = ttk.Frame(panel)
        preview_modes.grid(row=2, column=1, sticky="w")
        ttk.Radiobutton(
            preview_modes, text="Selected Color", variable=self.preview_mode, value="selected_color"
        ).pack(side="left")
        ttk.Radiobutton(
            preview_modes, text="Mask", variable=self.preview_mode, value="mask"
        ).pack(side="left", padx=15)

        self.original_preview = ttk.Label(panel)
        self.original_preview.grid(row=3, column=0, padx=(0, 10), pady=6)
        self.filtered_preview = ttk.Label(panel)
        self.filtered_preview.grid(row=3, column=1, pady=6)
        blank_image = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
        self.display_image(self.original_preview, blank_image)
        self.display_image(self.filtered_preview, blank_image)

        controls = ttk.LabelFrame(panel, text="HSV Thresholds", padding=8)
        controls.grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)
        for row, (channel, title, limit) in enumerate(
            (("h", "H - Hue", 179), ("s", "S - Saturation", 255), ("v", "V - Value", 255))
        ):
            for column, endpoint in enumerate(("min", "max")):
                group = ttk.Frame(controls)
                group.grid(row=row, column=column, sticky="ew", padx=8)
                ttk.Label(group, text=f"{title} ({endpoint})", width=24).pack(side="left")
                tk.Scale(
                    group, from_=0, to=limit, orient="horizontal", length=270,
                    variable=self.hsv_vars[channel + endpoint], resolution=1,
                    highlightthickness=0,
                ).pack(side="left")

        buttons = ttk.Frame(panel)
        buttons.grid(row=5, column=0, columnspan=2, sticky="w", pady=8)
        self.capture_button = ttk.Button(
            buttons, text="Capture Photo (C)", command=self.capture_photo, state="disabled"
        )
        self.capture_button.pack(side="left")
        self.resume_button = ttk.Button(
            buttons, text="Resume Camera", command=self.resume_camera, state="disabled"
        )
        self.resume_button.pack(side="left", padx=8)
        self.save_button = ttk.Button(
            buttons, text="Save Photo + HSV (S)", command=self.save, state="disabled"
        )
        self.save_button.pack(side="left")
        self.send_button = ttk.Button(
            buttons,
            text="Enviar para Raspberry",
            command=self.enviar_para_raspberry,
            state="disabled",
        )
        self.send_button.pack(side="left", padx=8)
        ttk.Button(buttons, text="Quit (Q)", command=self.close).pack(side="left", padx=8)

        ttk.Label(panel, textvariable=self.status, wraplength=970).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )
        ttk.Label(panel, text=f"Output: {OUTPUT_DIR} | {IMAGE_FILENAME} + {CSV_FILENAME}", wraplength=970).grid(
            row=7, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

    def current_hsv(self):
        return {name: variable.get() for name, variable in self.hsv_vars.items()}

    def display_image(self, label, image):
        if image is None:
            return
        if image.ndim == 2:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_image.thumbnail((PREVIEW_WIDTH, PREVIEW_HEIGHT))
        background = Image.new("RGB", (PREVIEW_WIDTH, PREVIEW_HEIGHT), "#171717")
        background.paste(
            pil_image,
            ((PREVIEW_WIDTH - pil_image.width) // 2, (PREVIEW_HEIGHT - pil_image.height) // 2),
        )
        tk_image = ImageTk.PhotoImage(background)
        label.configure(image=tk_image)
        label.image = tk_image

    def update(self):
        if self.closed:
            return
        if self.captured_photo is None:
            try:
                success, frame = self.video.read()
            except cv2.error:
                success, frame = False, None
            if not success or frame is None or frame.size == 0:
                self.current_frame = None
                self.capture_button.configure(state="disabled")
                self.camera_failures += 1
                self.status.set("Failed to receive an image from the camera.")
                if self.camera_failures >= MAX_CAMERA_FAILURES:
                    messagebox.showerror(
                        "Camera",
                        "The camera stopped providing images. Check the connection and reopen the application.",
                        parent=self.window,
                    )
                    self.close()
                    return
                self.scheduled_update = self.window.after(UPDATE_INTERVAL_MS, self.update)
                return
            if self.camera_failures:
                self.status.set("Camera recovered. Position the target and capture a photo.")
            self.camera_failures = 0
            self.current_frame = frame
            self.capture_button.configure(state="normal")

        image = self.captured_photo if self.captured_photo is not None else self.current_frame
        if image is None:
            blank_image = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
            self.display_image(self.original_preview, blank_image)
            self.display_image(self.filtered_preview, blank_image)
            self.scheduled_update = self.window.after(UPDATE_INTERVAL_MS, self.update)
            return

        self.display_image(self.original_preview, image)
        try:
            selected, mask = segment(image, self.current_hsv())
            self.display_image(
                self.filtered_preview,
                mask if self.preview_mode.get() == "mask" else selected,
            )
            if self.captured_photo is not None:
                self.save_button.configure(state="normal")
            if self.state.get().startswith("INVALID HSV"):
                self.state.set("PHOTO CAPTURED" if self.captured_photo is not None else "LIVE CAMERA")
        except ValueError as error:
            self.state.set(f"INVALID HSV: {error}")
            self.save_button.configure(state="disabled")
            self.display_image(self.filtered_preview, np.zeros_like(image))

        self.scheduled_update = self.window.after(UPDATE_INTERVAL_MS, self.update)

    def capture_photo(self):
        if self.captured_photo is not None or self.current_frame is None:
            return
        self.captured_photo = self.current_frame.copy()
        self.state.set("PHOTO CAPTURED")
        self.capture_button.configure(state="disabled")
        self.resume_button.configure(state="normal")
        self.send_button.configure(state="normal")
        self.status.set("Photo captured. Adjust the HSV thresholds and click Save Photo + HSV.")

    def resume_camera(self):
        self.captured_photo = None
        self.current_frame = None
        self.state.set("LIVE CAMERA")
        self.capture_button.configure(state="disabled")
        self.resume_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.send_button.configure(state="disabled")
        blank_image = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
        self.display_image(self.original_preview, blank_image)
        self.display_image(self.filtered_preview, blank_image)
        self.status.set("Position the target and capture a new photo.")

    def save(self):
        if self.captured_photo is None:
            self.status.set("Capture a photo before saving.")
            return
        try:
            image_path, csv_path = save_files(self.captured_photo, self.current_hsv(), OUTPUT_DIR)
        except (OSError, ValueError, cv2.error) as error:
            messagebox.showerror("Save Error", str(error), parent=self.window)
            return
        self.status.set(f"Saved: {image_path} and {csv_path}")

    def enviar_para_raspberry(self):
        if self.captured_photo is None:
            messagebox.showwarning("Aviso", "Capture uma foto antes de enviar.", parent=self.window)
            self.status.set("Capture a photo before sending to Raspberry.")
            return
        try:
            image_path, csv_path = save_files(self.captured_photo, self.current_hsv(), OUTPUT_DIR)
            self.status.set("Enviando arquivos para Raspberry...")
            self.window.update_idletasks()

            sucesso = enviar_arquivos(csv_path, image_path)

            if sucesso:
                self.status.set("Arquivos enviados com sucesso!")
                messagebox.showinfo(
                    "Sucesso",
                    "Imagem e CSV enviados para Raspberry!",
                    parent=self.window,
                )
            else:
                self.status.set("Falha na transferência.")
                messagebox.showerror(
                    "Erro",
                    "Não foi possível enviar os arquivos.",
                    parent=self.window,
                )

        except Exception as erro:
            messagebox.showerror("Erro", str(erro), parent=self.window)

    def handle_key(self, event):
        key = event.keysym.lower()
        if key == "c":
            self.capture_photo()
        elif key == "s":
            self.save()
        elif key in ("q", "escape"):
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.scheduled_update is not None:
            self.window.after_cancel(self.scheduled_update)
            self.scheduled_update = None
        if self.video is not None:
            self.video.release()
        self.window.destroy()


def main():
    validate_hsv(INITIAL_HSV)
    window = tk.Tk()
    HSVInterface(window)
    window.mainloop()

if __name__ == "__main__":
    main()
