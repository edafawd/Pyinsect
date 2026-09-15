import os
import csv
from datetime import datetime

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.clock import Clock
from kivy.utils import platform

from PIL import Image as PILImage
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import tflite_runtime.interpreter as tflite

MODEL_FILE = "model.tflite"
CLASS_FILE = "class_names.txt"
LOG_FILE = "invasive_detections.csv"

INVASIVE_SPECIES = {
    "japanese_beetle",
    "spotted_lanternfly",
    "brown_marmorated_stink_bug",
    "emerald_ash_borer",
    "asian_longhorned_beetle",
    "spongy_moth",
    "northern_giant_hornet",
    "light_brown_apple_moth",
    "european_cherry_fruit_fly",
    "mediterranean_oak_borer",
}


def get_gps_from_image(image_path):
    try:
        img = PILImage.open(image_path)
        exif = img._getexif()
        if not exif:
            return None

        gps_info = {}
        for tag, value in exif.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub = GPSTAGS.get(t, t)
                    gps_info[sub] = value[t]

        if "GPSLatitude" not in gps_info or "GPSLongitude" not in gps_info:
            return None

        def to_degrees(v):
            d, m, s = float(v[0]), float(v[1]), float(v[2])
            return d + m / 60.0 + s / 3600.0

        lat = to_degrees(gps_info["GPSLatitude"])
        if gps_info.get("GPSLatitudeRef") == "S":
            lat = -lat

        lon = to_degrees(gps_info["GPSLongitude"])
        if gps_info.get("GPSLongitudeRef") == "W":
            lon = -lon

        return lat, lon
    except Exception:
        return None


def make_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"


def ensure_csv_header():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "image_path", "species", "confidence_percent",
                "latitude", "longitude", "google_maps_link", "has_gps"
            ])


def log_detection(image_path, species, confidence, gps):
    ensure_csv_header()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if gps:
        lat, lon = gps
        link = make_maps_link(lat, lon)
        has_gps = "yes"
    else:
        lat = lon = link = ""
        has_gps = "no"

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, image_path, species, f"{confidence:.2f}",
            lat, lon, link, has_gps
        ])


def preprocess_image(image_path):
    img = PILImage.open(image_path).convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    # TFLite models converted via onnx2tf typically use batch dimension [1, 224, 224, 3]
    arr = np.expand_dims(arr, axis=0)
    return arr


class InsectApp(App):
    def build(self):
        self.interpreter = None
        self.class_names = []
        self.last_image_path = None

        root = BoxLayout(orientation="vertical", padding=12, spacing=10)

        self.preview = KivyImage(size_hint=(1, 0.55))
        root.add_widget(self.preview)

        self.status = Label(
            text="Loading TFLite model...",
            size_hint=(1, 0.18),
            halign="center",
            valign="middle",
        )
        self.status.bind(size=lambda inst, val: setattr(inst, "text_size", (inst.width * 0.95, None)))
        root.add_widget(self.status)

        row = BoxLayout(size_hint=(1, 0.14), spacing=8)
        btn_camera = Button(text="Camera")
        btn_gallery = Button(text="Gallery")
        btn_detect = Button(text="Detect")

        btn_camera.bind(on_press=self.take_photo)
        btn_gallery.bind(on_press=self.pick_from_gallery)
        btn_detect.bind(on_press=self.run_detection)

        row.add_widget(btn_camera)
        row.add_widget(btn_gallery)
        row.add_widget(btn_detect)
        root.add_widget(row)

        Clock.schedule_once(lambda dt: self.load_model(), 0.4)
        return root

    def load_model(self):
        try:
            if not os.path.exists(MODEL_FILE):
                self.status.text = f"Missing {MODEL_FILE}"
                return

            self.interpreter = tflite.Interpreter(model_path=MODEL_FILE)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            if os.path.exists(CLASS_FILE):
                with open(CLASS_FILE, "r", encoding="utf-8") as f:
                    self.class_names = [line.strip() for line in f if line.strip()]

            self.status.text = f"Model loaded ({len(self.class_names)} classes)\nTake or select a photo"
        except Exception as e:
            self.status.text = f"Model load error:\n{e}"

    def take_photo(self, *args):
        if platform != "android":
            self.status.text = "Camera works only in the Android APK"
            return
        try:
            from plyer import camera
            filename = f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            path = os.path.join(self.user_data_dir, filename)
            camera.take_picture(filename=path, on_complete=self.on_photo_taken)
        except Exception as e:
            self.status.text = f"Camera error:\n{e}"

    def on_photo_taken(self, path):
        if path and os.path.exists(path):
            self.last_image_path = path
            self.preview.source = path
            self.preview.reload()
            self.status.text = f"Photo ready:\n{os.path.basename(path)}"
        else:
            self.status.text = "Camera capture failed"

    def pick_from_gallery(self, *args):
        if platform != "android":
            self.status.text = "Gallery works only in the Android APK"
            return
        try:
            from jnius import autoclass, cast
            from android import activity

            Intent = autoclass("android.content.Intent")
            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            intent.addCategory(Intent.CATEGORY_OPENABLE)

            currentActivity = cast("android.app.Activity", PythonActivity.mActivity)
            currentActivity.startActivityForResult(intent, 42)

            def on_activity_result(request_code, result_code, result_intent):
                if request_code != 42:
                    return
                if result_code != -1 or result_intent is None:
                    self.status.text = "Gallery cancelled"
                    return
                try:
                    uri = result_intent.getData()
                    path = self._android_uri_to_path(uri)
                    if path and os.path.exists(path):
                        self.last_image_path = path
                        self.preview.source = path
                        self.preview.reload()
                        self.status.text = f"Selected:\n{os.path.basename(path)}"
                    else:
                        self.status.text = "Could not read gallery image"
                except Exception as e:
                    self.status.text = f"Gallery error:\n{e}"

            activity.bind(on_activity_result=on_activity_result)
        except Exception as e:
            self.status.text = f"Gallery setup error:\n{e}"

    def _android_uri_to_path(self, uri):
        try:
            from jnius import autoclass, cast
            PythonActivity = autoclass("org.kivy.android.PythonActivity")
            resolver = PythonActivity.mActivity.getContentResolver()
            input_stream = resolver.openInputStream(uri)

            out_path = os.path.join(
                self.user_data_dir,
                f"gallery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )

            FileOutputStream = autoclass("java.io.FileOutputStream")
            fos = FileOutputStream(out_path)

            try:
                data = input_stream.readAllBytes()
                fos.write(data)
            except Exception:
                b = input_stream.read()
                while b != -1:
                    fos.write(b)
                    b = input_stream.read()

            fos.close()
            input_stream.close()
            return out_path
        except Exception:
            try:
                return uri.getPath()
            except Exception:
                return None

    def run_detection(self, *args):
        if self.interpreter is None:
            self.status.text = "Model not loaded"
            return
        if not self.last_image_path or not os.path.exists(self.last_image_path):
            self.status.text = "No image selected"
            return

        try:
            arr = preprocess_image(self.last_image_path)
            
            # Set tensor and invoke interpreter
            self.interpreter.set_tensor(self.input_details[0]['index'], arr)
            self.interpreter.invoke()
            
            output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

            # softmax
            exp = np.exp(output - np.max(output))
            probs = exp / np.sum(exp)
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx] * 100.0)

            if self.class_names and pred_idx < len(self.class_names):
                species = self.class_names[pred_idx]
            else:
                species = f"class_{pred_idx}"

            gps = get_gps_from_image(self.last_image_path)

            msg = f"{species}\nConfidence: {confidence:.1f}%"
            if gps:
                msg += f"\nGPS: {gps[0]:.6f}, {gps[1]:.6f}"
                msg += f"\n{make_maps_link(gps[0], gps[1])}"
            else:
                msg += "\nNo GPS in photo"

            if species.lower() in INVASIVE_SPECIES:
                log_detection(self.last_image_path, species, confidence, gps)
                msg += "\n\n[INVASIVE LOGGED]"

            self.status.text = msg
        except Exception as e:
            self.status.text = f"Detection error:\n{e}"


if __name__ == "__main__":
    InsectApp().run()
