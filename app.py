import toga
from toga.style import Pack
from toga.style.pack import COLUMN
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

class InsectDetector(toga.App):
    def startup(self):
        model_path = str(self.paths.app / "resources" / "insect_model.tflite")
        
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            status_text = "Model loaded successfully!"
        except Exception as e:
            status_text = f"Error loading model: {e}"

        main_box = toga.Box(style=Pack(direction=COLUMN, padding=15))
        self.status_label = toga.Label(status_text, style=Pack(padding=5))
        main_box.add(self.status_label)

        self.image_viewer = toga.ImageView(style=Pack(width=300, height=300, padding=10))
        main_box.add(self.image_viewer)

        scan_button = toga.Button(
            'Select Insect Image', 
            on_press=self.open_file_dialog,
            style=Pack(padding=10)
        )
        main_box.add(scan_button)

        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    async def open_file_dialog(self, widget):
        try:
            file_path = await self.main_window.select_file_dialog(
                title="Select an Insect Image",
                file_types=['png', 'jpg', 'jpeg']
            )
            if file_path is not None:
                self.image_viewer.image = file_path
                self.process_and_predict(file_path)
        except Exception as e:
            self.status_label.text = f"File selection error: {e}"

    def process_and_predict(self, image_path):
        self.status_label.text = "Processing image..."
        
        target_height = 224
        target_width = 224

        img = Image.open(image_path).convert('RGB')
        img = img.resize((target_width, target_height))

        input_data = np.array(img, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)

        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details['index'])
        best_match_index = np.argmax(output_data)
        confidence = output_data[0][best_match_index]

        self.status_label.text = f"Result index: {best_match_index} ({confidence*100:.1f}% confidence)"
