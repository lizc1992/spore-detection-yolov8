import shutil
import sys
import os
import csv
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QProgressBar, QTextEdit
)
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QPixmap
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


class ProcessingThread(QThread):
    """Thread for running object detection to keep UI responsive"""
    progress = Signal(int)
    finished = Signal(dict)
    log = Signal(str)

    def __init__(self, input_folder):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = None
        self.result_folder = None
        self.intermediate_folder = None
        # Load YOLO model
        model_path = resource_path("best.pt")
        self.log.emit(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

    def split_and_save(self, img_path):
        img = mpimg.imread(img_path)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        height, width = img.shape[0], img.shape[1]
        mid_y, mid_x = height // 2, width // 2

        quadrants = [
            img[:mid_y, :mid_x],  # Top-left
            img[:mid_y, mid_x:],  # Top-right
            img[mid_y:, :mid_x],  # Bottom-left
            img[mid_y:, mid_x:],  # Bottom-right
        ]

        for i, quadrant in enumerate(quadrants, start=1):
            save_path = os.path.join(self.intermediate_folder, f"{basename}_{i}.png")
            plt.imsave(save_path, quadrant)

    def draw_bboxes(self, image, boxes):
        vis = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0)  # Green box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        return vis

    def merge_and_save(self, img_paths, save_path):
        """
        Merge 4 images into one big image and save.

        img_paths: list or tuple of 4 paths in this exact order:
            [top-left, top-right, bottom-left, bottom-right]
        save_path: path to write merged image (png/jpg)
        """
        if len(img_paths) != 4:
            raise ValueError("Provide exactly 4 image paths: [TL, TR, BL, BR].")

        imgs = [mpimg.imread(p) for p in img_paths]

        # Convert all to numpy arrays and unify dtype
        imgs = [np.array(im) for im in imgs]
        # All images must have same height/width for their quadrants:
        hs = [im.shape[0] for im in imgs]
        ws = [im.shape[1] for im in imgs]
        # Check top row heights equal and bottom row heights equal, left col widths equal, right col widths equal
        if not (hs[0] == hs[1] and hs[2] == hs[3] and ws[0] == ws[2] and ws[1] == ws[3]):
            raise ValueError(
                "Quadrant sizes mismatch. Required: "
                "top-left.height == top-right.height and bottom-left.height == bottom-right.height, "
                "top-left.width == bottom-left.width and top-right.width == bottom-right.width. "
                f"Got heights: {hs}, widths: {ws}"
            )

        # Determine channel target (max channels among images)
        chans = [im.shape[2] if im.ndim == 3 else 1 for im in imgs]
        target_ch = max(chans)

        # Build top and bottom rows
        top = np.concatenate((imgs[0], imgs[1]), axis=1)  # TL + TR
        bottom = np.concatenate((imgs[2], imgs[3]), axis=1)  # BL + BR

        merged = np.concatenate((top, bottom), axis=0)

        # Ensure folder exists and save
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.imsave(save_path, merged)
        print(f"Saved merged image to: {save_path}")

    def run(self):
        try:
            self.log.emit(f"Processing folder: {self.input_folder}")

            # Get image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            input_path = Path(self.input_folder)
            image_files = sorted([f for f in input_path.iterdir()
                          if f.suffix.lower() in image_extensions])

            total_images = len(image_files)
            self.log.emit(f"Found {total_images} images - {total_images*4} split images to process:")

            if total_images == 0:
                self.log.emit("No images found in folder!")
                self.finished.emit({'total_objects': 0, 'images_processed': 0})
                return

            # Create intermediate folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_folder = input_path / f"{timestamp}"
            self.output_folder.mkdir(exist_ok=True)

            self.intermediate_folder = self.output_folder / f"{input_path.name}_intermediate"
            self.intermediate_folder.mkdir(exist_ok=True)

            # Split images
            for img_path in image_files:
                self.split_and_save(img_path)

            # Create output folder
            self.result_folder = self.output_folder / f"{input_path.name}_results"
            self.result_folder.mkdir(exist_ok=True)

            # Create CSV file
            csv_path = self.output_folder / f"detection_results_{timestamp}.csv"
            csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Image_Path', 'Object_Count'])


            total_objects = 0

            # Process each original image
            for img_idx, image_file in enumerate(image_files):
                basename = image_file.stem
                image_object_count = 0

                # Process all 4 quadrants of this image
                img_paths = []
                for quadrant_idx in range(1, 5):
                    quadrant_filename = f"{basename}_{quadrant_idx}.png"
                    quadrant_path = self.intermediate_folder / quadrant_filename

                    if not quadrant_path.exists():
                        self.log.emit(f"Warning: {quadrant_filename} not found")
                        continue

                    img_path = str(quadrant_path)
                    image = cv2.imread(img_path)
                    if image is None:
                        self.log.emit(f"Failed to read {quadrant_filename}")
                        continue

                    results = self.model.predict(img_path, conf=0.25, iou=0.6, verbose=False)

                    boxes_all = []
                    for r in results:
                        if r.boxes is not None and len(r.boxes.xyxy) > 0:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            boxes_all.extend(boxes)

                    if boxes_all:
                        count = len(boxes_all)
                        image_object_count += count
                        bbox_img = self.draw_bboxes(image, boxes_all)
                        save_path = os.path.join(self.result_folder, f"{quadrant_path.stem}.png")
                        img_paths.append(save_path)
                        cv2.imwrite(save_path, bbox_img)
                        self.log.emit(f"    Processed {quadrant_filename}: {count} objects")
                    else:
                        self.log.emit(f"No objects detected in {quadrant_filename}")

                # Write row to CSV for this original image
                csv_writer.writerow([str(image_file), image_object_count])
                total_objects += image_object_count
                save_path = os.path.join(self.result_folder, f"{basename}_spore_count_{image_object_count}.png")
                self.merge_and_save(img_paths, save_path)

                self.log.emit(f"        → Image {image_file.name}: Total objects = {image_object_count}")

                # Update progress
                progress = int((img_idx + 1) / total_images * 100)
                self.progress.emit(progress)

            # Close CSV file
            csv_file.close()

            # Delete intermediate folder
            if self.intermediate_folder and self.intermediate_folder.exists():
                try:
                    shutil.rmtree(self.intermediate_folder)
                except Exception as e:
                    self.log.emit(f"Warning: Could not delete intermediate folder: {e}")

            # Emit results
            results = {
                'total_objects': total_objects,
                'images_processed': total_images,
                'split_images_processed': total_images * 4,
                'output_folder': str(self.result_folder),
                'csv_path': str(csv_path)
            }
            self.finished.emit(results)

        except Exception as e:
            self.log.emit(f"Error: {str(e)}")
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished.emit({'error': str(e)})


class ObjectDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_folder = None
        self.processing_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Spores Counter Tool by LO")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # Logo section
        logo_layout = QHBoxLayout()
        self.left_logo = QLabel()
        self.left_logo.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.left_logo.setFixedSize(180, 60)
        self.left_logo.setScaledContents(True)
        left_logo_path = resource_path("LOlogo.png")
        if os.path.exists(left_logo_path):
            pixmap = QPixmap(left_logo_path)
            self.left_logo.setPixmap(pixmap if not pixmap.isNull() else QPixmap())
        else:
            self._set_logo_placeholder(self.left_logo, "Logo 1")
        logo_layout.addWidget(self.left_logo)
        logo_layout.addStretch()
        self.right_logo = QLabel()
        self.right_logo.setAlignment(Qt.AlignRight | Qt.AlignTop)
        self.right_logo.setFixedSize(180, 60)
        self.right_logo.setScaledContents(True)
        right_logo_path = resource_path("zoologo.png")
        if os.path.exists(right_logo_path):
            pixmap = QPixmap(right_logo_path)
            self.right_logo.setPixmap(pixmap if not pixmap.isNull() else QPixmap())
        else:
            self._set_logo_placeholder(self.right_logo, "Logo 2")
        logo_layout.addWidget(self.right_logo)
        layout.addLayout(logo_layout)

        # Title
        title = QLabel("Spores Counter Tool by LO")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Folder selection
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("""
            QLabel { border: 2px dashed #cccccc; border-radius: 5px; padding: 10px; background-color: #f9f9f9; }
        """)
        folder_layout.addWidget(self.folder_label, stretch=1)
        self.select_button = QPushButton("Select Folder")
        self.select_button.setMinimumHeight(40)
        self.select_button.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; border: none; border-radius: 5px; padding: 10px 20px; font-size: 14px; }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.select_button.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.select_button)
        layout.addLayout(folder_layout)

        # Run button
        self.run_button = QPushButton("Run Detection")
        self.run_button.setMinimumHeight(50)
        self.run_button.setEnabled(False)
        self.run_button.setStyleSheet("""
            QPushButton { background-color: #2196F3; color: white; border: none; border-radius: 5px; padding: 15px; font-size: 16px; font-weight: bold; }
            QPushButton:hover { background-color: #0b7dda; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.run_button.clicked.connect(self.run_detection)
        layout.addWidget(self.run_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Log area
        log_label = QLabel("Log:")
        log_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit { border: 1px solid #cccccc; border-radius: 5px; padding: 10px; background-color: #f5f5f5; font-family: Consolas, monospace; }
        """)
        layout.addWidget(self.log_text)

        # Results label
        self.results_label = QLabel("")
        self.results_label.setStyleSheet("""
            QLabel { font-size: 14px; font-weight: bold; color: #2196F3; padding: 10px; }
        """)
        self.results_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.results_label)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            "",
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks
        )
        if folder:
            self.selected_folder = folder
            self.folder_label.setText(folder)
            self.run_button.setEnabled(True)
            self.log_text.append(f"Selected folder: {folder}")

    def run_detection(self):
        if not self.selected_folder:
            return
        self.run_button.setEnabled(False)
        self.select_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.results_label.setText("")
        self.log_text.clear()

        self.processing_thread = ProcessingThread(self.selected_folder)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.log.connect(self.add_log)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def add_log(self, message):
        self.log_text.append(message)

    def processing_finished(self, results):
        self.run_button.setEnabled(True)
        self.select_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        if 'error' in results:
            self.results_label.setText(f"❌ Error: {results['error']}")
            self.results_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.log_text.append("\n=== Processing Complete ===")
            self.log_text.append(f"Images processed: {results['images_processed']}")
            self.log_text.append(f"Split images processed: {results['split_images_processed']}")
            self.log_text.append(f"Total objects detected: {results['total_objects']}")
            if 'output_folder' in results:
                self.log_text.append(f"Results saved to: {results['output_folder']}")
            if 'csv_path' in results:
                self.log_text.append(f"CSV file saved to: {results['csv_path']}")

            self.results_label.setText(
                f"✓ Success! Processed {results['images_processed']} images, "
                f"detected {results['total_objects']} objects. CSV saved!"
            )
            self.results_label.setStyleSheet("color: green; font-weight: bold;")

    def _set_logo_placeholder(self, label, text):
        label.setText(text)
        label.setStyleSheet("""
            QLabel { border: 2px dashed #cccccc; background-color: #f0f0f0; color: #999999; }
        """)


def main():
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()