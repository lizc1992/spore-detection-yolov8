
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Circle
import numpy as np
import json
import argparse
import os


class SporeMarkingTool:

    RADIUS = 15  # Radius of the marker circle

    def __init__(self, image_path, json_path):
        """
        Initialize the spore marking tool with an image

        Args:
            image_path (str): Path to the microscopy image
        """
        self.image_path = image_path
        self.json_path = json_path
        self.markers = []
        self.fig = None
        self.ax = None
        self.image = None
        self.marker_circles = []

        # Load and display the image
        self.setup_plot()

    def setup_plot(self):
        """Set up the matplotlib plot with the image"""
        try:
            # Load the image
            self.image = mpimg.imread(self.image_path)

            # Create figure and axis
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.ax.imshow(self.image)
            self.ax.set_title("Click on spores to mark them\nRight-click to remove nearest marker")

            # Connect mouse click event
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)

            # Remove axis ticks for cleaner look
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            self.load_markers(self.json_path)

            print("Instructions:")
            print("- Left click: Add spore marker")
            print("- Right click: Remove nearest marker")
            print("- Close window or press Ctrl+C to finish")

        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        if event.button == 1:  # Left click - add marker
            self.add_marker(x, y)
        elif event.button == 3:  # Right click - remove nearest marker
            self.remove_nearest_marker(x, y)

    def add_marker(self, x, y):
        """Add a spore marker at the clicked position"""
        marker_id = len(self.markers)
        self.markers.append({'id': marker_id, 'x': x, 'y': y})

        # Create visual marker (red circle with white border)
        circle = Circle((x, y), radius=self.RADIUS, fill=False,
                        edgecolor='red', linewidth=2, alpha=0.8)
        self.ax.add_patch(circle)
        self.marker_circles.append(circle)

        # Add number label
        text = self.ax.text(x + 7, y - 7, str(marker_id + 1),
                            color='red', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor='white', alpha=0.8))

        self.fig.canvas.draw()
        print(f"Marker {marker_id + 1} added at ({x:.1f}, {y:.1f})")

    def remove_nearest_marker(self, x, y):
        """Remove the marker nearest to the clicked position"""
        if not self.markers:
            return

        # Find nearest marker
        distances = []
        for i, marker in enumerate(self.markers):
            dist = np.sqrt((marker['x'] - x) ** 2 + (marker['y'] - y) ** 2)
            distances.append((dist, i))

        # Get index of nearest marker
        nearest_dist, nearest_idx = min(distances)

        if nearest_dist < self.RADIUS*2:  # Only remove if click is close enough
            # Remove from markers list
            removed_marker = self.markers.pop(nearest_idx)

            # Remove visual elements
            if nearest_idx < len(self.marker_circles):
                self.marker_circles[nearest_idx].remove()
                self.marker_circles.pop(nearest_idx)

            # Refresh the plot
            self.refresh_plot()
            print(f"Removed marker at ({removed_marker['x']:.1f}, {removed_marker['y']:.1f})")

    def refresh_plot(self):
        """Refresh the entire plot with current markers"""
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title("Click on spores to mark them\nRight-click to remove nearest marker")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.marker_circles = []

        # Redraw all markers
        for i, marker in enumerate(self.markers):
            circle = Circle((marker['x'], marker['y']), radius=self.RADIUS, fill=False,
                            edgecolor='red', linewidth=2, alpha=0.8)
            self.ax.add_patch(circle)
            self.marker_circles.append(circle)

            # Add number label
            self.ax.text(marker['x'] + 7, marker['y'] - 7, str(i + 1),
                         color='red', fontsize=8, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.2",
                                   facecolor='white', alpha=0.8))

        self.fig.canvas.draw()

    def get_markers(self):
        """Return list of all markers"""
        return self.markers

    def save_markers(self, filename="spore_markers.json"):
        """Save markers to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.markers, f, indent=2)
        print(f"Markers saved to {filename}")

    def load_markers(self, filename):
        """Load markers from a JSON file"""
        if not os.path.exists(filename):
            print(f"No existing marker file found at {filename}. Starting fresh.")
            return
        try:
            with open(filename, 'r') as f:
                self.markers = json.load(f)
            self.refresh_plot()
            print(f"Markers loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found")

    def show_statistics(self):
        """Print statistics about marked spores"""
        count = len(self.markers)
        print(f"\nSpore Marking Statistics:")
        print(f"Total spores marked: {count}")

        if count > 0:
            x_coords = [m['x'] for m in self.markers]
            y_coords = [m['y'] for m in self.markers]

            print(f"X coordinate range: {min(x_coords):.1f} - {max(x_coords):.1f}")
            print(f"Y coordinate range: {min(y_coords):.1f} - {max(y_coords):.1f}")

            # Calculate approximate density (spores per 1000 pixels²)
            if hasattr(self, 'image') and self.image is not None:
                area = self.image.shape[0] * self.image.shape[1]
                density = (count / area) * 1000
                print(f"Approximate density: {density:.2f} spores per 1000 pixels²")

    def show(self):
        """Display the interactive plot"""
        plt.show()


# Example usage
#def main():
#
#    parser = argparse.ArgumentParser(description='Interactive spore marking tool for microscopy images')
#    parser.add_argument('--image_path', required=True, help='Path to the microscopy image')
#    args = parser.parse_args()
#
#    json_output_path = os.path.splitext(args.image_path)[0] + '.json'
#
#    try:
#        # Create the marking tool
#        tool = SporeMarkingTool(args.image_path, json_output_path)
#
#        tool.show()
#
#        tool.show_statistics()
#
#        tool.save_markers(json_output_path)
#
#        # Print all marker coordinates
#        markers = tool.get_markers()
#        if markers:
#            print("\nAll marked spore coordinates:")
#            for i, marker in enumerate(markers):
#                print(f"Spore {i + 1}: ({marker['x']:.1f}, {marker['y']:.1f})")
#
#    except Exception as e:
#        print(f"Error: {e}")
#

import tkinter as tk
from tkinter import filedialog
import os

def main():
    # Open a Tkinter file dialog to choose an image
    root = tk.Tk()
    root.withdraw()  # hide the main tkinter window

    image_path = filedialog.askopenfilename(
        title="Select a microscopy image",
        filetypes=[
            ("Image files", ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")),
            ("All files", "*.*")
        ]
    )

    if not image_path:
        print("No file selected. Exiting.")
        return

    # JSON will be saved next to the image
    json_output_path = os.path.splitext(image_path)[0] + '.json'

    try:
        # Create the marking tool
        tool = SporeMarkingTool(image_path, json_output_path)

        # Show the interactive matplotlib window
        tool.show()

        # After window closes, show statistics and save
        tool.show_statistics()
        tool.save_markers(json_output_path)

        # Print all marker coordinates
        markers = tool.get_markers()
        if markers:
            print("\nAll marked spore coordinates:")
            for i, marker in enumerate(markers):
                print(f"Spore {i + 1}: ({marker['x']:.1f}, {marker['y']:.1f})")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
