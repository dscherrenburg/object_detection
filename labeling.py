import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ImageLabelingTool:
    def __init__(self, image_dir, label_dir, dataset, shift_jump=20, max_interpolation=200):
        self.image_dir = os.path.join(image_dir, dataset)
        self.label_dir = os.path.join(label_dir, dataset)
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.num_images = len(self.image_files)
        self.hand_labeled_file = os.path.join(self.label_dir, "hand_labeled.txt")

        self.shift_jump = shift_jump
        self.max_interpolation = max_interpolation
        self.class_id = 0
        self.hand_labeled = self._load_hand_labeled()

        self._setup_root()
        self._ask_starting_point()

    def _ask_starting_point(self):
        """Pop-up window to ask where the user wants to start."""

        def _set_starting_point(index):
            if index is None:
                print("All images are labeled.")
                return
            self.current_index = index
            print(f"Starting from image {self.current_index + 1}")
            win.destroy()
            self._init_plot()

        win = tk.Toplevel()
        win.attributes('-topmost', True)
        win.geometry("500x300")
        win.title("Starting Point")
        tk.Label(win, text="Where would you like to start?").pack(pady=20)
        tk.Button(win, text="First Image", command=lambda: _set_starting_point(0)).pack(pady=10)
        tk.Button(win, text="Last Image", command=lambda: _set_starting_point(self.num_images - 1)).pack(pady=10)
        tk.Button(win, text="First Unlabeled Image", command=lambda: _set_starting_point(self._find_unlabeled_image("first"))).pack(pady=10)
        tk.Button(win, text="Last Unlabeled Image", command=lambda: _set_starting_point(self._find_unlabeled_image("last"))).pack(pady=10)
        win.mainloop()

    def _find_unlabeled_image(self, target="first"):
        """Find the first, last, next, or previous unlabeled image."""
        if target == "first":
            image_files = self.image_files
            start_index = 0
        elif target == "last":
            image_files = reversed(self.image_files)
            start_index = self.num_images - 1
        elif target == "next":
            image_files = self.image_files[self.current_index + 1:]
            start_index = self.current_index + 1
        elif target == "previous":
            image_files = reversed(self.image_files[:self.current_index])
            start_index = self.current_index - 1
        else:
            raise ValueError("Invalid target specified. Use 'first', 'last', 'next', or 'previous'.")
        for idx, image_file in enumerate(image_files):
            label_file = os.path.join(self.label_dir, image_file.rsplit('.', 1)[0] + '.txt')
            if not os.path.exists(label_file):
                return start_index + idx if target in ["first", "next"] else start_index - idx
        print(f"All images are labeled, starting from the {target} image.")
        return start_index

    def _load_labels(self, index):
        """Load YOLO-format labels (class_id, x_center, y_center, width, height)."""
        label_file = os.path.join(self.label_dir, self.image_files[index].rsplit('.', 1)[0] + '.txt')
        if not os.path.exists(label_file):
            return []
        with open(label_file, 'r') as f:
            return [(int(c), float(x), float(y), float(w), float(h)) for c, x, y, w, h in (line.strip().split() for line in f)]

    def _load_hand_labeled(self):
        if os.path.exists(self.hand_labeled_file):
            with open(self.hand_labeled_file, 'r') as f:
                return [int(line.strip()) for line in f.readlines()]
        return []

    def _setup_root(self):
        self.root = tk.Tk()
        self.root.title("Image Labeling Tool")        
        self.root.attributes('-topmost', False)
        self.root.attributes('-fullscreen', True)
        self.root.bind("<Escape>", self._quit)

        # Create a title frame at the top of the window
        title_frame = tk.Frame(self.root, bg="lightgrey", height=20)
        title_frame.pack(side=tk.TOP, fill=tk.X)
        spacer = tk.Label(title_frame, text=" " * 10, bg="lightgrey")
        spacer.pack(side=tk.LEFT, padx=10, pady=5)
        title_label = tk.Label(title_frame, text="Image Labeling Tool", bg="lightgrey", font=("Arial", 14))
        title_label.pack(side=tk.LEFT, padx=10, pady=5, expand=True)

        # Add Help Button next to the title, aligned to the right
        help_button = tk.Button(title_frame, text="Help", command=self._show_help)
        help_button.pack(side=tk.RIGHT, padx=10, pady=5)

    def _show_help(self):
        """Display a popup with an explanation of all key functions."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Help - Key Functions")
        help_window.geometry("650x500")

        help_text = (
            "Key Functions:\n\n"
            "Navigation:\n"
            " - Arrow Keys or Mouse Wheel: Navigate through images\n"
            "   - Hold Shift: Increase navigation speed\n"
            "   - Hold Alt: Jump to the next/previous unlabeled image\n"
            " - Home: Go to the first image\n"
            " - End: Go to the last image\n\n"
            "Annotation:\n"
            " - Click and drag to create a bounding box\n"
            " - i: Interpolate labels between current and adjacent hand-labeled images\n"
            " - d: Delete the last annotation in the current image\n"
            " - Ctrl + Delete: Remove all annotations\n\n"
            "Zoom and View:\n"
            " - Ctrl + Mouse Wheel: Zoom in/out\n"
            " - Ctrl + Drag: Pan the image\n"
            " - r: Reset view to original zoom level\n\n"
            "Exit:\n"
            " - q or Escape: Quit the application"
        )
        # Add scrollable text area
        help_label = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        help_label.insert(tk.END, help_text)
        help_label.config(state=tk.DISABLED)  # Make it read-only
        help_label.pack(expand=True, fill=tk.BOTH)
        # Add a Close button
        close_button = tk.Button(help_window, text="Close", command=help_window.destroy)
        close_button.pack(pady=10)
        
    def _init_plot(self):
        self.dragging = False
        self.panning = False
        self.editing = False
        self.start_point = None
        self.temp_rect = None
        self.cross_lines = []
        self.selected_box = None
        img = cv2.imread(os.path.join(self.image_dir, self.image_files[self.current_index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_size = (img.shape[1], img.shape[0])
        self._reset_view()

        self.fig, self.ax = plt.subplots(figsize=(20, 16))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.08)
    
        # Slider for navigating images
        ax_slider = plt.axes([0.175, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, '', 1, self.num_images, valinit=self.current_index+1, valstep=1, initcolor='none')
        self.slider.valtext.set_fontsize(10)
        self.slider.valtext.set_position((0.5, 0))
        self.slider.valtext.set_verticalalignment('center')
        self.slider.valtext.set_horizontalalignment('center')
        self.slider.on_changed(self._on_slider_change)

        # Enable blitting
        self.canvas = self.fig.canvas
        self.background = None  # To store the background
        self._connect_events()
        self.show_image(self.current_index)

        # Embed the plot in Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.root.mainloop()  # Start Tkinter main loop

    def _update_background(self):
        """Update the background to enable efficient redrawing."""
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def show_image(self, index):
        img = cv2.imread(os.path.join(self.image_dir, self.image_files[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_size = (img.shape[1], img.shape[0])

        self.ax.clear()
        # self.ax.set_xticks([])
        # self.ax.set_yticks([])
        self.ax.imshow(img, extent=[0, self.img_size[0], self.img_size[1], 0])
        self.slider.valtext.set_text(f"{self.current_index+1} / {self.num_images}")

        # Draw existing annotations
        labels = self._load_labels(index)
        for label in labels:
            class_id, x_center, y_center, w_norm, h_norm = label
            x = (x_center - w_norm / 2) * self.img_size[0]
            y = (y_center - h_norm / 2) * self.img_size[1]
            w = w_norm * self.img_size[0]
            h = h_norm * self.img_size[1]
            self.ax.add_patch(Rectangle((x, y), w, h, edgecolor='red', facecolor='none', linewidth=self.img_size[0] // 1000))
        
        if self.selected_box is not None:
            class_id, x_center, y_center, w_norm, h_norm = labels[self.selected_box]
            x = (x_center - w_norm / 2) * self.img_size[0]
            y = (y_center - h_norm / 2) * self.img_size[1]
            w = w_norm * self.img_size[0]
            h = h_norm * self.img_size[1]
            self.ax.add_patch(Rectangle((x, y), w, h, edgecolor='blue', facecolor='none', linewidth=self.img_size[0] // 1000))
            self.ax.plot(x, y, 'bo', markersize=5)
            self.ax.plot(x + w, y, 'bo', markersize=5)
            self.ax.plot(x, y + h, 'bo', markersize=5)
            self.ax.plot(x + w, y + h, 'bo', markersize=5)

        # Set the axes limits and zoom state
        self.ax.set_xlim(self.zoom_xlim)
        self.ax.set_ylim(self.zoom_ylim)

        self.canvas.draw()
    
    def _reset_editor(self):
        self.dragging = False
        self.panning = False
        self.editing = False
        self.selected_box = None
        self.start_point = None
        self.temp_rect = None

    def _on_key(self, event):
        key_mapping = {
            'right': lambda: self._navigate_images(1),
            'left': lambda: self._navigate_images(-1),
            'shift+right': lambda: self._navigate_images(self.shift_jump),
            'shift+left': lambda: self._navigate_images(-self.shift_jump),
            'alt+right': lambda: self._navigate_images(self._find_unlabeled_image("next") - self.current_index),
            'alt+left': lambda: self._navigate_images(self._find_unlabeled_image("previous") - self.current_index),
            'home': lambda: self._navigate_images(-self.current_index),
            'end': lambda: self._navigate_images(self.num_images - self.current_index - 1),

            'i': lambda: self._interpolate_labels(),
            'r': lambda: self._reset_view(),
            'd': lambda: self._remove_annotations(self.current_index, -1),
            'delete': lambda: self._remove_annotations(self.current_index,-1),
            'ctrl+delete': lambda: self._remove_annotations('all'),
            'q': lambda: self._quit(),
            'escape': lambda: self._quit(),
        }
        if event.key in key_mapping:
            key_mapping[event.key]()
        else:
            return
        self.show_image(self.current_index)

    def _navigate_images(self, step):
        new_index = (self.current_index + step) % self.num_images
        if new_index != self.current_index:
            self.current_index = new_index
            self._reset_editor()
            self.slider.set_val(self.current_index + 1)

    def _on_slider_change(self, val):
        self._navigate_images(int(val) - self.current_index - 1)
        self.show_image(self.current_index)

    def _remove_annotations(self, index=None, box_index=None, edit=True):
        """Remove annotations for a specific image index."""
        if index == "all":
            confirm = messagebox.askyesno("Confirm Deletion", "Are you sure you want to remove all annotations?")
            if confirm:
                for file in os.listdir(self.label_dir):
                    os.remove(os.path.join(self.label_dir, file))
                self.hand_labeled = []
                self._reset_editor()
                print("All annotations removed.")
            else:
                print("Annotations not removed.")
        elif box_index is not None:
            if self.selected_box is not None:
                box_index = self.selected_box
            labels = self._load_labels(index)
            if box_index < len(labels) and len(labels) > 0:
                labels.pop(box_index)
                self._save_labels(index, labels)
                if edit:
                    print(f"Annotation removed in image {index + 1}")
                    self.selected_box = None
                    self.editing = False
                if len(labels) == 0:
                    if index in self.hand_labeled:
                        self.hand_labeled.remove(index)
                        self._save_hand_labeled()
            else:
                print(f"No annotation found in image {index + 1}")
        else:        
            if index is None:
                index = self.current_index
            label_file = os.path.join(self.label_dir, self.image_files[index].rsplit('.', 1)[0] + '.txt')
            if os.path.exists(label_file):
                os.remove(label_file)
                if index in self.hand_labeled:
                    self.hand_labeled.remove(index)
                    self._save_hand_labeled()
                print(f"Annotations removed for image {index + 1}")
            else:
                print(f"No annotations found for image {index + 1}")

    def _reset_view(self):
        x_margin = self.img_size[0] * 0.05
        y_margin = self.img_size[1] * 0.05
        self.zoom_xlim = (-x_margin, self.img_size[0] + x_margin)
        self.zoom_ylim = (self.img_size[1] + y_margin, -y_margin)

    def _clip_coordinates(self, x, y, w, h):
        img_width, img_height = self.img_size
        x = max(0, min(x, img_width))
        y = max(0, min(y, img_height))
        w = min(w, img_width - x)
        h = min(h, img_height - y)
        return x, y, w, h

    def _on_mouse_press(self, event):
        if event.inaxes == self.ax:
            if event.button == 1 and event.xdata and event.ydata:
                if event.key == 'control':  # Enable panning when holding Ctrl
                    self.panning = True
                    self.start_point = (event.xdata, event.ydata)
                else:  # Normal click for annotation
                    if self.editing:
                        self.start_point = self._initialize_editing(event.xdata, event.ydata)
                        if self.start_point is None:
                            self.selected_box = None
                            self.editing = False
                            return
                        self._remove_annotations(self.current_index, self.selected_box, edit=False)
                        self.selected_box = None
                        self.show_image(self.current_index)
                    else:
                        self.start_point = self._clip_coordinates(event.xdata, event.ydata, 0, 0)[:2]
                    self.dragging = True
                    w, h = event.xdata - self.start_point[0], event.ydata - self.start_point[1]
                    self.temp_rect = Rectangle(self.start_point, w, h, edgecolor='blue', facecolor='none')
                    self.ax.add_patch(self.temp_rect)
                plt.draw()
        
    def _on_mouse_motion(self, event):
        if event.inaxes != self.ax:  # Ignore motion outside the axes
            return
        
        if self.background is not None:
            self.canvas.restore_region(self.background)

        # Remove previous cross lines
        while self.cross_lines:
            line = self.cross_lines.pop()
            line.remove()

        if event.xdata and event.ydata:
            # Draw new cross lines
            h_line = self.ax.axhline(event.ydata, linestyle='dashed', color='white', linewidth=self.img_size[0] // 1000)
            v_line = self.ax.axvline(event.xdata, linestyle='dashed', color='white', linewidth=self.img_size[0] // 1000)
            self.cross_lines.extend([h_line, v_line])

        if self.dragging and event.xdata and event.ydata:
            x1, y1 = self.start_point
            x2, y2 = self._clip_coordinates(event.xdata, event.ydata, 0, 0)[:2]
            if not self.temp_rect:
                self.temp_rect = Rectangle((x1, y1), 0, 0, edgecolor='blue', facecolor='none')
                self.ax.add_patch(self.temp_rect)
            self.temp_rect.set_width(abs(x2 - x1))
            self.temp_rect.set_height(abs(y2 - y1))
            self.temp_rect.set_xy((min(x1, x2), min(y1, y2)))
        elif self.panning and event.xdata and event.ydata:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            dx = self.start_point[0] - event.xdata
            dy = self.start_point[1] - event.ydata
            self.ax.set_xlim(xlim[0] + dx, xlim[1] + dx)
            self.ax.set_ylim(ylim[0] + dy, ylim[1] + dy)
            self.zoom_xlim = self.ax.get_xlim()
            self.zoom_ylim = self.ax.get_ylim()
            
        # Redraw only the updated elements
        self.ax.draw_artist(self.ax)
        self.canvas.blit(self.ax.bbox)

    def _on_mouse_release(self, event):
        if self.dragging and event.xdata and event.ydata:
            x1, y1 = self.start_point
            x2, y2 = self._clip_coordinates(event.xdata, event.ydata, 0, 0)[:2]
            x, y, w, h = min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)
            if w > 3 and h > 3:
                x, y, w, h = self._clip_coordinates(x, y, w, h)
                labels = self._load_labels(self.current_index)
                # Convert to YOLO format (normalize values)
                x_center = (x + w / 2) / self.img_size[0]
                y_center = (y + h / 2) / self.img_size[1]
                w_norm = w / self.img_size[0]
                h_norm = h / self.img_size[1]
                labels.append((self.class_id, x_center, y_center, w_norm, h_norm))
                if self.editing:
                    print(f"Modified annotation in image {self.current_index + 1}")
                    self.editing = False
                else:
                    print(f"Added annotation to image {self.current_index + 1}")
                self.temp_rect.remove()
                self._save_labels(self.current_index, labels)
                self._save_hand_labeled(self.current_index)
            else:
                self.temp_rect.remove()
                self.editing = False
                self.selected_box = self._select_box(event.xdata, event.ydata)
                if self.selected_box is not None:
                    self.editing = True
        self.dragging = False
        self.panning = False
        self.show_image(self.current_index)

    def _on_scroll(self, event):
        if event.key == 'control':  # Ctrl + Mouse Wheel for zoom
            scale_factor = 1.1 if event.step < 0 else 0.9
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            mouse_x, mouse_y = event.xdata, event.ydata
            self.ax.set_xlim([mouse_x + (x - mouse_x) * scale_factor for x in xlim])
            self.ax.set_ylim([mouse_y + (y - mouse_y) * scale_factor for y in ylim])
            self.zoom_xlim = self.ax.get_xlim()
            self.zoom_ylim = self.ax.get_ylim()
            plt.draw()
        elif event.key == 'alt':  # Alt + Mouse Wheel to jump to the next/previous unlabeled image
            step = 1 if event.step > 0 else -1
            target_index = self._find_unlabeled_image("next" if step > 0 else "previous")
            self._navigate_images(target_index - self.current_index)
        else:  # Default Mouse Wheel for navigation
            step = self.shift_jump if event.key == 'shift' else 1
            self._navigate_images(step if event.step > 0 else -step)
        self.show_image(self.current_index)
    
    def _select_box(self, x, y):
        """Select a bounding box by clicking on it. If x, y is inside multiple boxes, take the box of which the center is closest to the x, y."""
        labels = self._load_labels(self.current_index)
        x, y = x/self.img_size[0], y/self.img_size[1]
        closest_box = None
        min_distance = float('inf')
        for i, (class_id, x_center, y_center, w, h) in enumerate(labels):
            if x_center - w / 2 <= x <= x_center + w / 2 and y_center - h / 2 <= y <= y_center + h / 2:
                distance = np.sqrt((x_center - x) ** 2 + (y_center - y) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_box = i
        return closest_box
    
    def _initialize_editing(self, x, y, margin=0.05):
        """Identify the corner of the bounding box that is being edited and return the opposite corner."""
        labels = self._load_labels(self.current_index)
        class_id, x_box, y_box, w, h = labels[self.selected_box]
        x_box, y_box, w, h = x_box * self.img_size[0], y_box * self.img_size[1], w * self.img_size[0], h * self.img_size[1]
        corners = [
            (x_box - w / 2, y_box - h / 2, x_box + w / 2, y_box + h / 2),
            (x_box - w / 2, y_box + h / 2, x_box + w / 2, y_box - h / 2),
            (x_box + w / 2, y_box - h / 2, x_box - w / 2, y_box + h / 2),
            (x_box + w / 2, y_box + h / 2, x_box - w / 2, y_box - h / 2)
        ]
        closest_corner = None
        margin = (self.zoom_xlim[1] - self.zoom_xlim[0]) * margin
        min_distance = float('inf')
        for x1, y1, x2, y2 in corners:
            if x1 - margin <= x <= x1 + margin and y1 - margin <= y <= y1 + margin:
                distance = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_corner = (x2, y2)
        return closest_corner

    def _interpolate_labels(self):
        """Interpolate labels between the current image and adjacent hand-labeled images."""
        self._reset_editor()
        prev_index = next((i for i in reversed(self.hand_labeled) if i < self.current_index), None)
        next_index = next((i for i in self.hand_labeled if i > self.current_index), None)
        t_current = int(self.image_files[self.current_index].split('.')[0])

        if prev_index is None and next_index is None:
            print("No previous or next hand-labeled image to interpolate with.")
            return

        def match_objects(prev_labels, curr_labels):
            """Match objects based on class ID and spatial proximity."""
            matched = []
            used_indices = set()
            for prev_obj in prev_labels:
                best_match = None
                min_dist = float('inf')
                for i, curr_obj in enumerate(curr_labels):
                    if i in used_indices:
                        continue
                    if prev_obj[0] == curr_obj[0]:  # Match class ID
                        dist = ((prev_obj[1] - curr_obj[1]) ** 2 + (prev_obj[2] - curr_obj[2]) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            best_match = i
                if best_match is not None:
                    used_indices.add(best_match)
                    matched.append((prev_obj, curr_labels[best_match]))
            return matched

        def interpolate_between(indices, t_start, t_end, labels_start, labels_end):
            for i in indices:
                t_i = int(self.image_files[i].split('.')[0])
                t = (t_i - t_start) / (t_end - t_start)
                interpolated_labels = []

                matched_objects = match_objects(labels_start, labels_end)
                for (class_id, x1, y1, w1, h1), (_, x2, y2, w2, h2) in matched_objects:
                    x_interp = (1 - t) * x1 + t * x2
                    y_interp = (1 - t) * y1 + t * y2
                    w_interp = (1 - t) * w1 + t * w2
                    h_interp = (1 - t) * h1 + t * h2
                    interpolated_labels.append((class_id, x_interp, y_interp, w_interp, h_interp))

                self._save_labels(i, interpolated_labels)

        if prev_index is not None:
            if abs(self.current_index - prev_index) <= self.max_interpolation:
                t_prev = int(self.image_files[prev_index].split('.')[0])
                prev_labels = self._load_labels(prev_index)
                curr_labels = self._load_labels(self.current_index)
                if prev_labels and curr_labels:
                    interpolate_between(range(prev_index + 1, self.current_index), t_prev, t_current, prev_labels, curr_labels)
            else:
                print(f"Skipping interpolation with image {prev_index + 1} due to large distance.")

        if next_index is not None:
            if abs(next_index - self.current_index) <= self.max_interpolation:
                t_next = int(self.image_files[next_index].split('.')[0])
                next_labels = self._load_labels(next_index)
                curr_labels = self._load_labels(self.current_index)
                if curr_labels and next_labels:
                    interpolate_between(range(self.current_index + 1, next_index), t_current, t_next, curr_labels, next_labels)
            else:
                print(f"Skipping interpolation with image {next_index + 1} due to large distance.")

        print(f"Interpolated labels between images {f'{prev_index+1} and {self.current_index+1}' if prev_index is not None else ''}{' & ' if prev_index is not None and next_index is not None else ''}{f'{self.current_index+1} and {next_index+1}' if next_index is not None else ''}")


    def _save_labels(self, index, labels=None):
        """Save labels for a specific image."""
        label_file = os.path.join(self.label_dir, self.image_files[index].rsplit('.', 1)[0] + '.txt')
        with open(label_file, 'w') as f:
            for class_id, x, y, w, h in labels:
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    def _save_hand_labeled(self, index=None):
        """Save hand-labeled images to a text file."""
        if index is not None:
            self.hand_labeled = sorted(list(set(self.hand_labeled + [index])))
        with open(self.hand_labeled_file, 'w') as f:
            f.write('\n'.join(str(i) for i in self.hand_labeled))
    
    def _quit(self, event=None):
        finished = messagebox.askyesnocancel("Exit", "Is the annotation finished?")        
        if finished:
            for i in range(self.num_images):
                label_file = os.path.join(self.label_dir, self.image_files[i].rsplit('.', 1)[0] + '.txt')
                if not os.path.exists(label_file):
                    with open(label_file, 'w') as f:
                        pass  # Create an empty label file
            print("Empty label files added for all remaining frames.")
        elif finished is None:
            return
        
        plt.close()     # Close the matplotlib window after Tkinter root is destroyed
        self.root.destroy()
        exit()


    def _connect_events(self):
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)

if __name__ == "__main__":
    IMAGE_DIR = "/home/daan/Data/dock_data/images/"
    LABEL_FILE = "/home/daan/Data/dock_data/labels/"
    DATASET = "rosbag2_2024_09_17-15_40_51"

    shift_jump = 5
    max_interpolation = 200

    ImageLabelingTool(IMAGE_DIR, LABEL_FILE, DATASET, shift_jump=shift_jump, max_interpolation=max_interpolation)