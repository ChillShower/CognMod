import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import csv
import random

class ImageReviewer:
    def __init__(self, root):
        random.seed(25)
        self.root = root
        self.root.title("Image Reviewer")

        # Empty array of reviewed images
        self.reviewed = []

        # Image display
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Grading slider
        self.grade_slider = tk.Scale(root, from_=0, to=5, orient=tk.HORIZONTAL,
                                     label="Grade")
        self.grade_slider.pack()

        # Buttons
        self.keep_button = tk.Button(root, text="Keep", command=self.keep_image)
        self.keep_button.pack(side=tk.LEFT, padx=5)

        self.discard_button = tk.Button(root, text="Discard", command=self.discard_image)
        self.discard_button.pack(side=tk.LEFT, padx=5)


        # Select image folder
        self.folder = filedialog.askdirectory(title="Select Image Folder")
        self.images = [os.path.join(self.folder, f) for f in os.listdir(self.folder)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.index = -1

        # Prepare CSV file
        self.csv_file = os.path.join(self.folder, "image_reviews.csv")
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["image", "decision", "grade"])
        random.shuffle(self.images)
        self.load_next_image()

    def load_next_image(self):
        if not self.images:
            self.image_label.config(text="No images found.")
            return
        self.index += 1

        if self.index > 10 & len(set(self.reviewed)) == 10:
            self.image_label.config(text="All images reviewed.")
            self.root.destroy()
            return
        
        self.img_path = random.choice(self.images)
        #img_path = self.images[self.index]
        
        if self.img_path not in self.reviewed:
            self.reviewed.append(self.img_path)

        img = Image.open(self.img_path)
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_img)
        self.grade_slider.set(5)  # reset slider to middle value

    def save_result(self, decision):
        #img_path = random.choice(self.images)
        #img_path = self.images[self.index]
        grade = self.grade_slider.get()
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([os.path.basename(self.img_path), decision, grade])
        print(f"{decision}: {self.img_path}, grade={grade}")

    def keep_image(self):
        self.save_result("keep")
        self.load_next_image()

    def discard_image(self):
        self.save_result("discard")
        self.load_next_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageReviewer(root)
    root.mainloop()
