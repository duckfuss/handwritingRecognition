import tkinter as tk
import random
import time
# Constants for chart dimensions
CANVAS_WIDTH = 750
CANVAS_HEIGHT = 300
BAR_WIDTH = 55
BAR_SPACING = 20
CHART_TOP_MARGIN = 30

def draw_bar_chart(canvas, data):
    max_value = max(data, key=lambda item: item[1])[1]

    # Calculate scaling factor to fit data within the canvas
    scale = (CANVAS_HEIGHT - CHART_TOP_MARGIN) / max_value

    x = BAR_SPACING

    for category, value in data:
        bar_height = int(value * scale)
        canvas.create_rectangle(
            x, CANVAS_HEIGHT - bar_height, x + BAR_WIDTH, CANVAS_HEIGHT,
            fill="blue", outline="black"
        )
        canvas.create_text(
            x + BAR_WIDTH // 2, CANVAS_HEIGHT - bar_height - 10, text=category
        )
        x += BAR_WIDTH + BAR_SPACING

root = tk.Tk()
root.title("neural network outuput")

canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg="white")
canvas.pack()

labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
data = [0.001, 0.01, 0.1, 0.9, 0.9, 0.005, 0.0, 0.0, 0.1, 1]
# Draw the bar chart on the canvas
def update(drawOnly = True, data=[]):
    if not drawOnly:
        canvas.delete("all")
        draw_bar_chart(canvas, list(zip(labels,data)))
    root.update()

