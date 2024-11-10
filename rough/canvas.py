import cv2
import tkinter as tk
from PIL import Image, ImageTk

class VideoStreamApp:
    def __init__(self, rtsp_url):
        self.root = tk.Tk()
        self.root.title("RTSP Video Stream")

        self.root.resizable(0, 0)
        self.root.overrideredirect(True)

        sw = self.root.winfo_screenwidth()
        # 得到屏幕宽度
        sh = self.root.winfo_screenheight()
        # 得到屏幕高度

        # 窗口宽高
        ww = 640
        wh = 480
        x = (sw - ww) / 2
        y = (sh - wh) / 2
        self.root.geometry("%dx%d+%d+%d" % (ww, wh, x, y))

        # 创建退出按键
        self.button = tk.Button(self.root, text='退出', command=self.root.quit)
        self.button.pack()
        
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        self.rtsp_url = rtsp_url
        self.cap = cv2.VideoCapture(rtsp_url)

        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert the frame to ImageTk format
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)

            # Update the canvas with the new frame
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.img = img_tk  # Keep a reference to avoid garbage collection

        # Call update after 10 ms
        self.root.after(10, self.update)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    rtsp_url = "..."  # the rtsp url of your device]
    app = VideoStreamApp(rtsp_url)
    app.run()