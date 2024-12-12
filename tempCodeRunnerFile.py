img_rgb = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        img_pil = PILImage.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.config(image=img_tk)
        self.canvas.image = img_tk