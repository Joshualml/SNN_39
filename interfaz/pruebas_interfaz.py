import tkinter as tk

root = tk.Tk()
root.config(bg="blue")

label1 = tk.Label(root, text = "Este es mi Label")
label1.pack()
label1.config(bg="green")

root.mainloop()