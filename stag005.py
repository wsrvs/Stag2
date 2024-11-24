import tkinter as tk
from tkinter import Tk, filedialog, Canvas, Scrollbar, messagebox, Frame, Label, Entry, Text, Button, Radiobutton, StringVar, BooleanVar, Checkbutton, END
from PIL import Image, ImageTk
import fitz  # PyMuPDF
import pytesseract
import easyocr
import subprocess
import cv2
import sys
import numpy as np

# print("Используется Python из:", sys.executable)
# Импортируем класс MyClass из файла tmp778.py
# from tmp778 import ImageTransformer, ImageProcessor

# Указываем путь к Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class DocumentProcessorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Документный процессор")

        # Создаем рамку (Frame) для группы выбора файла/каталога и восстановления перспективы
        select_recovery_frame = Frame(master, borderwidth=2, relief="groove")
        select_recovery_frame.pack(pady=10, padx=10, fill="x")  # Размещаем рамку в главном окне

        # Создаем кнопку для выбора файла или каталога
        self.select_button = Button(
            select_recovery_frame,  # Указываем, что кнопка будет находиться в нашей созданной рамке
            text="Выбери файл (pdf, jpg, png) или каталог",  # Текст на кнопке
            command=self.select_file  # Привязываем команду к функции выбора файла
        )
        self.select_button.pack(pady=5)  # Размещаем кнопку в рамке с вертикальным отступом

        # Создаем кнопку для восстановления перспективы
        self.recovery_button = Button(
            select_recovery_frame,  # Указываем, что кнопка будет находиться в нашей созданной рамке
            text="Восстановление перспективы",  # Текст на кнопке
            command=self.run_recovery_program  # Привязываем команду к функции, которая запускает программу
        )
        self.recovery_button.pack(pady=5)  # Размещаем кнопку в рамке с вертикальным отступом


        # Группа для детектирования объектов
        detection_frame = Frame(master, borderwidth=2, relief="groove")
        detection_frame.pack(pady=5, padx=10, fill="x")

        Label(detection_frame, text="Детектирование объектов").pack(pady=5)

        self.object_detection_var = StringVar()
        self.yolo5_button = Radiobutton(detection_frame, text="YOLOv5", variable=self.object_detection_var,
                                        value="YOLOv5")
        self.yolo8_button = Radiobutton(detection_frame, text="YOLOv8", variable=self.object_detection_var,
                                        value="YOLOv8")
        self.yolo5_button.pack(anchor="w")
        self.yolo8_button.pack(anchor="w")

        # Группа для распознавания текста
        recognition_frame = Frame(master, borderwidth=2, relief="groove")
        recognition_frame.pack(pady=5, padx=10, fill="x")

        Label(recognition_frame, text="Распознавание текста").pack(pady=5)

        self.text_recognition_var = StringVar()
        self.tesseract_button = Radiobutton(recognition_frame, text="Tesseract OCR", variable=self.text_recognition_var,
                                            value="Tesseract")
        self.easyocr_button = Radiobutton(recognition_frame, text="EasyOCR", variable=self.text_recognition_var,
                                          value="EasyOCR")
        self.tesseract_button.pack(anchor="w")
        self.easyocr_button.pack(anchor="w")

        # Путь сохранения
        self.save_path_label = Label(master, text="Путь сохранения:")
        self.save_path_label.pack(pady=5)
        self.save_path_entry = Entry(master)
        self.save_path_entry.pack(pady=5)

        self.browse_button = Button(master, text="Выбрать каталог", command=self.browse_directory)
        self.browse_button.pack(pady=5)

        # Кнопка запуска распознавания
        self.start_button = Button(master, text="Запуск распознавания", command=self.start_recognition)
        self.start_button.pack(pady=10)

        # Предпросмотр документа

        self.preview_frame = Frame(master, width=400, height=300)
        self.preview_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.image_label = Label(self.preview_frame)
        self.image_label.pack()

        # Вывод сообщений
        self.output_text = Text(master, wrap=tk.WORD, height=10, width=50)
        self.output_text.pack(side=tk.RIGHT, padx=10, pady=10)

    def run_recovery_program(self):
        # Запуск процесса восстановления, используя ImageProcessor
        try:
            # Создаем экземпляр ImageProcessor и запускаем его
            processor = ImageProcessor()
            processor.run()  # Вызываем метод обработки изображений
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")

    def select_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf"), ("Image files", "*.jpg;*.png")])
        if not self.file_path:
            self.file_path = filedialog.askdirectory()

        if self.file_path:
            self.preview_document()

    def preview_document(self):
        try:
            if self.file_path.endswith('.pdf'):
                doc = fitz.open(self.file_path)
                page = doc.load_page(0)  # Загружаем первую страницу
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img.thumbnail((400, 300))  # Изменяем размер для предпросмотра
                self.image = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.image)
                self.output_text.insert(tk.END, "Предпросмотр PDF загружен.\n")
            else:
                img = Image.open(self.file_path)
                img.thumbnail((400, 300))  # Изменяем размер для предпросмотра
                self.image = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.image)
                self.output_text.insert(tk.END, "Предпросмотр изображения загружен.\n")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить документ: {e}")

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.save_path_entry.delete(0, tk.END)  # Очистить предыдущее значение
            self.save_path_entry.insert(0, directory)  # Вставить новый путь

    def start_recognition(self):
        if self.file_path:
            self.output_text.insert(tk.END, "Запуск распознавания...\n")
            if self.text_recognition_var.get() == "Tesseract":
                self.output_text.insert(tk.END, "Выбран Tesseract OCR.\n")
                self.recognize_text_tesseract()
            elif self.text_recognition_var.get() == "EasyOCR":
                self.output_text.insert(tk.END, "Выбран EasyOCR.\n")
                self.recognize_text_easyocr()
            else:
                self.output_text.insert(tk.END, "Распознавание текста не выбрано.\n")
        else:
            messagebox.showwarning("Предупреждение", "Не выбран файл или каталог.")

    def recognize_text_tesseract(self):
        try:
            if self.file_path.endswith('.pdf'):
                doc = fitz.open(self.file_path)
                text = ""
                for page in doc:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text += pytesseract.image_to_string(img) + "\n"
                self.output_text.insert(tk.END, text)
            else:
                img = Image.open(self.file_path)
                text = pytesseract.image_to_string(img)
                self.output_text.insert(tk.END, text)
        except Exception as e:
            self.output_text.insert(tk.END, f"Ошибка при распознавании текста: {e}")

    def recognize_text_easyocr(self):
        try:
            reader = easyocr.Reader(['en'])
            if self.file_path.endswith('.pdf'):
                doc = fitz.open(self.file_path)
                text = ""

                for page in doc:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    result = reader.readtext(img)
                    for (bbox, text_recog, prob) in result:
                        text += f"{text_recog} (Confidence: {prob})\n"
                self.output_text.insert(tk.END, text)
            else:
                img = Image.open(self.file_path)
                result = reader.readtext(img)
                text = ""
                for (bbox, text_recog, prob) in result:
                    text += f"{text_recog} (Confidence: {prob})\n"
                self.output_text.insert(tk.END, text)
        except Exception as e:
            self.output_text.insert(tk.END, f"Ошибка при распознавании текста: {e}")

class ImageTransformer:
    def __init__(self, image):
        """
        Инициализация объекта ImageTransformer.

        Параметры:
        image: Исходное изображение, которое требуется скорректировать для перспективы.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Необходимо передать действительное изображение.")

        self.original_image = image  # Хранит оригинальное изображение
        self.points_src = []  # Список для хранения координат выбранных точек

    def set_points(self, points):
        """
        Устанавливает точки, которые будут использоваться для коррекции перспективы.

        Параметры:
        points: Список из 4 точек, выбранных пользователем.
        """
        if len(points) != 4:
            raise ValueError("Должно быть ровно 4 точки для коррекции перспективы.")
        self.points_src = points

    def correct_perspective(self):
        """
        Корректирует перспективу исходного изображения на основе заданных точек.

        Возвращает:
        corrected_image: Изображение с исправленной перспективой.

        Исключения:
        ValueError: Если задано меньше 4 точек.
        """
        if len(self.points_src) < 4:
            raise ValueError("Требуется 4 точки для выполнения изменения перспективы.")

        # Получаем размеры оригинального изображения
        orig_height, orig_width = self.original_image.shape[:2]

        # Определяем целевые точки для исправленного изображения
        points_dst = np.array([[0, 0],
                               [orig_width, 0],
                               [orig_width, orig_height],
                               [0, orig_height]], dtype='float32')

        # Вычисляем матрицу преобразования перспективы
        matrix = cv2.getPerspectiveTransform(np.array(self.points_src, dtype='float32'), points_dst)

        # Применяем трансформацию к оригинальному изображению
        corrected_image = cv2.warpPerspective(self.original_image, matrix, (orig_width, orig_height))
        return corrected_image  # Возвращаем скорректированное изображение


class ImageProcessor:
    def __init__(self):
        self.points_src = []  # Список для хранения выбранных точек
        self.original_image = None  # Хранение оригинального изображения
        self.display_image = None  # Хранение изображения для отображения
        self.img_tk = None  # Хранение ссылки на изображение Tkinter
        self.scale_factor = 1.0  # Коэффициент масштаба
        self.current_point = None  # Текущая активная точка для добавления
        self.root = Tk()
        self.create_interface()

    def log_message(self, message):
        self.message_area.insert(END, message + '\n')
        self.message_area.see(END)

    def create_interface(self):
        self.root.title("Image Processor")
        self.root.minsize(800, 600)

        self.frame = Frame(self.root)
        self.frame.pack(fill="both", expand=True)

        self.canvas = Canvas(self.frame)
        self.scroll_y = Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        self.scroll_x = Scrollbar(self.frame, orient="horizontal", command=self.canvas.xview)

        self.scroll_x.pack(side="bottom", fill="x")
        self.scroll_y.pack(side="right", fill="y")

        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        self.message_area = Text(self.root, height=5, wrap='word')
        self.message_area.pack(side="bottom", fill="x")

        button_frame = Frame(self.root)
        button_frame.pack(side="bottom", fill="x")

        button = Button(button_frame, text="Выбрать изображение", command=self.select_image_file)
        button.pack(side="left", padx=5, pady=5)

        button_correct = Button(button_frame, text="Исправить изображение", command=self.correct_image)
        button_correct.pack(side="left", padx=5, pady=5)

        button_save = Button(button_frame, text="Сохранить", command=self.save_image)
        button_save.pack(side="left", padx=5, pady=5)

        button_exit = Button(button_frame, text="Закрыть", command=self.root.quit)
        button_exit.pack(side="left", padx=5, pady=5)

    def select_image_file(self):
        self.log_message("Выбор файла...")
        filename = filedialog.askopenfilename(title='Выберите изображение',
                                              filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if filename:
            self.log_message(f"Выбран файл: {filename}")
            self.load_and_show_image(filename)

    def load_and_show_image(self, image_path):
        self.original_image = cv2.imread(image_path)

        if self.original_image is None:
            self.log_message(f"Не удалось загрузить изображение по пути: {image_path}.")
            return

        # Конвертируем изображение BGR (OpenCV) в RGB (PIL)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        # Сохраняем оригинальный размер изображения
        self.orig_height, self.orig_width = self.original_image.shape[:2]

        # Масштабируем изображение под размеры канваса
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # Вычисляем коэффициент масштаба
        self.scale_factor = min(canvas_width / self.orig_width, canvas_height / self.orig_height)

        new_width = int(self.orig_width * self.scale_factor)
        new_height = int(self.orig_height * self.scale_factor)

        # Изменяем размер изображения для отображения
        self.display_image = cv2.resize(self.original_image, (new_width, new_height))

        # Создаем объект PhotoImage
        self.img_tk = ImageTk.PhotoImage(Image.fromarray(self.display_image))

        # Удаляем старые элементы и добавляем новое изображение
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.img_tk, anchor="nw")  # Добавление нового изображения

        # Сохраняем ссылку на изображение, чтобы предотвратить сборку мусора
        self.canvas.image = self.img_tk  # Содержим ссылку на изображение

        # Привязываем клик мыши к функции
        self.canvas.bind("<Button-1>", self.locate_point)

    def locate_point(self, event):
        if self.original_image is not None:
            x = self.canvas.canvasx(event.x) / self.scale_factor
            y = self.canvas.canvasy(event.y) / self.scale_factor

            # Проверка границ
            if x < 0 or x >= self.orig_width or y < 0 or y >= self.orig_height:
                self.log_message("Координаты вне границ изображения.")
                return

            self.current_point = (int(x), int(y))

            # Проверяем, добавляется ли текущая точка
            if len(self.points_src) < 4:
                self.points_src.append(self.current_point)
                self.log_message(
                    f"Добавлена точка {len(self.points_src)}: {self.current_point}. Количество точек: {len(self.points_src)}")

                # Показываем точки на изображении
                self.show_point_on_image()  # Обновляем отображение

                # После добавления предыдущей точки можно сбросить current_point
                self.current_point = None  # Reset current_point после добавления
            else:
                self.log_message("Максимальное количество точек уже добавлено.")

    def show_point_on_image(self):
        img_with_points = self.display_image.copy()  # Используем текущее отображаемое изображение

        # Рисуем все предыдущие точки, если они существуют
        for point in self.points_src:
            cv2.circle(img_with_points, (int(point[0] * self.scale_factor),
                                         int(point[1] * self.scale_factor)), 5, (0, 255, 0), -1)  # Зеленая точка

        # Проверяем, установлена ли текущая точка
        if self.current_point is not None:
            cv2.circle(img_with_points, (int(self.current_point[0] * self.scale_factor),
                                         int(self.current_point[1] * self.scale_factor)), 5, (255, 0, 0),
                       -1)  # Красная точка

        self.img_tk = ImageTk.PhotoImage(Image.fromarray(img_with_points))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.img_tk, anchor="nw")

    def confirm_point(self):
        if self.current_point and len(self.points_src) < 4:
            self.points_src.append(self.current_point)
            self.log_message(
                f"Добавлена точка {len(self.points_src)}: {self.current_point}. Количество точек: {len(self.points_src)}")

            # Показываем предыдущие точки на изображении
            self.show_point_on_image()  # Обновляем отображение

            # После добавления предыдущей точки можно сбросить current_point
            self.current_point = None

            # Проверяем, нужно ли включить кнопку подтверждения
            if len(self.points_src) == 4:
                self.confirm_button['state'] = 'normal'  # Включаем кнопку подтверждения
        else:
            self.log_message("Точка не может быть добавлена.")

    def confirm_all_points(self):
        """Подтвердить все добавленные точки."""
        if len(self.points_src) == 4:
            self.log_message("Все точки подтверждены.")
            self.confirm_button['state'] = 'disabled'  # Отключаем кнопку
        else:
            self.log_message("Необходимы 4 точки для подтверждения.")

    def correct_image(self):

        """Коррекция изображения."""
        if len(self.points_src) < 4:
            self.log_message("Требуется добавить 4 точки для выполнения изменения перспективы.")
            return

        # Создаем экземпляр ImageTransformer
        transformer = ImageTransformer(self.original_image)
        transformer.set_points(self.points_src)
        try:
            corrected_image = transformer.correct_perspective()
        except ValueError as e:
            self.log_message(str(e))
            return

        # Изменяем размер исправленного изображения для отображения с тем же масштабом
        new_display_width = int(self.orig_width * self.scale_factor)
        new_display_height = int(self.orig_height * self.scale_factor)

        self.display_image = cv2.resize(corrected_image, (new_display_width, new_display_height))

        # Обновляем изображение на канвасе
        self.img_tk = ImageTk.PhotoImage(Image.fromarray(self.display_image))
        self.canvas.delete("all")  # Удаляем старое изображение
        self.canvas.create_image(0, 0, image=self.img_tk, anchor="nw")

        # Обновляем оригинальное изображение и сбрасываем точки
        self.original_image = corrected_image
        self.points_src.clear()  # Очищаем точки после коррекции
        self.log_message("Изображение исправлено.")

    def save_image(self):
        if self.original_image is None:
            self.log_message("Изображение не исправлено. Сначала исправьте изображение.")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png"),
                                                           ("JPEG files", "*.jpg;*.jpeg"),
                                                           ("All files", "*.*")])
        if filename:
            # Преобразуем изображение в формате RGB в BGR для сохранения
            corrected_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, corrected_image)
            self.log_message(f"Изображение сохранено как: {filename}")

    def run(self):
        self.root.mainloop()  # Запуск основного цикла программы


if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentProcessorApp(root)
    root.mainloop()
