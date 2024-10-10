import cv2
import pytesseract
from PIL import Image, ImageTk
import re
from sympy import sympify, solve, Symbol, Eq, pretty
import tkinter as tk
from tkinter import filedialog, messagebox
import spacy
import requests
import numpy as np
import os

# ---- Ayarlar ----

# Tesseract OCR yolunu belirtin
# Windows için örnek yol:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Diğer işletim sistemlerinde genellikle PATH'e eklenmiş olabilir.
# Aşağıdaki satırı kendi sisteminize göre düzenleyin veya gerekirse yoruma alın.
# Örneğin:
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'

# Örnek için Windows yolu bırakıldı, kendi sisteminize göre düzenleyin
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows için örnek

# Wolfram Alpha API Ayarları
WOLFRAM_ALPHA_APP_ID = 'YOUR_WOLFRAM_ALPHA_APP_ID'  # Buraya kendi App ID'nizi girin

# spaCy dil modelleri
LANGUAGE = 'en'  # 'en' veya 'tr'

if LANGUAGE == 'en':
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
elif LANGUAGE == 'tr':
    try:
        nlp = spacy.load("tr_core_news_sm")
    except:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "tr_core_news_sm"])
        nlp = spacy.load("tr_core_news_sm")
else:
    raise ValueError("Desteklenmeyen dil seçildi.")

# ---- Fonksiyonlar ----

def preprocess_image(image_path):
    """
    Görüntüyü okuyup ön işleme tabi tutar:
    - Gri tonlamaya dönüştürme
    - Gürültü azaltma
    - Perspektif düzeltme
    - Kenar tespiti
    - Morfolojik işlemler
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Görüntü okunamadı. Lütfen geçerli bir dosya seçtiğinizden emin olun.")
    
    # Gri tonlamaya çevirme
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perspektif düzeltme (varsa)
    # Bu adım, eğilmiş veya bozulmuş görüntüler için gereklidir
    # Basit bir uygulama için burada geçiliyor, ihtiyaç halinde geliştirilebilir
    
    # Gürültü azaltma
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Kenar tespiti
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Morfolojik işlemler
    kernel = np.ones((3,3), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Eşikleme ile ikili görüntü oluşturma
    _, thresh = cv2.threshold(morphed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    preprocessed_image_path = 'preprocessed_image.png'
    cv2.imwrite(preprocessed_image_path, thresh)
    return preprocessed_image_path

def extract_text(image_path):
    """
    OCR kullanarak görüntüden metni çıkarır.
    """
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='eng')  # Dili ihtiyaca göre değiştirin
    return text

def parse_math_problem(text):
    """
    Metni analiz ederek matematiksel denklemi oluşturur.
    """
    doc = nlp(text)
    equation = ""
    for token in doc:
        if token.like_num:
            equation += token.text
        elif token.text in ['+', '-', '*', '/', '=', '(', ')', '^']:
            equation += token.text
        elif LANGUAGE == 'en' and token.text.lower() in ['plus', 'minus', 'multiplied', 'divided', 'equals', 'multiply', 'times', 'divide']:
            # İngilizce operatör çevirisi
            if token.text.lower() == 'plus':
                equation += '+'
            elif token.text.lower() == 'minus':
                equation += '-'
            elif token.text.lower() in ['multiplied', 'multiply', 'times']:
                equation += '*'
            elif token.text.lower() in ['divided', 'divide']:
                equation += '/'
            elif token.text.lower() == 'equals':
                equation += '='
        elif LANGUAGE == 'tr' and token.text.lower() in ['artı', 'eksi', 'çarpı', 'bölü', 'eşittir']:
            # Türkçe operatör çevirisi
            if token.text.lower() == 'artı':
                equation += '+'
            elif token.text.lower() == 'eksi':
                equation += '-'
            elif token.text.lower() in ['çarpı', 'carpi']:
                equation += '*'
            elif token.text.lower() in ['bölü', 'bolu']:
                equation += '/'
            elif token.text.lower() == 'eşittir':
                equation += '='
    
    # Regex ile gereksiz karakterleri temizleme
    equation = re.sub(r'[^0-9+\-*/=().x]', '', equation)
    
    # Değişken adı belirleme (varsayılan olarak 'x')
    if 'x' not in equation and 'X' not in equation:
        equation += '=0'  # Değişken ekleme
    
    return equation

def solve_equation(equation_str):
    """
    Matematiksel denklemi çözer. SymPy kullanarak çözüm bulmaya çalışır.
    Başarısız olursa Wolfram Alpha API ile çözüm denemesi yapar.
    """
    try:
        if '=' in equation_str:
            left, right = equation_str.split('=')
            eq = Eq(sympify(left), sympify(right))
            symbols = eq.free_symbols
            if len(symbols) == 1:
                var = symbols.pop()
                solution = solve(eq, var)
            else:
                solution = "Çözüm için birden fazla değişken tespit edildi."
        else:
            # Tek taraflı denklemler
            solution = sympify(equation_str).evalf()
        return solution
    except Exception as e:
        # Wolfram Alpha API kullanarak çözüm denemesi
        wolfram_solution = solve_with_wolfram(equation_str)
        return wolfram_solution if wolfram_solution else f"Çözümleme Hatası: {e}"

def solve_with_wolfram(equation_str):
    """
    Wolfram Alpha API kullanarak denklemi çözer.
    """
    try:
        if WOLFRAM_ALPHA_APP_ID == 'YOUR_WOLFRAM_ALPHA_APP_ID':
            return "Wolfram Alpha API anahtarı eklenmedi."
        url = "http://api.wolframalpha.com/v1/result"
        params = {
            'i': equation_str,
            'appid': WOLFRAM_ALPHA_APP_ID
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.text
        else:
            return "Wolfram Alpha çözüm bulunamadı."
    except Exception as e:
        return f"Wolfram Alpha Hatası: {e}"

def get_step_by_step_solution(equation_str):
    """
    SymPy kullanarak adım adım çözüm sağlar.
    """
    try:
        # SymPy'nin solve fonksiyonu adım adım çözüm sunmaz, ancak yapabiliriz
        # Daha gelişmiş adım adım çözümler için SymPy'nin 'manual' çözümlerini kullanabiliriz
        # Bu örnekte basit bir çözüm sunulmaktadır
        if '=' in equation_str:
            left, right = equation_str.split('=')
            eq = Eq(sympify(left), sympify(right))
            symbols = eq.free_symbols
            if len(symbols) == 1:
                var = symbols.pop()
                solution_steps = solve(eq, var, dict=True)
                steps = ""
                for step in solution_steps:
                    steps += f"{var} = {step[var]}\n"
                return steps
            else:
                return "Adım adım çözüm için birden fazla değişken tespit edildi."
        else:
            # Tek taraflı denklemler için adım adım çözüm
            expr = sympify(equation_str)
            steps = f"Sonuç: {expr.evalf()}"
            return steps
    except Exception as e:
        return f"Adım Adım Çözüm Hatası: {e}"

def display_solution(problem, solution, steps=None):
    """
    Çözümü kullanıcıya gösterir.
    """
    solution_text = f"Matematik Problemi:\n{problem}\n\nÇözüm:\n{solution}"
    if steps:
        solution_text += f"\n\nAdım Adım Çözüm:\n{steps}"
    messagebox.showinfo("Çözüm", solution_text)

def browse_image():
    """
    Kullanıcının fotoğraf seçmesini sağlar ve işlemleri başlatır.
    """
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
    )
    if file_path:
        try:
            # Görüntü Ön İşleme
            preprocessed_image = preprocess_image(file_path)
            
            # Metni Çıkarma
            extracted_text = extract_text(preprocessed_image)
            print("Çıkarılan Metin:")
            print(extracted_text)
            
            # Matematik Problemini Anlama
            math_equation = parse_math_problem(extracted_text)
            print("\nParçalanan Denklemler:")
            print(math_equation)
            
            # Denklemi Çözme
            solution = solve_equation(math_equation)
            print("\nÇözüm:")
            print(solution)
            
            # Adım Adım Çözüm
            steps = get_step_by_step_solution(math_equation)
            print("\nAdım Adım Çözüm:")
            print(steps)
            
            # Çözümü Gösterme
            display_solution(math_equation, solution, steps)
        except Exception as e:
            messagebox.showerror("Hata", str(e))

def main_gui():
    """
    Grafiksel Kullanıcı Arayüzünü oluşturur.
    """
    root = tk.Tk()
    root.title("Matematik Problemi Çözümleyici")
    root.geometry("500x300")
    root.resizable(False, False)

    # Başlık
    title = tk.Label(root, text="Matematik Problemi Çözümleyici", font=("Helvetica", 16, "bold"))
    title.pack(pady=20)

    # Açıklama
    description = tk.Label(root, text="Matematik probleminizi içeren fotoğrafı seçin:", font=("Helvetica", 12))
    description.pack(pady=10)

    # Fotoğraf Seçme Butonu
    browse_button = tk.Button(root, text="Fotoğraf Seç", command=browse_image, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), padx=20, pady=10)
    browse_button.pack(pady=10)

    # Desteklenen Formatlar
    instructions = tk.Label(root, text="Desteklenen formatlar: PNG, JPG, JPEG, BMP, TIFF", fg="gray", font=("Helvetica", 10))
    instructions.pack(pady=10)

    # Fotoğraf Önizleme Alanı
    preview_label = tk.Label(root, text="Fotoğraf Önizlemesi:", font=("Helvetica", 12))
    preview_label.pack(pady=5)

    preview_canvas = tk.Canvas(root, width=300, height=200, bg="white")
    preview_canvas.pack(pady=5)

    def update_preview(image_path):
        """
        Yüklenen fotoğrafın önizlemesini günceller.
        """
        try:
            img = Image.open(image_path)
            img.thumbnail((300, 200))
            img = ImageTk.PhotoImage(img)
            preview_canvas.image = img
            preview_canvas.create_image(150, 100, image=img)
        except Exception as e:
            print(f"Önizleme Hatası: {e}")

    def browse_image_with_preview():
        """
        Fotoğraf seçme ve önizlemeyi güncelleme.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")]
        )
        if file_path:
            update_preview(file_path)
            try:
                # Görüntü Ön İşleme
                preprocessed_image = preprocess_image(file_path)
                
                # Metni Çıkarma
                extracted_text = extract_text(preprocessed_image)
                print("Çıkarılan Metin:")
                print(extracted_text)
                
                # Matematik Problemini Anlama
                math_equation = parse_math_problem(extracted_text)
                print("\nParçalanan Denklemler:")
                print(math_equation)
                
                # Denklemi Çözme
                solution = solve_equation(math_equation)
                print("\nÇözüm:")
                print(solution)
                
                # Adım Adım Çözüm
                steps = get_step_by_step_solution(math_equation)
                print("\nAdım Adım Çözüm:")
                print(steps)
                
                # Çözümü Gösterme
                display_solution(math_equation, solution, steps)
            except Exception as e:
                messagebox.showerror("Hata", str(e))

    # Fotoğraf Seçme Butonunu Güncelleme
    browse_button.config(command=browse_image_with_preview)

    root.mainloop()

if __name__ == "__main__":
    main_gui()
