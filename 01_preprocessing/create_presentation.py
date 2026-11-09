import os
import sys
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
IMAGES_DIR = os.path.join(CURRENT_DIR, 'images')
OUTPUT_PPTX = os.path.join(CURRENT_DIR, 'presentation_stage1.pptx')


def create_presentation():
    prs = pptx.Presentation()
    # 16:9 widescreen format
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    COLOR_TITLE = RGBColor(29, 53, 87)       # #1d3557 (темно-синій)
    COLOR_BODY_BOLD = RGBColor(29, 53, 87)   # #1d3557
    COLOR_BODY_TEXT = RGBColor(40, 40, 40)   # темний графіт

    # ==========================================================
    # СЛАЙД 1: Картинка EDA 1 (Дисбаланс класів)
    # ==========================================================
    slide1 = prs.slides.add_slide(blank_layout)
    img1_path = os.path.join(IMAGES_DIR, '01_class_imbalance.png')
    if os.path.exists(img1_path):
        # 8:5 aspect ratio -> height 6.6 in, width 10.56 in -> centered
        slide1.shapes.add_picture(img1_path, Inches(1.386), Inches(0.45), width=Inches(10.56))
    
    notes1 = (
        "Слайд 1. Дисбаланс класів цільової змінної Subscription.\n"
        "У датасеті 45 211 клієнтів. Лише 11.7% (5 289 осіб) погодилися відкрити депозит, "
        "тоді як 88.3% відмовилися. Це сильний дисбаланс, тому точність (Accuracy) "
        "буде оманливою метрикою. Для навчання обов'язково використовувати PR-AUC, ROC-AUC, F1-score "
        "та стратифікований поділ."
    )
    slide1.notes_slide.notes_text_frame.text = notes1

    # ==========================================================
    # СЛАЙД 2: Картинка EDA 2 (Розподіли числових ознак)
    # ==========================================================
    slide2 = prs.slides.add_slide(blank_layout)
    img2_path = os.path.join(IMAGES_DIR, '02_numerical_distributions.png')
    if os.path.exists(img2_path):
        # 15:9.5 aspect ratio -> height 6.8 in, width 10.74 in -> centered
        slide2.shapes.add_picture(img2_path, Inches(1.3), Inches(0.35), width=Inches(10.74))

    notes2 = (
        "Слайд 2. Розподіли основних числових колонок (фокус на 97-99% вибірки).\n"
        "Вік має розподіл, близький до нормального (медіана 39 років, середнє 40.9).\n"
        "Натомість Баланс і Тривалість мають екстремальний правий скіл: медіана балансу 448 €, "
        "але багаті вкладники (до 102 тис. €) тягнуть середнє до 1 362 € (трикратний розрив!). "
        "У Campaign медіана 2 дзвінки (97.4% клієнтів мають до 10 дзвінків). "
        "Такий розрив математично доводить, що StandardScaler тут не підходить — потрібен RobustScaler."
    )
    slide2.notes_slide.notes_text_frame.text = notes2

    # ==========================================================
    # СЛАЙД 3: Картинка EDA 3 (Виявлення аномалій за 1.5 IQR)
    # ==========================================================
    slide3 = prs.slides.add_slide(blank_layout)
    img3_path = os.path.join(IMAGES_DIR, '03_outliers_iqr_boxplots.png')
    if os.path.exists(img3_path):
        slide3.shapes.add_picture(img3_path, Inches(1.3), Inches(0.35), width=Inches(10.74))

    notes3 = (
        "Слайд 3. Виявлення аномалій за правилом 1.5 IQR.\n"
        "За класичним правилом Тьюкі 1.5 IQR, аномаліями вважаються значення вище верхнього вуса: "
        "баланс понад 3 462 € (10.5% вибірки) та тривалість понад 643 с (7.2%).\n"
        "Проте аналіз показує: це не помилки введення даних, а реальні заможні клієнти та довгі успішні розмови. "
        "Видаляти їх категорично не можна, інакше модель втратить найцінніших клієнтів банку."
    )
    slide3.notes_slide.notes_text_frame.text = notes3

    # ==========================================================
    # СЛАЙД 4: Розбір скрипта — Крок 1: Підготовка та Спліт
    # ==========================================================
    slide4 = prs.slides.add_slide(blank_layout)
    
    # Title
    t_box4 = slide4.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.333), Inches(0.8))
    tf4 = t_box4.text_frame
    tf4.word_wrap = True
    p4 = tf4.paragraphs[0]
    p4.text = "Крок 1. Підготовка даних та захист від Data Leakage"
    p4.font.name = "Arial"
    p4.font.size = Pt(26)
    p4.font.bold = True
    p4.font.color.rgb = COLOR_TITLE

    # Body
    b_box4 = slide4.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(11.333), Inches(4.8))
    btf4 = b_box4.text_frame
    btf4.word_wrap = True

    points4 = [
        ("Виділення цільової змінної:", 
         "Кодування Subscription у бінарний вигляд: 0 — відмова, 1 — відкриття депозиту."),
        
        ("Feature Engineering (нова ознака):", 
         "Створено бінарну ознаку Previously_Contacted = (Pdays != -1) для збереження семантики первинних клієнтів."),
        
        ("Стратифікований поділ (80/20):", 
         "train_test_split(..., stratify=y) суворо фіксує співвідношення класів 88.3% / 11.7% у Train (36 168) та Test (9 043)."),
        
        ("Запобігання Data Leakage (вимога лекції):", 
         "Поділ виконується ДО будь-яких обчислень середніх, медіан, масштабування чи кодування категорій.")
    ]

    for title, desc in points4:
        bp = btf4.add_paragraph()
        bp.space_after = Pt(20)
        run1 = bp.add_run()
        run1.text = "•  " + title + " "
        run1.font.name = "Arial"
        run1.font.size = Pt(19)
        run1.font.bold = True
        run1.font.color.rgb = COLOR_BODY_BOLD
        
        run2 = bp.add_run()
        run2.text = desc
        run2.font.name = "Arial"
        run2.font.size = Pt(18)
        run2.font.bold = False
        run2.font.color.rgb = COLOR_BODY_TEXT

    notes4 = (
        "Слайд 4. Крок 1 нашого скрипта data_preprocessing.py.\n"
        "Спочатку кодуємо таргет у 0 та 1. Далі створюємо нову ознаку Previously_Contacted: "
        "у колонці pdays значення -1 означає, що клієнту раніше не дзвонили (81.7% вибірки). "
        "Найважливіше — стратифікований поділ 80 на 20 робиться ДО конвеєра, щоб уникнути витоку даних."
    )
    slide4.notes_slide.notes_text_frame.text = notes4

    # ==========================================================
    # СЛАЙД 5: Розбір скрипта — Крок 2: ColumnTransformer
    # ==========================================================
    slide5 = prs.slides.add_slide(blank_layout)
    
    t_box5 = slide5.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.333), Inches(0.8))
    tf5 = t_box5.text_frame
    tf5.word_wrap = True
    p5 = tf5.paragraphs[0]
    p5.text = "Крок 2. Архітектура конвеєра перетворень (ColumnTransformer)"
    p5.font.name = "Arial"
    p5.font.size = Pt(26)
    p5.font.bold = True
    p5.font.color.rgb = COLOR_TITLE

    b_box5 = slide5.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(11.333), Inches(4.8))
    btf5 = b_box5.text_frame
    btf5.word_wrap = True

    points5 = [
        ("Числові ознаки (8 колонок):", 
         "SimpleImputer(strategy='median') + RobustScaler(). Масштабування через медіану та IQR надійно захищає від впливу викидів без їх видалення."),
        
        ("Порядкова ознака (Education):", 
         "OrdinalEncoder зі збереженням логічного рангу освіти: unknown (-1) -> primary (0) -> secondary (1) -> tertiary (2)."),
        
        ("Номінальні ознаки (8 колонок: Job, Marital, Contact тощо):", 
         "OneHotEncoder(drop='first', sparse_output=False). Параметр drop='first' запобігає пастці фіктивних змінних (dummy variable trap)."),
        
        ("Модульність та безпека:", 
         "Всі перетворення об'єднані в єдиний ColumnTransformer, що гарантує відтворюваність та відсутність зсуву даних.")
    ]

    for title, desc in points5:
        bp = btf5.add_paragraph()
        bp.space_after = Pt(20)
        run1 = bp.add_run()
        run1.text = "•  " + title + " "
        run1.font.name = "Arial"
        run1.font.size = Pt(19)
        run1.font.bold = True
        run1.font.color.rgb = COLOR_BODY_BOLD
        
        run2 = bp.add_run()
        run2.text = desc
        run2.font.name = "Arial"
        run2.font.size = Pt(18)
        run2.font.bold = False
        run2.font.color.rgb = COLOR_BODY_TEXT

    notes5 = (
        "Слайд 5. Крок 2 нашого скрипта — конвеєр перетворень ColumnTransformer.\n"
        "Ми розділили всі 16 ознак на 3 окремі потоки: числові масштабуються RobustScaler (бо медіана та IQR нечутливі до викидів); "
        "освіта кодується OrdinalEncoder, оскільки тут є природна ієрархія; "
        "а решта категоріальних колонок кодується через OneHotEncoder з drop='first', щоб прибрати лінійну залежність."
    )
    slide5.notes_slide.notes_text_frame.text = notes5

    # ==========================================================
    # СЛАЙД 6: Розбір скрипта — Крок 3: Навчання та Результати
    # ==========================================================
    slide6 = prs.slides.add_slide(blank_layout)
    
    t_box6 = slide6.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.333), Inches(0.8))
    tf6 = t_box6.text_frame
    tf6.word_wrap = True
    p6 = tf6.paragraphs[0]
    p6.text = "Крок 3. Навчання конвеєра та підсумкові артефакти"
    p6.font.name = "Arial"
    p6.font.size = Pt(26)
    p6.font.bold = True
    p6.font.color.rgb = COLOR_TITLE

    b_box6 = slide6.shapes.add_textbox(Inches(1.0), Inches(1.85), Inches(11.333), Inches(4.8))
    btf6 = b_box6.text_frame
    btf6.word_wrap = True

    points6 = [
        ("Суворе правило fit/transform:", 
         "preprocessor.fit(X_train) розраховує всі параметри ТІЛЬКИ на Train. Тестова вибірка трансформується виключно через .transform()."),
        
        ("Розширений простір ознак:", 
         "Розмірність матриці ознак зросла з 16 до 41 інформативного стовпця. Жоден рядок не втрачено."),
        
        ("Формування артефактів для команди:", 
         "У папку processed_data/ збережено 4 чисті CSV-файли: X_train_preprocessed.csv, X_test_preprocessed.csv, y_train.csv, y_test.csv."),
        
        ("Готовність до Етапу 2 (Моделювання):", 
         "Колеги по команді можуть одразу завантажувати CSV-файли та навчати фінальні моделі (Logistic Regression, Random Forest, XGBoost).")
    ]

    for title, desc in points6:
        bp = btf6.add_paragraph()
        bp.space_after = Pt(20)
        run1 = bp.add_run()
        run1.text = "•  " + title + " "
        run1.font.name = "Arial"
        run1.font.size = Pt(19)
        run1.font.bold = True
        run1.font.color.rgb = COLOR_BODY_BOLD
        
        run2 = bp.add_run()
        run2.text = desc
        run2.font.name = "Arial"
        run2.font.size = Pt(18)
        run2.font.bold = False
        run2.font.color.rgb = COLOR_BODY_TEXT

    notes6 = (
        "Слайд 6. Фінальний крок скрипта та передача результатів колегам.\n"
        "Параметри масштабування та моди обчислюються суворо на навчальній вибірці (X_train), "
        "після чого тест просто трансформується. У результаті ми отримали 41 числову ознаку. "
        "Скрипт генерує 4 готові файли у папку processed_data/, завдяки чому колеги можуть миттєво приступати до етапу навчання моделей."
    )
    slide6.notes_slide.notes_text_frame.text = notes6

    # Збереження
    prs.save(OUTPUT_PPTX)
    print(f"Презентацію на 6 слайдів успішно збережено: {OUTPUT_PPTX}")
    return OUTPUT_PPTX


if __name__ == '__main__':
    create_presentation()
