import pytesseract
from pdf2image import convert_from_bytes, convert_from_path
import numpy as np
from PIL import Image
import cv2
from spellchecker import SpellChecker
from sklearn.feature_extraction.text import CountVectorizer


pytesseract.pytesseract.tesseract_cmd = r'C:\Users\ELE60b4f26c29bc1\AppData\Local\Tesseract-OCR\tesseract.exe'

custom_config = r'--oem 3 --psm 6'


imgs = convert_from_path("impot-1.pdf", 500, use_pdftocairo=True, strict=False, poppler_path="poppler-0.68.0/bin")



def get_greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




img = Image.fromarray(get_greyscale(np.array(imgs[0])))

print(img)

result_txt = pytesseract.image_to_string(img, config=custom_config, lang='fra')


spell = SpellChecker(language='fr')

for mot in result_txt.split(' '):
    mot_corrige = spell.correction(mot)
    mot = mot_corrige



tmp = []
for i in result_txt.split(' '):
    if i.isalpha():
        tmp.append(i)


result_txt = ' '.join(tmp)

stop_word = open("stopwords.txt")
stop_word_list = stop_word.read()
stop_word.close()

tmp = []
for i in result_txt.split(' '):
    if i not in stop_word_list:
        tmp.append(i)

result_txt = ' '.join(tmp)

""""
categories = ['attestation_hebergement', 'impot', 'taxe_fonciere']
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
X = vectorizer.fit_transform(result_txt).toarray()

https://github.com/javedsha/text-classification/blob/master/Text%2BClassification%2Busing%2Bpython%2C%2Bscikit%2Band%2Bnltk.ipynb
https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a

Créer un X vectorier comme sur le site
Puis créer un Y en tableau numpy avec 0,1,2 pour impot, taxe, et hebergement
Fit le model
Puis faire un training pour voir le score
"""
