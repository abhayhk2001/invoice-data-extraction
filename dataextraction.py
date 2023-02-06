
from pdf2image import convert_from_path
import os
import pytesseract
from PIL import Image
import pandas as pd


class text_extract:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text_write_path = os.path.join('ocr_results', 'results.txt')
    csv_write_path = os.path.join('ocr_results', 'fields.csv')
    table_write_path = os.path.join('ocr_results', 'table.csv')
    fields = []

    def retrieve_text(self, field_name, image_path):
        try:
            img = Image.open(image_path)
            tessdata_dir_config = r'--tessdata-dir "./ocr-layers" --psm 6'
            img_text = pytesseract.image_to_string(
                img,  lang='eng_layer', config=tessdata_dir_config)
            print(field_name + ':' + img_text)

            self.fields.append([field_name,img_text])

            with open(self.text_write_path, 'a') as f:
                f.writelines(field_name + ':' + img_text)
                f.close()
        except Exception as e:
            # print(e)
            print(field_name + ' not found')
    
    def save_fields(self):
        df = pd.Dataframe(self.fields,columns=["Field", "Value"])
        df.to_csv(self.csv_write_path)
        
