from pdf2image import convert_from_path
import os
import argparse

from preprocess import preimgpdf
from yolov import model
from dataextraction import text_extract

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


parser = argparse.ArgumentParser(description='run invoice detection')
parser.add_argument('--file', help='enter path of invoice')
parser.add_argument('--type', help='type of table')
args = parser.parse_args()


def main():
    preprocess = preimgpdf()
    detector = model('models/object-detection.pt')
    ocr = text_extract()
    invoice_path = args.file
    invoice = invoice_path
    # path to image
    img_supp_types = '.jpg' or '.png'

    if invoice.endswith(img_supp_types):
        detector.predict(invoice_path)
        crop_img_paths, table_dir = preprocess.set_dir(invoice, 1)
        file_path = crop_img_paths[0]['path'].split("\\")[:3]
        file_path = '\\'.join(file_path)
        for img in crop_img_paths:
            ocr.retrieve_text(img['field'], img['path'])
        table_path = table_dir
        if (os.path.exists(table_path)):
            print('TABLE DETAILS:')
            os.system(
                'python table-extraction\\table-extractor.py ' + table_path)
        else:
            print("no table detected")

        ocr.save_fields(file_path)

    elif invoice.endswith('.pdf'):
        images = preprocess.pdf_images(invoice)
        for i, imag in enumerate(images):
            detector.predict(imag)
            crop_img_paths, table_dir = preprocess.set_dir(imag, i+1)
            file_path = crop_img_paths[0]['path'].split("\\")[:3]
            file_path = '\\'.join(file_path)
            for img in crop_img_paths:
                ocr.retrieve_text(img['field'], img['path'])

            table_path = table_dir
            if (os.path.exists(table_path)):
                print('TABLE DETAILS:')
                os.system(
                    'python table-extraction\\table-extractor.py ' + table_path)
            else:
                print("no table detected")
            ocr.save_fields(file_path)


if __name__ == "__main__":
    main()

# !python3 table_transformer.py --table-type borderless -i "/content/open-intelligence-backend/datasets/all_tables/2.png"
