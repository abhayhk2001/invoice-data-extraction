from regex import E
from pdf2image import convert_from_path
import os
import shutil


class preimgpdf:
    save_dir = 'runs/detect'
    poppler_path = 'poppler-22.04.0/Library/bin'
    temp_path = 'runs/temp'

    def __init__(self):
        print("--------------------")
        print("Clearing Directory")
        self.clear_directory()

    def clear_directory(self):
        if (os.path.exists(self.save_dir)):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        if (os.path.exists(self.temp_path)):
            shutil.rmtree(self.temp_path)
        os.makedirs(self.temp_path)
        print("Cleared Directories")

    def set_dir(self, invoice_path, i):
        if i == 1:
            app = ''
        else:
            app = str(i)
        invoice_name = os.path.split(invoice_path)[1]
        company_dir = os.path.join(
            'runs', 'detect', 'exp'+app, 'crops', 'COMPANY', invoice_name)
        invoice_date_dir = os.path.join(
            'runs', 'detect', 'exp'+app, 'crops', 'INVOICE DATE', invoice_name)
        table_dir = os.path.join(
            'runs', 'detect', 'exp'+app, 'crops', 'TABLE', invoice_name)
        total_dir = os.path.join(
            'runs', 'detect', 'exp'+app, 'crops', 'TOTAL', invoice_name)
        gst_dir = os.path.join('runs', 'detect', 'exp' +
                               app, 'crops', 'GST', invoice_name)
        abn_dir = os.path.join('runs', 'detect', 'exp' +
                               app, 'crops', 'ABN', invoice_name)
        account_dir = os.path.join(
            'runs', 'detect', 'exp'+app, 'crops', 'ACCOUNT_DETAILS', invoice_name)
        crop_img_paths = [{'field': 'company', 'path': company_dir}, {'field': 'invoice date', 'path': invoice_date_dir}, {'field': 'Total', 'path': total_dir}, {
            'field': 'gst', 'path': gst_dir}, {'field': 'abn', 'path': abn_dir}, {'field': 'Account Details', 'path': account_dir}]
        return crop_img_paths, table_dir

    def pdf_images(self, pdfpath):
        pages = convert_from_path(
            pdfpath, 700, poppler_path=self.poppler_path)
        for i in range(0, len(pages)):
            pages[i].save(os.path.join(self.temp_path, str(i)+".jpg"), 'JPEG')
        paths = []
        for image in os.scandir(self.temp_path):
            imgpath = image.path
            imgname = image.name
            paths.append(imgpath)
        return paths
