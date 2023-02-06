import sys
from PIL import Image
import string
from collections import Counter
from itertools import tee, count
from pytesseract import pytesseract, Output
import pandas as pd
import cv2
import numpy as np
from transformers import DetrFeatureExtractor
from transformers import TableTransformerForObjectDetection
import torch
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def PIL_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv_to_PIL(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def pytess(cell_pil_img):
    return ' '.join(pytesseract.image_to_data(cell_pil_img, output_type=Output.DICT, config='-c tessedit_char_blacklist=œ˜â€œï¬â™Ã©œ¢!|”?«“¥ --psm 6 preserve_interword_spaces')['text']).strip()


def sharpen_image(pil_img):

    img = PIL_to_cv(pil_img)
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    pil_img = cv_to_PIL(sharpen)
    return pil_img


def uniquify(seq, suffs=count(1)):
    """Make all the items unique by adding a suffix (1, 2, etc).
    Credit: https://stackoverflow.com/questions/30650474/python-rename-duplicates-in-list-with-progressive-numbers-without-sorting-list
    `seq` is mutable sequence of strings.
    `suffs` is an optional alternative suffix iterable.
    """
    not_unique = [k for k, v in Counter(seq).items() if v > 1]

    suff_gens = dict(zip(not_unique, tee(suffs, len(not_unique))))
    for idx, s in enumerate(seq):
        try:
            suffix = str(next(suff_gens[s]))
        except KeyError:
            continue
        else:
            seq[idx] += suffix

    return seq


def binarizeBlur_image(pil_img):
    image = PIL_to_cv(pil_img)
    thresh = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY_INV)[1]

    result = cv2.GaussianBlur(thresh, (5, 5), 0)
    result = 255 - result
    return cv_to_PIL(result)


def td_postprocess(pil_img):
    '''
    Removes gray background from tables
    '''
    img = PIL_to_cv(pil_img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 100), (255, 5, 255)
                       )  # (0, 0, 100), (255, 5, 255)
    # (0, 0, 5), (255, 255, 255))
    nzmask = cv2.inRange(hsv, (0, 0, 5), (255, 255, 255))
    nzmask = cv2.erode(nzmask, np.ones((3, 3)))  # (3,3)
    mask = mask & nzmask

    new_img = img.copy()
    new_img[np.where(mask)] = 255

    return cv_to_PIL(new_img)


def table_detector(image, THRESHOLD_PROBA):
    '''
    Table detection using DEtect-object TRansformer pre-trained on 1 million tables
    '''

    feature_extractor = DetrFeatureExtractor(
        do_resize=True, size=800, max_size=800)
    encoding = feature_extractor(image, return_tensors="pt")

    model = TableTransformerForObjectDetection.from_pretrained(
        "./models/table-transformer-detection")

    with torch.no_grad():
        outputs = model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(
        outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    return (model, probas[keep], bboxes_scaled)


def table_struct_recog(image, THRESHOLD_PROBA):
    '''
    Table structure recognition using DEtect-object TRansformer pre-trained on 1 million tables
    '''

    feature_extractor = DetrFeatureExtractor(
        do_resize=True, size=1000, max_size=1000)
    encoding = feature_extractor(image, return_tensors="pt")

    model = TableTransformerForObjectDetection.from_pretrained(
        "./models/table-transformer-structure-recognition")
    with torch.no_grad():
        outputs = model(**encoding)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > THRESHOLD_PROBA

    target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(
        outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]

    return (model, probas[keep], bboxes_scaled)


class TableExtractionPipeline():

    colors = ["red", "blue", "green", "yellow", "orange", "violet"]

    def add_padding(self, pil_img, top, right, bottom, left, color=(255, 255, 255)):
        '''
        Image padding as part of TSR pre-processing to prevent missing table edges
        '''
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result

    def crop_tables(self, pil_img, prob, boxes, delta_xmin, delta_ymin, delta_xmax, delta_ymax):
        '''
        crop_tables and plot_results_detection must have same co-ord shifts because 1 only plots the other one updates co-ordinates 
        '''
        cropped_img_list = []

        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

            xmin, ymin, xmax, ymax = xmin-delta_xmin, ymin - \
                delta_ymin, xmax+delta_xmax, ymax+delta_ymax
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cropped_img_list.append(cropped_img)

        return cropped_img_list

    def generate_structure(self, model, pil_img, prob, boxes, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
        '''
        Co-ordinates are adjusted here by 3 'pixels'
        To plot table pillow image and the TSR bounding boxes on the table
        '''

        rows = {}
        cols = {}
        idx = 0
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

            xmin, ymin, xmax, ymax = xmin, ymin, xmax, ymax
            cl = p.argmax()
            class_text = model.config.id2label[cl.item()]
            if class_text == 'table row':
                rows['table row.'+str(idx)] = (xmin, ymin-expand_rowcol_bbox_top,
                                               xmax, ymax+expand_rowcol_bbox_bottom)
            if class_text == 'table column':
                cols['table column.'+str(idx)] = (
                    xmin, ymin-expand_rowcol_bbox_top, xmax, ymax+expand_rowcol_bbox_bottom)
            idx += 1

        return rows, cols

    def sort_table_featuresv2(self, rows: dict, cols: dict):
        rows_ = {table_feature: (xmin, ymin, xmax, ymax) for table_feature, (
            xmin, ymin, xmax, ymax) in sorted(rows.items(), key=lambda tup: tup[1][1])}
        cols_ = {table_feature: (xmin, ymin, xmax, ymax) for table_feature, (
            xmin, ymin, xmax, ymax) in sorted(cols.items(), key=lambda tup: tup[1][0])}

        return rows_, cols_

    def individual_table_featuresv2(self, pil_img, rows: dict, cols: dict):

        for k, v in rows.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            rows[k] = xmin, ymin, xmax, ymax, cropped_img

        for k, v in cols.items():
            xmin, ymin, xmax, ymax = v
            cropped_img = pil_img.crop((xmin, ymin, xmax, ymax))
            cols[k] = xmin, ymin, xmax, ymax, cropped_img

        return rows, cols

    def object_to_cellsv2(self, master_row: dict, cols: dict, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left):
        '''Removes redundant bbox for rows&columns and divides each row into cells from columns
        Args:
        Returns:

        '''
        cells_img = {}
        row_idx = 0
        new_cols = {}
        new_master_row = {}
        new_cols = cols
        new_master_row = master_row
        for k_row, v_row in new_master_row.items():

            _, _, _, _, row_img = v_row
            xmax, ymax = row_img.size
            xa, ya, xb, yb = 0, 0, 0, ymax
            row_img_list = []
            for idx, kv in enumerate(new_cols.items()):
                k_col, v_col = kv
                xmin_col, _, xmax_col, _, col_img = v_col
                xmin_col, xmax_col = xmin_col - padd_left - 10, xmax_col - padd_left
                xa = xmin_col
                xb = xmax_col
                if idx == 0:
                    xa = 0
                if idx == len(new_cols)-1:
                    xb = xmax
                xa, ya, xb, yb = xa, ya, xb, yb

                row_img_cropped = row_img.crop((xa, ya, xb, yb))
                row_img_list.append(row_img_cropped)

            cells_img[k_row+'.'+str(row_idx)] = row_img_list
            row_idx += 1

        return cells_img, len(new_cols), len(new_master_row)-1

    def clean_dataframe(self, df):
        '''
        Remove irrelevant symbols that appear with tesseractOCR
        '''
        for col in df.columns:
            df[col] = df[col].str.replace("'", '', regex=True)
            df[col] = df[col].str.replace('"', '', regex=True)
            df[col] = df[col].str.replace(']', '', regex=True)
            df[col] = df[col].str.replace('[', '', regex=True)
            df[col] = df[col].str.replace('{', '', regex=True)
            df[col] = df[col].str.replace('}', '', regex=True)
            df[col] = df[col].str.replace('-', '', regex=True)
            df[col] = df[col].str.replace('|', '', regex=True)
            df[col] = df[col].str.replace('_', '', regex=True)
            df[col] = df[col].str.replace('_', '', regex=True)

        return df

    def convert_df(self, df):
        return df.to_csv().encode('utf-8')

    def create_dataframe(self, cells_pytess_result: list, max_cols: int, max_rows: int, image_path):
        '''Create dataframe using list of cell values of the table, also checks for valid header of dataframe
        Args:
            cells_pytess_result: list of strings, each element representing a cell in a table
            max_cols, max_rows: number of columns and rows
        Returns:
            dataframe : final dataframe after all pre-processing 
        '''

        headers = cells_pytess_result[:max_cols]
        new_headers = uniquify(
            headers, (f' {x!s}' for x in string.ascii_lowercase))
        counter = 0

        cells_list = cells_pytess_result[max_cols:]
        df = pd.DataFrame("", index=range(0, max_rows), columns=new_headers)

        cell_idx = 0
        for nrows in range(max_rows):
            for ncols in range(max_cols):
                df.iat[nrows, ncols] = str(cells_list[cell_idx])
                cell_idx += 1
        for x, col in zip(string.ascii_lowercase, new_headers):
            if f' {x!s}' == col:
                counter += 1
        df = self.clean_dataframe(df)
        image_path = image_path.split(".")[0] + ".csv"
        # print(image_path)
        df.to_csv(image_path)
        return df

    def start_process(self, image_path: str, TD_THRESHOLD, TSR_THRESHOLD, padd_top, padd_left, padd_bottom, padd_right, delta_xmin, delta_ymin, delta_xmax, delta_ymax, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom):
        '''
        Initiates process of generating pandas dataframes from raw pdf-page images
        '''
        image = Image.open(image_path).convert("RGB")
        model, probas, bboxes_scaled = table_detector(
            image, THRESHOLD_PROBA=TD_THRESHOLD)

        if bboxes_scaled.nelement() == 0:
            print('No table found in the pdf-page image')
            return ''
        cropped_img_list = self.crop_tables(
            image, probas, bboxes_scaled, delta_xmin, delta_ymin, delta_xmax, delta_ymax)
        for unpadded_table in cropped_img_list:
            table = self.add_padding(
                unpadded_table, padd_top, padd_right, padd_bottom, padd_left)
            model, probas, bboxes_scaled = table_struct_recog(
                table, THRESHOLD_PROBA=TSR_THRESHOLD)
            rows, cols = self.generate_structure(
                model, table, probas, bboxes_scaled, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom)
            rows, cols = self.sort_table_featuresv2(rows, cols)
            master_row, cols = self.individual_table_featuresv2(
                table, rows, cols)
            cells_img, max_cols, max_rows = self.object_to_cellsv2(
                master_row, cols, expand_rowcol_bbox_top, expand_rowcol_bbox_bottom, padd_left)
            sequential_cell_img_list = []
            for k, img_list in cells_img.items():
                for img in img_list:
                    sequential_cell_img_list.append(pytess(img))
            cells_pytess_result = sequential_cell_img_list
            return self.create_dataframe(cells_pytess_result, max_cols, max_rows, image_path)


if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: python table-extractor.py <image_path> <table_detection_treshold> <table_detection_treshold> <padding_top> <padding_left> <padding_bottom> <padding_right>")
        exit()
    elif (len(sys.argv) >= 2):
        img_name = sys.argv[1]
        TableDetection_threshold = 0.6
        TableRecognition_threshold = 0.8
        padd_top = padd_left = padd_right = padd_bottom = 20

    elif (len(sys.argv) == 4):
        img_name = sys.argv[1]
        TableDetection_threshold = TableRecognition_threshold = float(
            sys.argv[2])
        padd_top = padd_left = padd_right = padd_bottom = int(sys.argv[3])

    elif (len(sys.argv) > 4 and len(sys.argv) < 8):
        print("Usage: python table-extractor.py <image_path> <table_detection_treshold> <table_detection_treshold> <padding_top> <padding_left> <padding_bottom> <padding_right>")
        exit()

    else:
        # 0.0 to 1.0 default 0.6
        TableDetection_threshold = float(sys.argv[2])
        # 0.0 to 1.0 default 0.8
        TableRecognition_threshold = float(sys.argv[3])

        # 0 to 200 default 20
        padd_top = int(sys.argv[4])
        # 0 to 200 default 20
        padd_left = int(sys.argv[5])
        # 0 to 200 default 20
        padd_right = int(sys.argv[6])
        # 0 to 200 default 20
        padd_bottom = int(sys.argv[7])

    te = TableExtractionPipeline()
    if img_name is not None:
        print(te.start_process(img_name, TD_THRESHOLD=TableDetection_threshold, TSR_THRESHOLD=TableRecognition_threshold, padd_top=padd_top, padd_left=padd_left, padd_bottom=padd_bottom,
              padd_right=padd_right, delta_xmin=0, delta_ymin=0, delta_xmax=0, delta_ymax=0, expand_rowcol_bbox_top=0, expand_rowcol_bbox_bottom=0))
