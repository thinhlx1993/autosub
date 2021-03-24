"""
Defines autosub's main functionality.
"""
from __future__ import absolute_import, print_function, unicode_literals

import time
import argparse
import cv2
import os
import sys
from datetime import datetime
try:
    from json.decoder import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from autosub_v2.constants import (
    LANGUAGE_CODES, GOOGLE_SPEECH_API_KEY, GOOGLE_SPEECH_API_URL,
)
from autosub_v2.formatters import FORMATTERS
from paddleocr import PaddleOCR


DEFAULT_SUBTITLE_FORMAT = 'srt'
DEFAULT_CONCURRENCY = 10
DEFAULT_SRC_LANGUAGE = 'en'
DEFAULT_DST_LANGUAGE = 'vi'


import six
from google.oauth2 import service_account
from google.cloud import translate_v2 as translate
from google.cloud import vision

credentials = service_account.Credentials.from_service_account_file(r"C:\autosub_models\key.json")
client = vision.ImageAnnotatorClient(credentials=credentials)
translate_client = translate.Client(credentials=credentials)
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def detect_texts_google_cloud(content):
    """Detects text in the file."""
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    predict_des = ""
    for text in texts:
        predict_des = text.description
        predict_des = predict_des.strip()
        return predict_des

    return predict_des


# def detect_texts(img_path, ocr):
#     """Detects text in the file."""
#     result = ocr.ocr(img_path, det=False, rec=True, cls=False)
#     for line in result:
#         # print(line[0])
#         if line[1] > 0.7:
#             return line[0]
#     return ""


def translate_text_google_cloud(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target)

    # print(u"Text: {}".format(result["input"]))
    print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
    return result["translatedText"]


def translate_text(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    return text


def generate_subtitles(
        source_path,
        output=None,
        dst_language=DEFAULT_DST_LANGUAGE,
        debug=False,
        cloud=False
    ):
    """
    Given an input audio/video file, generate subtitles in the specified language and format.
    """
    # Opens the Video file
    print("starting")
    if cloud:
        ocr = PaddleOCR(lang='ch', use_gpu=False,
                        rec_model_dir=r"C:\autosub_models\rec",
                        cls_model_dir=r"C:\autosub_models\cls",
                        det_model_dir=r"C:\autosub_models\det",
                        use_angle_cls=True,
                        rec_char_type='ch',
                        drop_score=0.8,
                        det_db_box_thresh=0.3,
                        cls=True)

    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1 / fps
    i = 0
    div_frame = 12
    sub_idx = 1
    list_srt = []
    old_des = ""
    prev_time = 0
    current_time = 0
    file_name = os.path.basename(source_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % div_frame == 0:
            prev_time_ts = datetime.utcfromtimestamp(prev_time).strftime('%H:%M:%S,%f')[:-4]
            current_time_ts = datetime.utcfromtimestamp(current_time).strftime('%H:%M:%S,%f')[:-4]
            h, w, c = frame.shape
            crop_img = frame[int(h * 0.94):h, 0:w]
            if debug:
                cv2.imshow('demo', crop_img)
                cv2.waitKey(1)

            description = ""
            if cloud:
                success, encoded_image = cv2.imencode('.jpg', crop_img)
                description = detect_texts_google_cloud(encoded_image)
            else:
                result = ocr.ocr(crop_img, det=False, rec=True, cls=False)
                for line in result:
                    # print(line[0])
                    if line[1] > 0.7:
                        description = line[0]
                        break

            if old_des != "" and (description != old_des or description == ""):
                list_srt.append({
                    "description": old_des,
                    "translate": translate_text_google_cloud(dst_language, old_des),
                    "first_time": prev_time_ts,
                    "last_time": current_time_ts,
                    "sub_idx": sub_idx
                })
                # with open(f"{os.path.splitext(file_name)[0]}_raw.srt", "a", encoding="utf-8") as myfile:
                #     myfile.write(f"{list_srt[-1]['sub_idx']}\n")
                #     myfile.write(f"{list_srt[-1]['first_time']} --> {list_srt[-1]['last_time']}\n")
                #     myfile.write(f"{list_srt[-1]['description']}\n")
                #     myfile.write('\n')
                #     myfile.close()

                with open(f"{os.path.splitext(file_name)[0]}.srt", "a", encoding="utf-8") as myfile_vi:
                    myfile_vi.write(f"{list_srt[-1]['sub_idx']}\n")
                    myfile_vi.write(f"{list_srt[-1]['first_time']} --> {list_srt[-1]['last_time']}\n")
                    myfile_vi.write(f"{list_srt[-1]['translate']}\n")
                    myfile_vi.write('\n')
                    myfile_vi.close()

                print(f"{list_srt[-1]['sub_idx']}\n")
                print(f"{list_srt[-1]['first_time']} --> {list_srt[-1]['last_time']}\n")
                print(f"{list_srt[-1]['description']}\n")
                print(f"{list_srt[-1]['translate']}\n")
                print('\n')

                sub_idx += 1
                prev_time = current_time

            if description == "":
                prev_time = current_time

            old_des = description
            current_time += time_per_frame * div_frame

        i += 1

    cap.release()
    return output


def validate(args):
    """
    Check that the CLI arguments passed to autosub are valid.
    """
    if args.format not in FORMATTERS:
        print(
            "Subtitle format not supported. "
            "Run with --list-formats to see all supported formats."
        )
        return False

    if args.src_language not in LANGUAGE_CODES.keys():
        print(
            "Source language not supported. "
            "Run with --list-languages to see all supported languages."
        )
        return False

    if args.dst_language not in LANGUAGE_CODES.keys():
        print(
            "Destination language not supported. "
            "Run with --list-languages to see all supported languages."
        )
        return False

    if not args.source_path:
        print("Error: You need to specify a source path.")
        return False

    return True


def main():
    """
    Run autosub as a command-line program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('source_path', help="Path to the video or audio file to subtitle",
                        nargs='?')
    parser.add_argument('-C', '--concurrency', help="Number of concurrent API requests to make",
                        type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument('-o', '--output',
                        help="Output path for subtitles (by default, subtitles are saved in \
                        the same directory and name as the source path)")
    parser.add_argument('-F', '--format', help="Destination subtitle format",
                        default=DEFAULT_SUBTITLE_FORMAT)
    parser.add_argument('-S', '--src-language', help="Language spoken in source file",
                        default=DEFAULT_SRC_LANGUAGE)
    parser.add_argument('-D', '--dst-language', help="Desired language for the subtitles",
                        default=DEFAULT_DST_LANGUAGE)
    parser.add_argument('-K', '--api-key',
                        help="The Google Translate API key to be used. \
                        (Required for subtitle translation)")
    parser.add_argument('--list-formats', help="List all available subtitle formats",
                        action='store_true')
    parser.add_argument('--list-languages', help="List all available source/destination languages",
                        action='store_true')

    parser.add_argument('--from', help="minimum height", type=float, default=0.9)

    parser.add_argument('--to', help="maximum height", type=float, default=1.0)

    parser.add_argument('--debug', help="Allows to show cropped image on the desktop", action='store_true')

    parser.add_argument('--cloud', help="Use google cloud compute to extract text", action='store_true')

    args = parser.parse_args()

    if args.list_formats:
        print("List of formats:")
        for subtitle_format in FORMATTERS:
            print("{format}".format(format=subtitle_format))
        return 0

    if args.list_languages:
        print("List of all languages:")
        for code, language in sorted(LANGUAGE_CODES.items()):
            print("{code}\t{language}".format(code=code, language=language))
        return 0

    if not validate(args):
        return 1

    try:
        st = time.time()
        subtitle_file_path = generate_subtitles(
            source_path=args.source_path,
            dst_language=args.dst_language,
            output=args.output,
            debug=args.debug,
            cloud=args.cloud
        )
        print("Subtitles file created at {} time consumer: {}".format(subtitle_file_path, time.time() - st))
    except KeyboardInterrupt:
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
