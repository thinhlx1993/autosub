"""
Defines autosub's main functionality.
"""

#!/usr/bin/env python

from __future__ import absolute_import, print_function, unicode_literals

import argparse
import audioop
import math
import multiprocessing
import cv2
import os
import subprocess
import sys
import tempfile
import wave
import json
import requests
from datetime import datetime
try:
    from json.decoder import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

from googleapiclient.discovery import build
from progressbar import ProgressBar, Percentage, Bar, ETA

from autosub_v2.constants import (
    LANGUAGE_CODES, GOOGLE_SPEECH_API_KEY, GOOGLE_SPEECH_API_URL,
)
from autosub_v2.formatters import FORMATTERS
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='ch', use_gpu=True, rec_model_dir=r"C:\autosub_models") # need to run only once to load model into memory

DEFAULT_SUBTITLE_FORMAT = 'srt'
DEFAULT_CONCURRENCY = 10
DEFAULT_SRC_LANGUAGE = 'en'
DEFAULT_DST_LANGUAGE = 'vi'


import six
# from google.oauth2 import service_account
# from google.cloud import vision
# from google.cloud import translate_v2 as translate

# account_info = {
#     "type": "service_account",
#     "project_id": "iconic-era-306703",
#     "private_key_id": "ad24193131c38ec8d36f32cf9385a48a1283f8fb",
#     "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCoj4rmZx8SKbsG\nrC7FEPKviLBYBdYXPcs1DEsbEHD4ZCx1/a2fIzZzCXrkxSdiFMC9XcJA01cbynqK\nnspKI84dV5lwOJ9s5NCstwdOF+EIn4HZegQbIO6K4sHhFoSEE+wHgBnB1wi3cf9l\n1hNPmHAWNewULizqJjtlRJzc5HJ2ls2Zo289Vz4DsLQhsbQl9u/pziOx7NuxF22G\nIIV8H+K2pc/ipnKpPgjyCxoqyAI7WvgnEHgu0KGpU6xGlp5X4DBve/JSf6GFsRV/\ntM9z71IuGM2BYSxFoX/Wlw/N435ocGc9mv3h7Zu9FSa8yIN2shjNeL2c5OzIUlE/\nWS+/ZjoxAgMBAAECggEAA/4MX+sq2vsGjUCoRe4iFWLDLH6M5NWHmdzN7Zjs2BFF\nVVEaGuYSXNSpZsA9r87GhuWw22i2DDg2QVDEAVAKSnsf7P7GVeWwhJq8L08U+yeO\nA4jmjn7v73Wx+mMWZetz4HSaB5OQhwnJ7w9MO0skwn3p9stmMHCR4RLoNToq9OCR\nHU9HKmsWZXItroYfjbBeeHZ8R7A5rLaxpRYzCzLF0CfUwMuRjD9A1npJ9dZH5JG4\nFsirVlTKEAJmqHbwJn0/sWgp2Ph4dOJ+DKrkjl3qnZq/WHqba3R0llr4DBMcTlU4\nQgPuhMgAeBrt34Pf3qZQxVUGUycf7VK8Bpytka6QvQKBgQDny2mW7mfp7MzT4qQY\nwMrc4xEZoQRRNgr+o3q/QQy/AccO/lZvXZgukINV88QjVfbSYuW5klctfoitYGtf\n+WjRtS9QkyZ+rkeykB91XUnpObrIWjXVmqVrFWbLTJWoLghP7KaGf3sew7bJMpLr\nNmRPMqdOdImFzvmRqdcCFTS+fQKBgQC6KbFgS5g9M3NWQXoHx6SwOYAaE3a55+Cg\n5JHhNl0QVwwKIzfrP0+/9JQaxzMVOKiF1TGcLwcjotIkZlaufJTHTcNf9nZMTUxJ\n4fIVZZtk0/uR6vB6Z41ey2eIUnCyPkcsl/a6eP81LJf3ytFh58rvY4VBceunjK2A\nfEh3kqt0xQKBgAIODpiU8nzjaYlzV+sUQngk1zD3+XbS2NQbFOp/JCLJXD9ox9Fi\n7gdzpoZri9CYYYDJ+alkf7tahNGsqicGqgQ56/p144B6AQ63MmAy/IXBykMecZ28\nKj1BylCBFE6SYeZ7fZpxpODH8WXlOeI18Du3gj4y0ElMZXACJnLRR09tAoGAQrgu\nmhR9u3F1JLTSx3cFzyLMhovzQS2ZlBBXOCADupd3+SomIGnQazt82RwLcs+blluS\nLCeup1bzeZgz+NUtfUChhQMP4sjRTqlr2b9QshJHV0Sca0IxqIe90124hilL2O+d\nvbcfwC77SBOody5bzPAeEhaCHsqMZEAmuLQYPwECgYEA58pot3jrSWKoEp6FUJc1\n1IZ9iNpaHkCElskAh039RqUUPpaP3kjiqhC3N00f9OegT7mbRpQeYJUOBsU0omOg\nkM2VPRfsQYNOJK6x0Jo7cUeG1h2BAniTUYVDokvNwlFto4aBtyVmZSTK8RLRa12D\n4tgtq8xzU/QzBNSWB9DB/B4=\n-----END PRIVATE KEY-----\n",
#     "client_email": "autogeneratesubtitle@iconic-era-306703.iam.gserviceaccount.com",
#     "client_id": "104824741467913688408",
#     "auth_uri": "https://accounts.google.com/o/oauth2/auth",
#     "token_uri": "https://oauth2.googleapis.com/token",
#     "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
#     "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/autogeneratesubtitle%40iconic-era-306703.iam.gserviceaccount.com"
# }
#
# credentials = service_account.Credentials.from_service_account_info(account_info)
# client = vision.ImageAnnotatorClient(credentials=credentials)
# translate_client = translate.Client(credentials=credentials)
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


def detect_texts(img_path):
    """Detects text in the file."""
    result = ocr.ocr(img_path, det=False, rec=True, cls=False)
    for line in result:
        return line[0]
    return ""


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
    # print(u"Translation: {}".format(result["translatedText"]))
    # print(u"Detected source language: {}".format(result["detectedSourceLanguage"]))
    return result["translatedText"]


def translate_text(target, text):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    return text


def generate_subtitles( # pylint: disable=too-many-locals,too-many-arguments
        source_path,
        output=None,
        concurrency=DEFAULT_CONCURRENCY,
        src_language=DEFAULT_SRC_LANGUAGE,
        dst_language=DEFAULT_DST_LANGUAGE,
        subtitle_file_format=DEFAULT_SUBTITLE_FORMAT,
        api_key=None,
    ):
    """
    Given an input audio/video file, generate subtitles in the specified language and format.
    """
    # Opens the Video file
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1 / fps
    i = 0
    sub_idx = 1
    list_srt = []
    old_des = ""
    prev_time = 0
    start_time = 0
    file_name = os.path.basename(source_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % 12 == 0:
            prev_time_ts = datetime.utcfromtimestamp(prev_time).strftime('%H:%M:%S,%f')[:-4]
            start_time_ts = datetime.utcfromtimestamp(start_time).strftime('%H:%M:%S,%f')[:-4]
            h, w, c = frame.shape
            crop_img = frame[int(h * 0.94):h, 0:w]
            success, encoded_image = cv2.imencode('.jpg', crop_img)
            cv2.imwrite('tmp.jpg', encoded_image)
            description = detect_texts('tmp.jpg')

            if old_des != "" and (description != old_des or description == ""):
                list_srt.append({
                    "description": old_des,
                    "translate": translate_text(dst_language, old_des),
                    "first_time": prev_time_ts,
                    "last_time": start_time_ts,
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
                prev_time = start_time

            old_des = description
            # cv2.imshow('none', crop_img)
            # cv2.waitKey(1)
            start_time += time_per_frame * 12

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
        subtitle_file_path = generate_subtitles(
            source_path=args.source_path,
            concurrency=args.concurrency,
            src_language=args.src_language,
            dst_language=args.dst_language,
            api_key=args.api_key,
            subtitle_file_format=args.format,
            output=args.output,
        )
        print("Subtitles file created at {}".format(subtitle_file_path))
    except KeyboardInterrupt:
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
