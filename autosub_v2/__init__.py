"""
Defines autosub's main functionality.
"""

#!/usr/bin/env python

from __future__ import absolute_import, print_function, unicode_literals

import time
import argparse
import audioop
import math
import multiprocessing
import cv2
import os
# import subprocess
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

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from googleapiclient.discovery import build
from progressbar import ProgressBar, Percentage, Bar, ETA

from autosub_v2.constants import (
    LANGUAGE_CODES, GOOGLE_SPEECH_API_KEY, GOOGLE_SPEECH_API_URL,
)
from autosub_v2.formatters import FORMATTERS
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang='ch', use_gpu=False,
                rec_model_dir=r"C:\autosub_models\rec",
                cls_model_dir=r"C:\autosub_models\cls",
                det_model_dir=r"C:\autosub_models\det",
                use_angle_cls=True,
                rec_char_type='ch',
                drop_score=0.8,
                det_db_box_thresh=0.3,
                cls=True)

DEFAULT_SUBTITLE_FORMAT = 'srt'
DEFAULT_CONCURRENCY = 10
DEFAULT_SRC_LANGUAGE = 'en'
DEFAULT_DST_LANGUAGE = 'vi'


import six
from google.oauth2 import service_account
from google.cloud import vision
from google.cloud import translate_v2 as translate

account_info = {
    "type": "service_account",
    "project_id": "iconic-era-306703",
    "private_key_id": "77b1f06485345d49bb49d7c9299456fb71303ab3",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDhWfJc9tZGHxlj\n19+m5aX/wLvZ671d3SSKDPmrDtIkIi1RxvSOMVbb/0mSUlOcrbi+2CVg6l0MASCt\n1k23mq4vDpPOBfPHY9OxwMVi499NkaBKf050g/l1Bnw3VIi3uu4EJaWQuFcoAXeS\nrVFHTP6JgB3V/iQSGG/gjEdIwvsTakSy4DOMAhitavYLU/fIIipDEwBOlL5tbQ+p\nxSRgA7Dpsr2FLM8gDWASTm6LoGVvpL/N11uuSrcKD9P/HWQyA+PI9S9PQWKRjkPo\n/S9A9oQgRMUEloj5QZHMqh9mpeEuoup6FqE1LfH7auNPmLyWvAwygw031Igqm0tk\nzCTGbw3bAgMBAAECggEAWsdxpzp2UfQYNczSFzj+uHrLEbvx4lyB6izU7LBBPfYY\nnI9Rl/BPRbtex1drwDuJJzQKRrLSfdH8eJrSXuqsTV+Jch6auBFCR6JYwX/7RhOq\nZyhGkhBSDu7oXh+rHZxrYndJ7XUdAwwoP4mbKuZcyUY3fqtsm2+FrgbEdo65NvXA\nCplWeDAujGSADMszX5tFii+uWbDEVrvNkvAwCpTv+PEclseFTKaLbhtMejeoHW0z\n1QbLDx2mCcYvIHQ52GGMFHsVHdfEYT/WFLWd5YnjX6KmY2kCFtQ6cWJ3Qo6FMUgF\nnYx/22G7S0GdYybzfqE2XnGUar0HR/Aoe25R2UkdtQKBgQD4qOCqGmDd2YF3DEI6\nP/m9epcSfJfmV4ZSLJ9VNo7nzVd/8swXpZBw2nqQizUeP4Rim5EH1gNbB5PVSumU\nX5hcBfdAOHc9DNNjexEeQQNR4JOPsMAN2BzVtgqP2hgZjkjBoHninIcd++29o4ga\nvvMyaKJ2/bD6jVjbsrxW/OF8HQKBgQDoAO2Sq4wX5jRhTA8xfiUCn4XRwRDN/IEK\ny00exRq3alFYOrQH3jYWxRGbFrwi6Br13rxWKcryukPwU7w/Xi5M4ytMpjESKJ96\n5NY1dU11rNgHd0kRYfuaaQv+X5ZNBG0AXYL+4KANnwumVNMV1In7TJXrBvqqn1++\n4O7dst9gVwKBgBq482P0b8KHtG0ZySg/ZdRiD0gyUZS0hT/hgcIDmfn5TFT4v8wu\nw8YNBKzx+ORmSRDbzQs9iaDHwLBkW5PRbis9jOO+7bmG3lTLjfxlWjj7XIBNq2YR\neo/Q/3OUKZDdhJ4iY9bhoXesclE1+NN+/93D9um4u8NBW3JI1Aq5JHZVAoGACBtQ\nMdnQsV0X43Z26XHQ9UCBuoyWe7wg/jGQZkzY3CPY585VUBkRpsYIEXU/6bBWkNTR\nm+kl8ElV6mXipAw0bfdaIfmEqW/F8tNgMMoChOQfQFOIuBTGZ+TXyHGqnNJUxLh/\nxUwwC4nNLGi2X4Lnt2I7stSxmZisDc1qS1DewU0CgYEA58RFBSp7v+rklgppKEq8\nP2H901xvB+KA8dX0Mt1uadIu7BuAw9oWxr8Kj1Vs0gtyU2OkIpMwx5xYu7YXYIFA\n4YcIj8r8YOpimATyyE5nzKmH5gkPkev3XZyFQ0fyng77ZME+/ng2cex+kCuM1KuK\nh5tYaUSRTWyUJR/YzKNi/Ao=\n-----END PRIVATE KEY-----\n",
    "client_email": "default@iconic-era-306703.iam.gserviceaccount.com",
    "client_id": "116179332925558425360",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/default%40iconic-era-306703.iam.gserviceaccount.com"
}


credentials = service_account.Credentials.from_service_account_info(account_info)
# client = vision.ImageAnnotatorClient(credentials=credentials)
translate_client = translate.Client(credentials=credentials)
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def detect_texts_google_cloud(content):
#     """Detects text in the file."""
#     image = vision.Image(content=content)
#     response = client.text_detection(image=image)
#     texts = response.text_annotations
#     predict_des = ""
#     for text in texts:
#         predict_des = text.description
#         predict_des = predict_des.strip()
#         return predict_des
#
#     return predict_des


def detect_texts(img_path):
    """Detects text in the file."""
    result = ocr.ocr(img_path, det=False, rec=True, cls=False)
    for line in result:
        # print(line[0])
        if line[1] > 0.7:
            return line[0]
    return ""


def translate_text_google_cloud(target='vi', text=''):
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target, source_language='ch')

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


def generate_subtitles(
        source_path,
        output=None,
        dst_language=DEFAULT_DST_LANGUAGE
    ):
    """
    Given an input audio/video file, generate subtitles in the specified language and format.
    """
    # Opens the Video file
    print("starting")
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
            cv2.imshow('demo', crop_img)
            cv2.waitKey(1)
            # success, encoded_image = cv2.imencode('.jpg', crop_img)
            cv2.imwrite('tmp.jpg', crop_img)
            description = detect_texts('tmp.jpg')

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
        )
        print("Subtitles file created at {} time consumer: {}".format(subtitle_file_path, time.time() - st))
    except KeyboardInterrupt:
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
