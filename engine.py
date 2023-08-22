import logging
alsa_logger = logging.getLogger('alsa')
alsa_logger.setLevel(logging.WARNING)

from absl import app
from absl import logging
import json

from paddlespeech.cli.tts.infer import TTSExecutor
import pygame
import pyttsx3
import requests
import speech_recognition as sr


def get_device_index():
    mics = sr.Microphone.list_microphone_names()
    for i, name in enumerate(mics):
        logging.info(name)
        if name.find("hw:1,7") > 0:
            device_index = i
    return device_index


class Engine():

    def __init__(self):
        self._tts = TTSExecutor()
        pygame.init()
        pygame.mixer.init()
        self._engine = None
        self.init_engine()
        self._device_index = get_device_index()

        # mdoel
        self._use_chatglm = False
        self._tokenizer = None
        self._model = None
        self.init_chatglm()
        self._history = []

    def init_engine(self):
        self._engine = pyttsx3.init()
        self._engine.setProperty('voice', 'zh')
        self._engine.setProperty('rate', 70)

    def speak(self, text):
        self._tts(text=text, output="output.wav")
        pygame.mixer.music.load('output.wav')
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()

    def init_chatglm(self):
        if not self._use_chatglm:
            return
        from transformers import AutoModel
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm-6b-int4", trust_remote_code=True)
        self._model = AutoModel.from_pretrained(
            "THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
        self._model = model.eval()

    def get_response(self, text):
        if self._use_chatglm:
            response, self._history = self._model.chat(self._tokenizer, text, self._history)
            return response
        url = 'http://10.5.18.8:5000/chat'
        data = {'chat_text': text}
        response = requests.post(url, json=data)
        result = response.json()
        return result['response']

    def run(self):
        text = None
        while True:
            recognizer = sr.Recognizer()
            try:
                with sr.Microphone(device_index=self._device_index) as source:
                    logging.info('listening')
                    recognizer.operation_timeout = 60
                    audio = recognizer.listen(source, timeout=60)
                    text = recognizer.recognize_google(audio, language='zh-CN')
            except Exception as e:
                logging.info(e)
                continue

            print('input_text : {}'.format(text))
            response = self.get_response(text)
            print('response text : {}'.format(response))
            self.speak(response)


def main(_):
    engine = Engine()
    engine.run()


if __name__ == '__main__':
    app.run(main)

