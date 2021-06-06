from data.audio import Audio
from model.factory import tts_ljspeech

model = tts_ljspeech()
audio = Audio.from_config(model.config)
out = model.predict('Please, say something.')

# Convert spectrogram to wav (with griffin lim)
wav = audio.reconstruct_waveform(out['mel'].numpy().T)