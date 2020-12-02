import librosa
import numpy as np
import matplotlib.pyplot as plt

class AudioAugmentation:
  def shuffle_name(self,number=8,name=''):
    names = []
    for i in range(number):
      names.append(''.join(random.sample(name,len(name))))
    return names  

  def read_audio_file(self, file_path):
      data     = librosa.core.load(file_path)[0]
      self.name= file_path.split('/')[-1].split('.')[0]
      return data

  def write_audio_file(self, file, data, sample_rate=44000):
      data.export(file, bitrate = sample_rate,format = "mp3")

  def add_noise(self, data):
      noise = np.random.randn(len(data))
      data_noise = data + 0.005 * noise
      return data_noise

  def shift(self, data):
      return np.roll(data, 1600)

  def stretch(self, data, rate=1):
      data = librosa.effects.time_stretch(data, rate)
      return data

# Create a new instance from AudioAugmentation class
aa = AudioAugmentation()

# Read and show cat sound
data = aa.read_audio_file("data/cat.wav")
aa.plot_time_series(data)

# Adding noise to sound
data_noise = aa.add_noise(data)
aa.plot_time_series(data_noise)

# Shifting the sound
data_roll = aa.shift(data)
aa.plot_time_series(data_roll)

# Stretching the sound
data_stretch = aa.stretch(data, 0.8)
aa.plot_time_series(data_stretch)

# Write generated cat sounds
aa.write_audio_file('output/generated_cat1.wav', data_noise)
aa.write_audio_file('output/generated_cat2.wav', data_roll)
aa.write_audio_file('output/generated_cat3.wav', data_stretch)
