import librosa
import numpy as np
import matplotlib.pyplot as plt

class AudioAugmentation:
  def shuffle_name(self,name=''):
    return ''.join(random.sample(name,len(name)))

  def __init__(self, file_path,number_of_shuffles=1):
    self.data = librosa.core.load(file_path)[0]
    self.name = file_path.split('/')[-1].split('.')[0]
    self.dir  = '/'.join(file_path.split('/')[:-1])

  def write_audio_file(self, file, data, sample_rate=44000):
    data.export(file, bitrate = sample_rate,format = "mp3")

  def add_noise(self, data,parameter = 0.005):
    noise = np.random.randn(len(data))
    data_noise = data + parameter * noise
    return data_noise

  def shift(self, data,roller=1600):
    return np.roll(data, roller)

  def stretch(self, data, rate=1):
    data = librosa.effects.time_stretch(data, rate)
    return data

  def random_augmentation(self):
    shoot = random.randint(0,3)
    if (shoot ==1):
      self.augdata = self.add_noise(self.data)
    elif (shoot ==2):
      self.augdata = self.shift(self.data)
    elif (shoot ==3):
      self.augdata = self.stretch(self.data)
    
    new_name = self.shuffle_name(self.name)
    new_dir  = os.path.join(self.dir,new_name)
    self.write_audio_file(new_dir,self.augdata)



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
