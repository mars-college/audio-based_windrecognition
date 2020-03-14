#  classify audio - wind or no wind 


#librosa == 0.5.1


from oscpy.server import OSCThreadServer
from time import sleep


import numpy as np
import librosa, pyaudio, time, request, json, keras, os, warnings
import tensorflow as tf

### ignore warnings 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

ifttt_event = 'wind or no wind | brahman'
ifttt_key = '*********************'
ifttt_url = 'https://maker.ifttt.com/trigger/{event}/with/key/{key}'.format(
    event=ifttt_event, key=ifttt_key)
def make_web_request(label):
    data = {}
    data['value1'] = label
    try:
        response = requests.post(ifttt_url, data=data)
        print('{0.status_code}: {0.text}'.format(response))
    except:
        print('Failed to make a web request')


sr = 16000 # make sure this is same as trained audio 
chunk = 1 * sr
ffs_sz = 256
threshold = 0.2

model_name = 'wind_nowind.h5'
json_name = 'state_names.json'

with open(json_name, 'r') as file_to_load:
    LABELS = json.load(file_to_load)
    LABELS.append('unknown')

N_LABELS = len(LABELS) - 1

last_label = LABELS.index('unknown')


model = keras.models.load_model(model_name)

count = 0
predictions_in_60_sec = np.empty((0, N_LABELS))

audio_interface = pyaudio.PyAudio()
audio_stream = audio_interface.open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=sr,
                                    input=True,
                                    frames_per_buffer=chunk)


try:
    while True:
        data = np.fromstring(audio_stream.read(chunk),
                             dtype=np.int16)

        audio_stream.stop_stream()

        start = time.time()

        state = last_label

        D = librosa.stft(librosa.util.normalize(data),
                         n_fft=ffs_sz,
                         window='hamming')
        magnitude = np.abs(D).transpose()

        predictions = model.predict_proba(magnitude, verbose=False)
        predictions_mean = predictions.mean(axis=0)
        predictions_in_60_sec = np.vstack([predictions_in_60_sec, predictions])
        count = count + 1
        
        #here for sending osc message
        if LABELS[predictions_mean.argmax()] == 'wind':
            print('its windy!')
            from oscpy.client import OSCClient
            address = "10.79.103.100"
            port = 7001
            osc = OSCClient(address, port)
            for i in range(1):
                osc.send_message(b'/wind', [i])
                print('osc sending')
            
      
        elapsed_time = time.time() - start

        localtime = '{0.tm_hour:02d}:{0.tm_min:02d}:{0.tm_sec:02d}'.format(
            time.localtime(time.time()))
        print('{0:s} {1:s} - (accuracy {2:.3f}, processed in {3:.3f} seconds)'.format(
            localtime,
            LABELS[predictions_mean.argmax()],
            predictions_mean.max(),
            elapsed_time))

        if predictions_mean.max() > threshold:
            label = predictions_mean.argmax()

        if last_label != label:
            print('Changed: {0} > {1}'.format(
                LABELS[last_label], LABELS[label]))
            last_label = label

        if (count == 60):
            start = time.time()
            label_in_60_sec = predictions_in_60_sec.mean(axis=0).argmax()
            make_web_request(LABELS[label_in_60_sec])
            predictions_in_60_sec = np.empty((0, N_LABELS))
            count = 0
            elapsed_time = time.time() - start
            print('Made a request in {0:.3f} seconds)'.format(elapsed_time))

        audio_stream.start_stream()

except KeyboardInterrupt:
    print('Requested to terminate')

finally:
    audio_stream.stop_stream()
    audio_stream.close()
    audio_interface.terminate()
    print('Terminated')
