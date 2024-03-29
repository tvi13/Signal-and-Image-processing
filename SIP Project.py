#importing libraries
import numpy as np  # linear algebra
import pandas as pd  # CSV file
import scipy.io.wavfile as sci_wav  # Open wav files
import matplotlib.pyplot as plt
import numpy as np
import cv2
import wave
from matplotlib.pyplot import *
import random
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


ROOT_DIR = 'cats_dogs'
CSV_PATH = '/Users/tvishamajithia/Desktop/Audios/train_test_split.csv'


def read_wav_files(wav_files):
    '''Returns a list of audio waves
    Params:
        wav_files: List of .wav paths

    Returns:
        List of audio signals
    '''
    if not isinstance(wav_files, list):
        wav_files = [wav_files]
    return [sci_wav.read(ROOT_DIR + f)[1] for f in wav_files]


def get_trunk(_X, idx, sample_len, rand_offset=False):
    '''Returns a trunk of the 1D array <_X>

    Params:
        _X: the concatenated audio samples
        idx: _X will be split in <sample_len> items. _X[idx]
        rand_offset: boolean to say whether or not we use an offset
    '''
    randint = np.random.randint(10000) if rand_offset is True else 0
    start_idx = (idx * sample_len + randint) % len(_X)
    end_idx = ((idx + 1) * sample_len + randint) % len(_X)
    if end_idx > start_idx:  # normal case
        return _X[start_idx: end_idx]
    else:
        return np.concatenate((_X[start_idx:], _X[:end_idx]))


def get_augmented_trunk(_X, idx, sample_len, added_samples=0):
    X = get_trunk(_X, idx, sample_len)

    # Add other audio of the same class to this sample
    for _ in range(added_samples):
        ridx = np.random.randint(len(_X))  # random index
        X = X + get_trunk(_X, ridx, sample_len)

    # One might add more processing (like adding noise)

    return X


def dataset_gen(is_train=True, batch_shape=(20, 16000), sample_augmentation=0):
    '''This generator is going to return training batchs of size <batch_shape>

    Params:
        is_train: True if you want the training generator
        batch_shape: a tupple (or list) consisting of 2 arguments, the number
            of samples per batchs and the number datapoints per samples
            (16000=1s)
        sample_augmentation: augment each audio sample by n other audio sample.
            Only works when <is_train> is True
    '''
    s_per_batch = batch_shape[0]
    s_len = batch_shape[1]

    X_cat = dataset['train_cat'] if is_train else dataset['test_cat']
    X_dog = dataset['train_dog'] if is_train else dataset['test_dog']

    # Random permutations (for X indexes)
    nbatch = int(max(len(X_cat), len(X_cat)) / s_len)
    perms = [list(enumerate([i] * nbatch)) for i in range(2)]
    perms = sum(perms, [])
    random.shuffle(perms)


    # Go through all the permutations
    y_batch = np.zeros(s_per_batch)
    X_batch = np.zeros(batch_shape)
    while len(perms) > s_per_batch:

        # Generate a batch
        for bidx in range(s_per_batch):
            perm, _y = perms.pop()  # Load the permutation
            y_batch[bidx] = _y  

            # Select wether the sample is a cat or a dog
            _X = X_cat if _y == 0 else X_dog

            # Apply the permutation to the good set
            if is_train:
                X_batch[bidx] = get_augmented_trunk(
                    _X,
                    idx=perm,
                    sample_len=s_len,
                    added_samples=sample_augmentation)
            else:
                X_batch[bidx] = get_trunk(_X, perm, s_len)

        yield (X_batch.reshape(s_per_batch, s_len, 1),
               y_batch.reshape(-1, 1))


def load_dataset(dataframe):
    '''Load the dataset in a dictionary.
    From the dataframe, it reads the [train_cat, train_dog, test_cat, test_dog]
    columns and loads their corresponding arrays into the <dataset> dictionary

    Params:
        dataframe: a pandas dataframe with 4 columns [train_cat, train_dog, 
        test_cat, test_dog]. In each columns, many WAV names (eg. ['cat_1.wav',
        'cat_2.wav']) which are going to be read and append into a list

    Return:
        dataset = {
            'train_cat': [[0,2,3,6,1,4,8,...],[2,5,4,6,8,7,4,5,...],...]
            'train_dog': [[sound 1],[sound 2],...]
            'test_cat': [[sound 1],[sound 2],...]
            'test_dog': [[sound 1],[sound 2],...]
        }
    '''
    df = dataframe

    dataset = {}
    for k in ['train_cat', 'train_dog', 'test_cat', 'test_dog']:
        v = list(df[k].dropna())
        v = read_wav_files(v)
        v = np.concatenate(v).astype('float32')

        # Compute mean and variance
        if k == 'train_cat':
            dog_std = dog_mean = 0
            cat_std, cat_mean = v.std(), v.mean()
        elif k == 'train_dog':
            dog_std, dog_mean = v.std(), v.mean()

        # Mean and variance suppression
        std, mean = (cat_std, cat_mean) if 'cat' in k else (dog_std, dog_mean)
        v = (v - mean) / std
        dataset[k] = v

        print('loaded {} with {} sec of audio'.format(k, len(v) / 16000))

    return dataset

wav_obj = wave.open('dog.wav','rb')
sample_freq = wav_obj.getframerate()
n_samples = wav_obj.getnframes()
n_channels = wav_obj.getnchannels()
signal_wave = wav_obj.readframes(n_samples)
t_audio = n_samples/sample_freq
signal_array = frombuffer(signal_wave, dtype=int16)
l_channel = signal_array[0::2]
r_channel = signal_array[1::2]
times = linspace(0, n_samples/sample_freq, num=n_samples)
plt.figure(figsize=(15, 5))
plt.plot(times, l_channel)
plt.title('Left Channel')
plt.ylabel('Signal Value')
plt.xlabel('Time (s)')
plt.xlim(0, t_audio)
plt.show()

df = pd.read_csv(CSV_PATH)
dataset = load_dataset(df)

# Function to extract features from audio file
def extract_features(audio_path):
    sample_rate, audio_data = wavfile.read(audio_path)
    # Convert audio data to mono if it has multiple channels
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    # Calculate Short-Time Fourier Transform (STFT)
    frequencies, times, spectrogram = plt.specgram(audio_data, Fs=sample_rate, NFFT=1024, noverlap=512)
    return frequencies, times, spectrogram.T

# Function to preprocess spectrogram features
def preprocess_spectrogram(spectrogram):
    scaler = StandardScaler()
    return scaler.fit_transform(spectrogram)

# Function to apply dimensionality reduction
def reduce_dimensions(spectrogram):
    pca = PCA(n_components=50)  # Reducing to 50 principal components
    return pca.fit_transform(spectrogram)

# Function to train a Support Vector Machine (SVM) classifier
def train_classifier(X, y):
    svm = SVC(kernel='rbf', C=1, gamma='auto')  # Using RBF kernel SVM
    svm.fit(X, y)
    return svm

# Function to classify a given audio file
def classify_audio(audio_path, classifier):
    _, _, spectrogram = extract_features(audio_path)
    preprocessed_spectrogram = preprocess_spectrogram(spectrogram)
    reduced_spectrogram = reduce_dimensions(preprocessed_spectrogram)
    prediction = classifier.predict(reduced_spectrogram)
    return prediction[0]

# Function to display image of animal
def display_animal_image(predicted_label):
    # Mapping between predicted labels and corresponding animal images
    animal_images = {
        0: "cat.jpeg",
        1: "dog.jpeg",
        2: "bird.jpeg",
        3: "elephant.jpeg",
        4: "horse.jpeg",
        5: "cow.jpeg",
        6: "monkey.jpeg",
        7: "lion.jpeg",
        8: "tiger.jpeg",
        9: "wolf.jpeg"
    }
    image_path = animal_images.get(predicted_label)
    if image_path:
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print("No image found for the predicted label.")

# Load and preprocess data
X = []  # Features
y = []  # Labels
for label in range(10):  # Assuming there are 10 classes
    for i in range(1, 11):  # Assuming there are 10 audio samples per class
        audio_path = f"/Users/tvishamajithia/Desktop/Audios.wav"
        frequencies, times, spectrogram = extract_features(audio_path)
        preprocessed_spectrogram = preprocess_spectrogram(spectrogram)
        reduced_spectrogram = reduce_dimensions(preprocessed_spectrogram)
        X.append(reduced_spectrogram)
        y.append(label)

X = np.concatenate(X)
y = np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
classifier = train_classifier(X_train, y_train)

# Predict labels for test data
y_pred = classifier.predict(X_test)

# Evaluate the classifier
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Path to the audio file for classification
audio_file_path = "audio_file.wav"

# Classify the given audio file
predicted_label = classify_audio(audio_file_path, classifier)
print("Predicted Label:", predicted_label)

# Display the image of the predicted animal
display_animal_image(predicted_label)
window_name = 'image'
cv2.imshow(window_name,img) 
cv2.waitKey(0) 
cv2.destroyAllWindows()