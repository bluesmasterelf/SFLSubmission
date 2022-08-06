# imports
from random import random
from mido import MidiFile
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.svm import OneClassSVM

# helper methods and class declarations
class Note():
    def __init__(self, pitch, start, stop):
        self.pitch = pitch
        self.start = start
        self.stop = stop
    def duration(self):
        return self.stop - self.start

def load_training_data():
    midis = []
    BACH_DIR = 'sfl-data/MusicNet/PS1/Bach/'
    BRAHMS_DIR = 'sfl-data/MusicNet/PS1/Brahms/'
    BEETHOVEN_DIR = 'sfl-data/MusicNet/PS1/Beethoven/'
    SCHUBERT_DIR = 'sfl-data/MusicNet/PS1/Schubert/'
    SOURCE_DIRS = [BACH_DIR, BRAHMS_DIR, BEETHOVEN_DIR, SCHUBERT_DIR]
    for dir in SOURCE_DIRS:
        for filename in os.listdir(dir):
            mid = MidiFile(dir + filename, clip=True)
            midis.append(mid)
    return midis

def load_test_data():
    midis = []
    TEST_DIR = 'sfl-data/MusicNet/PS2/'   
    for filename in os.listdir(TEST_DIR):
        mid = MidiFile(TEST_DIR + filename, clip=True)
        midis.append(mid)
    return midis

def optional_visualization():
    mid = MidiFile('composerMidis/musicnet_midis/Beethoven/2318_bh38m1.mid', clip=True) #C:\Users\hogan\OneDrive\Desktop\repos\SFLSubmission\composerMidis\musicnet_midis\Beethoven\2318_bh38m1.mid
    print(mid)
    random_track = mid.tracks[0]
    print(random_track)
    
    # view content a bit
    partial_track = random_track
    for message in partial_track:
        # print(message)
        # print(message.type)
        if message.type == 'note_on':
            print(message)
            print(message.note)
    for track in mid.tracks[1:2]:
            notes = [note for note in track if note.type == 'note_on']
            pitch = [note.note for note in notes]
            # tick = [note.time for note in notes]
            # tracks += [pitch]
            # display the midi track as a plot - adapted from https://gitlab.com/colinfwren/midivis/-/commit/6adc9a757dfd8d4a62be7275a6d24a1449755429
            plt.plot(*[pitch])
            plt.show()


def reduce_midi_to_np_array(mid):
    tracks = []
    for track in mid.tracks[1:2]:
        times = []
        time = 0
        for message in track:
            if message.type == 'note_on':
                # TODO use the note_off messages.
                pitch = message.note

                # Potentially very inefficient manner of walking the file and producing a list of notes with start and stop times.
                if message.time != 0:
                    time += message.time
                times.append(time)
        tracks.append(times)
        # start with the absurdly naive idea that a track can be reduced the total number of time stamps. 
    return [len(tracks[0])]

# testing code
if __name__=="__main__":
    # originally followed tutorial https://www.twilio.com/blog/working-with-midi-data-in-python-using-mido

    # load training data
    midis = load_training_data()

    # Sample vizualization
    #optional_visualization()    
    
    # load into numpy because sklearn likes numpy
    data = []
    for mid in midis:
        data.append(np.array(reduce_midi_to_np_array(mid)))
        # TODO next line fails. 
    #numpy_data = np.array(data)
    
    clf = OneClassSVM(gamma='auto').fit(data)
    var = clf.predict(data)
    print(var)

    # Now predict on the second set of data.
    test_data = []
    test_mids = load_test_data()
    for mid in test_mids:
        test_data.append(np.array(reduce_midi_to_np_array(mid)))

    print(clf.predict(test_data))