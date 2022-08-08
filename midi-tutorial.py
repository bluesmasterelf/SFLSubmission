# imports
from random import random
from mido import MidiFile
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB, CategoricalNB

from sklearn.linear_model import LogisticRegression

# Constants
# time_chunks = np.zeros(30 seconds * 1350 ticks-per-second) by 128 square (the total number of possible notes) 
MAX_TICKS = 30*1350
BACH_DIR = 'sfl-data/MusicNet/PS1/Bach/'
BRAHMS_DIR = 'sfl-data/MusicNet/PS1/Brahms/'
BEETHOVEN_DIR = 'sfl-data/MusicNet/PS1/Beethoven/'
SCHUBERT_DIR = 'sfl-data/MusicNet/PS1/Schubert/'

# Using similar but different data to turn this into a supervised problem.
RAVEL_DIR = 'composerMidis/musicnet_midis/Ravel/'
MOZART_DIR = 'composerMidis/musicnet_midis/Mozart/'
DVORAK_DIR = 'composerMidis/musicnet_midis/Dvorak/'
CAMBINI_DIR = 'composerMidis/musicnet_midis/Cambini/'
FAURE_DIR = 'composerMidis/musicnet_midis/Faure/'
HAYDN_DIR = 'composerMidis/musicnet_midis/Haydn/'

SOURCE_DIRS = [BACH_DIR, BRAHMS_DIR, BEETHOVEN_DIR, SCHUBERT_DIR]
NON_DIRS = [RAVEL_DIR, MOZART_DIR] 
NON_DIRS += [DVORAK_DIR, CAMBINI_DIR, FAURE_DIR, HAYDN_DIR]


# helper methods and class declarations
class Note():
    """ For the purposes of midi, a note is a pitch with a start and stop time. 
    instrument could be added later, if it adds value. """
    def __init__(self, pitch, start, stop):
        self.pitch = pitch
        self.start = start
        self.stop = stop
    def duration(self):
        return self.stop - self.start

class Chord():
    def __init__(self, notes):
        self.notes = notes
    def complexity(self): 
        return [
            # total number of notes is a measure of chord complexity
            len(self.notes), 
            # net dissonance internal to the chord is a measure of complexity. 
            chord_dissonance(self),
        ]

dissonance_dict = {
    (0, 1), # tonic
    (1, 15), # flat 2
    (2, 8), # second
    (3, 5), # min third
    (4, 4), # maj thirds
    (5, 3), # perfect fourth
    (6, 50), # tritone
    (7, 2), # perfect fifth
    (8, 5), # sharp 5
    (9, 3), # sixth
    (10, 4), # maj seventh
    (11, 8), # seventh
}

def chord_dissonance(chord):
    dissonances = []
    for i in range(len(chord.notes)):
        for j in range(i +1, len(chord.notes)):
            note_dist = chord.notes[j].pitch - chord.notes[i].pitch
            dissonances.append(dissonance_dict[note_dist % 12])
    dissonance_index = np.average(dissonances)
    return dissonance_index


def load_training_data():
    midis = []
    non_midis = []
    
    for dir in SOURCE_DIRS:
        for filename in os.listdir(dir):
            mid = MidiFile(dir + filename, clip=True)
            midis.append(mid)
    for dir in NON_DIRS:
        for filename in os.listdir(dir):
            mid = MidiFile(dir + filename, clip=True)
            non_midis.append(mid)#, filename])
    return midis, non_midis

def load_test_data():
    midis = []
    TEST_DIR = 'sfl-data/MusicNet/PS2/'   
    for filename in os.listdir(TEST_DIR):
        mid = MidiFile(TEST_DIR + filename, clip=True)
        midis.append([mid, filename])
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
    for track in mid.tracks:
        times = []
        time = 0
        note_events = []
        for message in track:
            if message.type == 'note_on':
                pitch = message.note

                # Potentially very inefficient manner of walking the file and producing a list of notes with start and stop times.
                if message.time != 0:
                    time += message.time
                times.append([time, pitch])
                # I want to transform all of these events so that they are easier for me to understand. 
                # optimization would involve engineering this transform out.
                message.time = time
                note_events.append(message)

        # At this point, we have all of the times, and the messages have been adapted so that they match up with these times.
        # we want to break the track into a series of pieces of seconds with notes in them. 
        time_chunks = np.zeros((MAX_TICKS, 127))

        bag_of_notes = []
        for i in range(len(note_events)):
            if note_events[i].velocity > 0:
                # Then this is the introduction of a note, construct a note object. 
                this_note = Note(note_events[i].note, note_events[i].time, None)
                # we need to find the note end event for this note
                for j in range(i+1, len(note_events)):
                    if note_events[j].note == this_note.pitch:
                        this_note.stop = note_events[j].time
                        # stop looking for the end of the note. 
                        break
                if this_note.stop is None:
                    # must be some last index kind of issue. Just have it terminate out one tick later for now. Perform clean up later.
                    this_note.stop = this_note.start + 1
                bag_of_notes.append(this_note)        

        # Now, for each note, simply add it to every click for which that note is active. 
        for note in bag_of_notes:
            # stop when we've reached 30 seconds (or whatever the max time is. )
            if note.stop > MAX_TICKS: 
                break
            # use numpy slicing to apply the note value to every first index that is between its left and right.
            time_chunks[range(note.start, note.stop), note.pitch] = 1
        
        # right now, the collection of ticks carries massive amounts of redundancy. 
        # We should down-sample significantly to hold down compute times.
        tracks.append(time_chunks[0::10])

    # Aggregate tracks by simply adding them. 
    total_tracks = sum(tracks)
    total_tracks = np.reshape(total_tracks, (total_tracks.shape[0]* total_tracks.shape[1]))
    return total_tracks

# testing code
if __name__=="__main__":
    # originally followed tutorial https://www.twilio.com/blog/working-with-midi-data-in-python-using-mido

    # load training data
    midis, non_midis = load_training_data()

    # Sample vizualization
    #optional_visualization()    
    
    # load into numpy because sklearn likes numpy
    print("Loading training and test data.")
    data = []
    for mid in midis:
        data.append(np.array(reduce_midi_to_np_array(mid)))
    #non_data = []
    for mid in non_midis:
        data.append(np.array(reduce_midi_to_np_array(mid)))#, mid[1]])
    # Create a label set, 1 up to 184, 0 after. 
    y_train = np.zeros(len(data))
    y_train[:184] = 1

    test_data = []
    test_mids = load_test_data()
    for mid in test_mids:
        test_data.append(np.array(reduce_midi_to_np_array(mid[0])))#, mid[1]])
    
    # one class svm performed Really badly. 
    # print("One class svm")    
    # svm = OneClassSVM(nu = 0.01, gamma='scale', kernel='rbf').fit(data)
    # print(svm.predict(data))

    # # Now predict on the second set of data.
    # print(svm.predict(test_data))

    # print("Isolation Forest")
    # isof =  IsolationForest(contamination=0.01, random_state=42)
    # isof.fit(data)

    # print("Model is fit, predicting on original data. ")
    # print(isof.predict(data))
    # print("Predicting on test data now.")
    # print(isof.predict(test_data))

    # Elliptic Envelope requires waaaaaay too much memory for a dataset like this. It immediately squares the input size in ram.
    # possibly this could be used if the midis were reduced much more, as I intend to try at some point. 
    # print("Robust covariance")
    # envelope = EllipticEnvelope(contamination=0.01)
    # envelope.fit(data)
    # print("Envelop is fit, predicting on original data")
    # print(envelope.predict(data))
    # print(envelope.predict(test_data))

    # Running out of easy unsupervised approaches. Try making it a supervised problem instead. 
    # X, y = load_iris(return_X_y=True)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

    # print("Using supervised Naive Bayes due to small sample size.")
    # gnb = GaussianNB()
    # y_pred = gnb.fit(data, y_train)
    # print(gnb.predict(data))
    # print(gnb.predict(test_data))

    # That appeared to work. Now I want to prove it is correct. To do that, I'd have to compare to all the midis in the non-data.
    # for outlier in test_data:
    #     for other in data:
    #         diff = outlier[0] - other[0]
    #         total_diff = np.sum(diff)
    #         #if total_diff < 15 and total_diff > -15:
    #         #print(other)
    #         #print(total_diff)
    #         if total_diff == 0:
    #             print(outlier[1] + other[1])
    # It did work, but only because the outliers ended up in my training set. Need to do better than that.

    #gnb = MultinomialNB()
    # multinomialNB got 2 out of 3, but false positived on 8 others. [0. 1. 0. 1. 1. 0. 0. 1. 1. 0. 1. 1. 1. 0. 1. 0. 1. 1. 1. 1. 0. 1. 1. 0.
    #1. 1. 1. 0. 1. 1. 0. 1. 1. 1. 1.] barely better than random.
    #gnb = CategoricalNB()

    print("Using supervised Logistic Regression due to small sample size.")
    gnb = LogisticRegression(solver='liblinear', max_iter=250, penalty='l1')

    # ComplementNB was identical to MultinomialNB - the dataset must not be sufficiently imbalanced for it to work.
    # BernoulliNB was pretty crap
    # so was CategorigcalNB
    y_pred = gnb.fit(data, y_train)
    # print(gnb.predict(data))
    # print(gnb.predict(test_data))

    # When the mids in this set were processed down before, they were altered. Processing them down again
    # somehow allows the model to flag 7 outliers, 3 of which are the correct ones. 
    # I'm committing this because, hey, it kinda worked. 
    # Thinking about how processing twice would work, I think the start and stop times of notes would be doubled. 
    # which should perturb the model on ALL of the inputs, but somehow this allows it to catch the Mozart pieces... 
    # Upon listening to the three outlier Mozart pieces, I can tell two of them are quite slow, so slowing them down much more
    # could be what is tripping off the detector. This could be a valid preprocessor. Something about length of notes
    # may be viable discriminator between composers within this particular problem space. 
    test_two = []
    for mid in test_mids:
        temp = np.array(reduce_midi_to_np_array(mid[0]))
        test_two.append(temp)
        pred = gnb.predict_proba([temp])
        if pred[0][1] < 0.44:
            print(str(pred[0][1]) + " " + mid[1])
            # This appears to get all three, plus one or two false positive errata. 
    print(gnb.predict(test_two))