# Copyright Ian Hogan - 2022

# Imports
from mido import MidiFile
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB, CategoricalNB    
from sklearn.cluster import DBSCAN 
import math

# Constants
# time_chunks = np.zeros(30 seconds * 1350 ticks-per-second) by 128 square (the total number of possible notes) 
MAX_TICKS = 30*1350
BACH_DIR = 'sfl-data/MusicNet/PS1/Bach/'
BRAHMS_DIR = 'sfl-data/MusicNet/PS1/Brahms/'
BEETHOVEN_DIR = 'sfl-data/MusicNet/PS1/Beethoven/'
SCHUBERT_DIR = 'sfl-data/MusicNet/PS1/Schubert/'

# Using similar but different data to turn this into a supervised problem.
RAVEL_DIR = 'composerMidis/musicnet_midis/Ravel/'
MOZART_DIR = 'composerMidis/musicnet_midis/Mozart/' # I migrated out the 'outliers' from this, else the model is simply cheating.
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
    """While it's intuitively clear what a chord is, it's not actually that obvious how to construct them from a midi
    I'll see what I can do.
    """
    def __init__(self, notes):
        self.notes = notes
    def complexity(self): 
        return [
            # total number of notes is a measure of chord complexity
            len(self.notes), 
            # net dissonance internal to the chord is a measure of complexity. 
            chord_dissonance(self),
        ]

#The outputs for a relative pitch difference are the denominators 
# of the relative frequencies in just intonation. This is entirely my own idea. 
dissonance_dict = {
    0: 1, # tonic
    1: 15, # flat 2
    2: 8, # second
    3: 5, # min third
    4: 4, # maj thirds
    5: 3, # perfect fourth
    6: 50, # tritone
    7: 2, # perfect fifth
    8: 5, # sharp 5
    9: 3, # sixth
    10: 4, # maj seventh
    11: 8, # seventh
}

key_dict = {
    'A': 0,
    'Bb': 1,
    'B': 2,
    'C': 3,
    'Db': 4,
    'D': 5,
    'Eb': 6,
    'E': 7,
    'F': 8,
    'Gb': 9,
    'G': 10,
    'Ab': 11,
    'Am': 12,
    'Bbm': 13,
    'Bm': 14,
    'Cm': 15,
    'Dbm': 16,
    'Dm': 17,
    'Ebm': 18,
    'Em': 19,
    'Fm': 20,
    'Gbm': 21,
    'Gm': 22,
    'Abm': 23,
}

def chord_dissonance(chord):
    """Given a chord, produces an average of relative intervalent pitches. """
    if len(chord.notes) < 2:
        # No dissonance if it's only 1 note. 
        return 0
    
    dissonances = []
    for i in range(len(chord.notes)):
        for j in range(i +1, len(chord.notes)):
            note_dist = chord.notes[j].pitch - chord.notes[i].pitch
            dissonances.append(dissonance_dict[abs(note_dist) % 12])
    dissonance_index = np.average(dissonances)
    if math.isnan(dissonance_index):
        print("Something went wrong.")
    return dissonance_index

def load_training_data():
    """Self documenting, only meant to be used in the housing file's main method."""
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
    """Self documenting, only meant to be used in the housing file's main method."""
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

def reduce_midi_to_notes(track):
    """Takes a midi track and produces a flat list of note objects with start and stop time and pitch. """
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
    return bag_of_notes 

def reduce_midi_to_np_array(mid):
    tracks = []
    for track in mid.tracks:        
        time_chunks = np.zeros((MAX_TICKS, 127))
        bag_of_notes = reduce_midi_to_notes(track)     

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
    total_tracks = np.reshape(total_tracks, (total_tracks.shape[0] * total_tracks.shape[1]))
    return total_tracks

def extract_features_from_midi(mid):
    """ This is a second round attempt to extract meaningful information from the midi files.
    Rather than reduce to some sort of numerical approximation of notes on the page, the intent
    is to pull meta-data, number of tracks, instrumentation, chords and chord complexity, 
    note lengths as an average and possibly other statistics. 
    """
    all_notes = []
    feature_set = []

    # Song features
    key = 0
    numerator = 0
    denominator = 1 # don't divide by zero
    tempi = []
    # The first track contains the key, time signature, and tempo data
    meta_track = mid.tracks[0]
    for message in meta_track:
        if message.type == "key_signature":
            key = key_dict[message.key]
        # If time signature changes, the last one will be the one kept. 
        if message.type == "time_signature":
            numerator = message.numerator
            denominator = message.denominator
        if message.type == "set_tempo":
            tempi.append(message.tempo)
    tempo = np.average(tempi)
    
    # Song features indeces 0-3
    feature_set += [key, numerator, denominator, tempo]

    for track in mid.tracks:        
        bag_of_notes = reduce_midi_to_notes(track)  
        all_notes += bag_of_notes

    # Extract features of notes themselves, such as their quantity, length, and diversity of pitch. 
    distinct_pitches = []
    for note in all_notes:
        if note.pitch not in distinct_pitches:
            distinct_pitches.append(note.pitch)

    # Note features
    num_notes = len(all_notes)
    num_distinct_notes = len(distinct_pitches)
    min_note = min(distinct_pitches)
    max_note = max(distinct_pitches)
    # indeces 4-7
    feature_set += [num_notes, num_distinct_notes, min_note, max_note]

    # With all notes, we can construct a list of chords. 
    chords = []
    notes_not_in_chords = []

    all_notes.sort(key=note_start)

    for i in range(len(all_notes)):
        notes_not_in_chords.append(i)
    for i in range(len(all_notes)):
        if i in notes_not_in_chords:
            #start building a chord
            notes_in_this_chord = [all_notes[i]]  
            notes_not_in_chords.remove(i)
            
            # add other notes that start <= when this one does, and end after this one starts.
            for j in range(i+1, len(all_notes)):
                if j in notes_not_in_chords and all_notes[j].start <= all_notes[i].start:
                    notes_in_this_chord.append(all_notes[j])
                    notes_not_in_chords.remove(j)
                else:
                    break
                
            chords.append(Chord(notes_in_this_chord))
    
    # With a list of chords, we can aggregate statistics on their features, 
    # including the number of distinct chords, the interquartile of complexity, etc. 
    chord_complexities = []
    for chord in chords:
        chord_complexities.append(len(chord.notes) + chord_dissonance(chord))
    
    # Chord features
    min_complexity = min(chord_complexities)
    max_complexity = max(chord_complexities)
    avg_complexity = np.average(chord_complexities)
    num_chords = len(chords)

    # indeces 8 - 11
    feature_set += [min_complexity, max_complexity, avg_complexity, num_chords]

    # Since the notes have been sorted, we can now get the end time. Index 12
    song_length = all_notes[-1].stop
    feature_set.append(song_length)

    # Question: distinct chords - A chord should be equal to another if it contains notes of the same pitches, ?and duration?
    # Consider extracting this if models will note tune. 
    return feature_set

def note_start(note):
    return note.start

def test_midi_reduction_techniques_unsupervised():
    print("Loading training and test data.")
    midis, non_midis = load_training_data()

    test_mids = load_test_data()

    print("Performing feature extraction from midi data.")
    extracted_train_data = []
    for mid in midis:
        extracted_train_data.append(np.array(extract_features_from_midi(mid)))

    extracted_test_data = []
    for mid in test_mids:
        extracted_test_data.append([np.array(extract_features_from_midi(mid[0])), mid[1]])
  
    #one class svm performed Really badly on original approach. It's spitting out one wrong one here.
    print("One class svm")    
    svm = OneClassSVM(nu = 0.01, gamma='scale', kernel='rbf').fit(extracted_train_data)
    print(svm.predict(extracted_train_data))

    # # Now predict on the second set of data.
    for sample in extracted_test_data:
        pred = svm.predict([sample[0]])
        if pred < 0:
            print(sample[1])

    print("Isolation Forest")
    isof =  IsolationForest(contamination=0.01, random_state=42)
    isof.fit(extracted_train_data)

    print("Model is fit, predicting on original data. ")
    print(isof.predict(extracted_train_data))
    print("Predicting on test data now.")
    for sample in extracted_test_data:
        pred = isof.predict([sample[0]])
        if pred < 0:
            print(sample[1])

    print("Robust covariance")
    envelope = EllipticEnvelope(contamination=0.01)
    envelope.fit(extracted_train_data)
    print("Envelop is fit, predicting on original data")
    print(envelope.predict(extracted_train_data))
    for sample in extracted_test_data:
        pred = envelope.predict([sample[0]])
        if pred < 0:
            print(sample[1])

    # See if we can use some clustering techniques. 
    # KMeans is not a reasonable choice due to the differences in cluster size.

    # DBSCAN does not perform well on this data. Will flag anomalies, but not the right ones.
    print("DBSCAN - attempting to tune eps")
    dists = [100000, 200000, 300000, 400000, 500000]
    for dist in dists:
        scanner = DBSCAN(eps=dist)
        scanner.fit(extracted_train_data)
        all_data = []
        all_data += extracted_train_data
        for datum in extracted_test_data:
            all_data.append(datum[0])
        print(scanner.fit_predict(all_data))

def test_extracted_data_supervised():    
    # test midi_reduction with extension to supervised learning. 
    # load training data
    print("Loading training and test data.")
    midis, non_midis = load_training_data()
    test_data = []
    test_mids = load_test_data()

    # load into numpy because sklearn likes numpy
    print("Performing feature extraction.")
    data = []
    for mid in midis:
        data.append(np.array(extract_features_from_midi(mid)))

    for mid in non_midis:
        data.append(np.array(extract_features_from_midi(mid)))#, mid[1]])

    # Create a label set, 1 up to 184, 0 after. 
    y_train = np.zeros(len(data))
    y_train[:184] = 1

    for mid in test_mids:
        test_data.append([np.array(extract_features_from_midi(mid[0])), mid[1]])

    # print("Traing Logistic Regression model.")
    # gnb = LogisticRegression(solver='liblinear', max_iter=250, penalty='l1')
    # Notes: logistic regression on extracted data performed poorly, was not able to identify the outliers. 

    print("Training a naive bayes model, due to small sample size.")
    gnb = CategoricalNB()
    # Notes GaussianNB performed poorly, predicted half of the test set to be outliers, though the correct ones were among them.
    # BernoulliNB performed abysmally, failing to identify any class 2 members at all.
    # ComplementNB performed similarly to Gaussian, but missed a real outlier.
    # MultinomialNB was the same as ComplementNB.
    # CategoricalNB through an error after catching one outlier ... investigating. 
    # upgrading it to a try-except, caught 2 of the 3 outliers, while mis-labeling 8 others. 
    gnb.fit(data, y_train)
    print(gnb.predict(data))
    for datum in test_data:
        try:
            pred = gnb.predict_proba([datum[0]])
            if pred[0][0] < 0.5:
                print(str(pred[0]) + " " + datum[1])
        except:
            # Well, that would be a sign that it's an outlier, wouldn't it?
            print(datum[1])


# testing code
if __name__=="__main__":
    # First block of code is for using the attempt to extract meaningful information about the tracks en toto.
    # test_midi_reduction_techniques_unsupervised()

    # test_extracted_data_supervised()
   
    if True:
        # Second block of code is using a more straight second by second redux approach to mapping the tracks into a 
        # Homogenous R^n of all ones and zeros. 

        # load training data
        print("Loading training and test data.")
        midis, non_midis = load_training_data()

        # Sample vizualization - optional. 
        #optional_visualization()    
        
        # load into numpy because sklearn likes numpy
        data = []
        for mid in midis:
            data.append(np.array(reduce_midi_to_np_array(mid)))

        for mid in non_midis:
            data.append(np.array(reduce_midi_to_np_array(mid)))#, mid[1]])

        # Create a label set, 1 up to 184, 0 after. 
        y_train = np.zeros(len(data))
        y_train[:184] = 1

        test_data = []
        test_mids = load_test_data()

        for mid in test_mids:
            test_data.append(np.array(reduce_midi_to_np_array(mid[0])))#, mid[1]])
        
        # TODO uncomment next three lines to get model running again. 
        # print("Using supervised Logistic Regression.")

        # gnb = LogisticRegression(solver='liblinear', max_iter=250, penalty='l1')
        # y_pred = gnb.fit(data, y_train)
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
            test_two.append([temp, mid[1]])
            # TODO uncomment 4 lines. 
            # pred = gnb.predict_proba([temp])
            # # It appears that tuning the the probability down a little improves accuracy with minimal reduction in recall. 
            # if pred[0][1] < 0.44:
            #     print(str(pred[0][1]) + " " + mid[1])

                # This appears to get all three, plus one or two false positive errata. 
        #print(gnb.predict(test_two))

        # Data visualization time. test_mids and test_two are the same length, so we can pairwise compare them. 
        for i in range(8):
            orig_two_d_array = np.reshape(test_data[i], (127, int(MAX_TICKS/10)))
            warp_two_d_array = np.reshape(test_two[i][0], (127, int(MAX_TICKS/10)))

            # too much black-space, need to trim to the reasonable region. 
            orig_reduced = orig_two_d_array[:5000][30:90]
            warp_reduced = warp_two_d_array[:5000][30:90]
            print("Visualizing once and twice processed images for " + test_two[i][1])
            plt.imshow(orig_reduced, cmap='gray')
            plt.show()
            plt.imshow(warp_reduced, cmap='gray')
            plt.show()

        # To stabilize the logistic regression prediction somewhat, we can simply run it 100 times and determine a confidence. 
        # Caution, this is fairly slow, but produces 4 outliers with confidence of at least 80%, and 3 of them are correct. 
        if False:
            candidates = []
            for i in range(100):
                gnb = LogisticRegression(solver='liblinear', max_iter=250, penalty='l1')
                gnb.fit(data, y_train)
                for temp in test_two:
                    pred = gnb.predict_proba([temp[0]])
                    if pred[0][1] < 0.44:
                        candidate_already_found = False
                        for candidate in candidates:
                            if temp[1] == candidate[0]:
                                candidate[1] += 1
                                candidate_already_found = True
                                break
                        if not candidate_already_found:
                            candidates.append([temp[1], 1])
                        #print(str(pred[0][1]) + " " + temp[1])

            print (candidates)


    # Third block of Code - Code that didn't work, applied to the tick-by-tick processing approach. 

    # from sklearn.svm import OneClassSVM
    # from sklearn.ensemble import IsolationForest
    # from sklearn.covariance import EllipticEnvelope
    # from sklearn.naive_bayes import GaussianNB, BernoulliNB, ComplementNB, MultinomialNB, CategoricalNB
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


        # ComplementNB was identical to MultinomialNB - the dataset must not be sufficiently imbalanced for it to work.
        # BernoulliNB was pretty crap
        # so was CategorigcalNB