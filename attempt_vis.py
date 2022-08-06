from visual_midi import Plotter
from visual_midi import Preset
from pretty_midi import PrettyMIDI

# Loading a file on disk using PrettyMidi, and show
pm = PrettyMIDI("composerMidis/musicnet_midis/Bach/2186_vs6_1.mid")
plotter = Plotter()

plotter.save(pm, "example-01.html")

plotter.show(pm, "example-01.html")
#plotter.show_notebook(pm)

# Converting to PrettyMidi from another library, like Magenta note-seq
# import magenta.music as mm
# pm = mm.midi_io.note_sequence_to_pretty_midi(sequence)
# plotter = Plotter()
# plotter.show(pm, "/tmp/example-02.html")