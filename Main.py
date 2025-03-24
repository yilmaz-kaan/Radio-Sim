from RadioClass import RadioModel, Modulator, Channel, Equalizer

# Initialize RadioModel class. Doing this also intializes a modulator, channel, and equalizer
test = RadioModel('bpsk', [1, 0.5 ,0.1], 0.2)
test.modulator.set_modulation('qpsk')

test.summary()