from music21 import *

### Notes
# a `Note` object is made up of `Pitch` and `Duration` objects
f = note.Note("F5")
f.name
f.octave
f.pitch
f.pitch.frequency

bflat = note.Note("B-2")
bflat.pitch.accidental
acc = bflat.pitch.accidental
acc.name
acc.alter

f.show()
f.show('midi')

d = bflat.transpose("M3")

r = note.Rest(type='whole')
r.show()

# Don't call a variable `note`
# it will shadow music21.note
# note = note.Note("C#3")



### Pitches, Durations, and Notes again
p1 = pitch.Pitch('b-4')
p1.octave
p1.name
p1.accidental.alter
p1.nameWithOctave
p1.midi

p1.name = 'd#'
p1.octave = 3
p1.nameWithOctave


p2 = p1.transpose('M7')
p2

csharp = note.Note('C#4')
csharp.pitch.unicodeName
csharp.pitch.getEnharmonic()
csharp.pitch.getLowerEnharmonic()


halfDuration = duration.Duration('half')
dottedQuarter = duration.Duration(1.5)
dottedQuarter.quarterLength
halfDuration.quarterLength
halfDuration.type
dottedQuarter.type

dottedQuarter.dots
dottedQuarter.dots = 2
dottedQuarter.quarterLength

dottedQuarter.quarterLength = 0.25
dottedQuarter.type



n1 = note.Note()
n1.pitch
n1.duration

n1.pitch.nameWithOctave = 'E-5'
n1.duration.quarterLength = 3.0
n1.duration.type
n1.duration.dots
n1.pitch.name
n1.pitch.accidental
n1.octave
n1.name
n1.quarterLength
n1.quarterLength


otherNote = note.Note("F6")
otherNote.lyric = "I'm the Queen of the Night!"

n1.addLyric(n1.nameWithOctave) # multiple lyrics
n1.addLyric(n1.pitch.pitchClassString)
n1.show()
n1.quarterLength = 6.25
n1.show()



### Lists, Streams (I) and Output
note1 = note.Note("C4")
note2 = note.Note("F#4")
noteList = [note1, note2]
note3 = note.Note("B-2")
noteList.append(note3)

for thisNote in noteList:
    print(thisNote.step)
noteList[0]
noteList[-1]
noteList.index(note2)



# The Stream object and its subclasses(Score, Part,
# Measure) are the fundamental containers for music21
# objects such as Note, Chord, Clef, TimeSignature objects.
stream1 = stream.Stream()
stream1.append(note1)
stream1.append(note2)
stream1.append(note3)

stream2 = stream.Stream()
n3 = note.Note('D#5')
stream2.repeatAppend(n3, 4)
stream2.show()

len(stream1)
stream1.show('text')
stream1.show()

for thisNote in stream1:
    print(thisNote.step)

stream1[0]
note3Index = stream1.index(note3)
stream1.pop(note3Index)
stream1.append(note3)

for thisNote in stream1.getElementsByClass(note.Note):
    print(thisNote, thisNote.offset)
for thisNote in stream1.getElementsByClass("Note"):
    print(thisNote, thisNote.offset)
for thisNote in stream1.getElementsByClass(["Note", "Rest"]):
    print(thisNote, thisNote.offset)

for thisNote in stream1.notes:
    print(thisNote)
for thisNote in stream1.notesAndRests:
    print(thisNote)
listOut = stream1.pitches
listOut

stream1.analyze('ambitus')
stream1.show('midi')

defaults.meterNumerator
defaults.meterDenominator



biggerStream = stream.Stream()
note2 = note.Note("D#5")
biggerStream.insert(0, note2)

biggerStream.append(stream1)
biggerStream.show('text')

note1 in stream1
note1 in biggerStream



### Streams (II): Hierarchies, Recursion, and Flattening

# A common arrangement of nested Streams is a Score Stream
# containing one or more Part Streams, each Part Stream in
# turn containing one or more Measure Streams.
sBach = corpus.parse('bach/bwv57.8')

# the Score which has six elements: a Metadata object,
# a StaffGroup object, and four Part objects
len(sBach)

# Then we find the length of first Part at index one which
# indicates 19 objects (18 of them are measures).
len(sBach[1])

# Within that part we find an object (a Measure) at index 1
len(sBach[1][1])

# More than just Measures might be stored in a Part object
# (such as Instrument objects), and more than just Note and
#  Rest objects might be stored in a Measure (such as
# TimeSignature and KeySignature objects). Therefore,
# it’s much safer to filter Stream and Stream subclasses
# by the class we seek.
len(sBach.getElementsByClass(stream.Part))
len(sBach.getElementsByClass(stream.Part)[0].getElementsByClass(stream.Measure))
len(sBach.getElementsByClass(stream.Part)[0].getElementsByClass(
    stream.Measure)[1].getElementsByClass(note.Note))

len(sBach.getElementsByClass('Part'))
len(sBach.parts)

alto = sBach.parts[1] # parts count from zero, so soprano is 0 and alto is 1
excerpt = alto.measures(1, 4)
excerpt.show()
measure2 = alto.measure(2) # measure not measure_s_
measure2.show()
measureStack = sBach.measures(2, 3)
measureStack.show()



s = stream.Score(id='mainScore')
p0 = stream.Part(id='part0')
p1 = stream.Part(id='part1')
m01 = stream.Measure(number=1)
m01.append(note.Note('C', type="whole"))
m02 = stream.Measure(number=2)
m02.append(note.Note('D', type="whole"))
p0.append([m01, m02])
m11 = stream.Measure(number=1)
m11.append(note.Note('E', type="whole"))
m12 = stream.Measure(number=2)
m12.append(note.Note('F', type="whole"))
p1.append([m11, m12])
s.insert(0, p0)
s.insert(0, p1)
s.show('text')

for el in s.recurse():
    print(el.offset, el, el.activeSite)

for el in s.flat:
    print(el.offset, el, el.activeSite)

len(sBach.getElementsByClass(note.Note))
len(sBach.flat.getElementsByClass(note.Note))
print(len(sBach.flat.getElementsByClass(instrument.Instrument)))
for p in sBach.getElementsByClass(instrument.Instrument):
    print(p)