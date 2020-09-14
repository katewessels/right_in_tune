# right_in_tune

"That melody is beautiful." "This baseline is funky and crisp." "And the fiddle riff is off the chain..." 

Ever hear a song you like and want to play along? If you play an instrument, you likely have had a thought like this before. What was that note again? How did they play that part? Let me listen again...

Music transcription is the practice of notating a piece of music, creating sheet music by writing down the notes, in music notation, that make up the piece. Music transcription can be challenging and time consuming, especially for an amateur musician. This is a project exploring music transcription through machine learning techniques. 

This project will use the labeled magenta NSynth Dataset, https://magenta.tensorflow.org/datasets/nsynth#files, an audio dataset containing 305,979  four-second audio samples of musical notes, each with a unique pitch, timbre and envelop. The dataset includes audio samples from 1,006 unique instruments, ranging over every pitch of a standard MIDI piano (21-108) as well as five different velocities (25, 50, 75, 100, 127). Some instruments were not capable of producing all 88 pitches in this range, resulting in an average of 65.4 pitches per unique instrument. 

The goal of this project is to build a machine learning model to predict the pitch of a note, given a four-second audio sample from the dataset described above.  
