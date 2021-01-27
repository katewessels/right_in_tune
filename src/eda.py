import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import json
plt.style.use('ggplot')
import seaborn as sns
sns.set()
pd.set_option('display.max_columns', None)


#box plot
def box_plot(list_of_lists, title, x_tick_label_list, y_label, ax):
    ax.boxplot(list_of_lists)
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(x_tick_label_list, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    plt.xticks(fontsize=16, rotation=45)
    plt.savefig(f'images/{y_label}_boxplot.png', bbox_inches='tight')

if __name__ == "__main__":
    # #get validation data
    # json_file_path = "/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-valid/valid_examples.json"
    # with open(json_file_path, 'r') as j:
    #     contents_valid = json.loads(j.read())
    # df_valid = pd.DataFrame(contents_valid).T.reset_index()
    # df_valid = df_valid.drop(columns=['index'])
    # df_valid.to_csv('data/valid_metadata.csv', index=None)
    df_valid = pd.read_csv('data/valid_metadata.csv')

    # #get test data
    # json_file_path = "/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-test/test_examples.json"
    # with open(json_file_path, 'r') as j:
    #     contents_test = json.loads(j.read())
    # df_test = pd.DataFrame(contents_test).T.reset_index()
    # df_test = df_test.drop(columns=['index'])
    # df_test.to_csv('data/test_metadata.csv', index=None)
    df_test = pd.read_csv('data/test_metadata.csv')

    # #get train data
    # json_file_path = "/Users/katewessels/Documents/capstones/right_in_tune/data/nsynth-train/train_examples.json"
    # with open(json_file_path, 'r') as j:
    #     contents = json.loads(j.read())
    # df = pd.DataFrame(contents).T.reset_index()
    # df = df.drop(columns=['index'])
    # df.to_csv('data/training_metadata.csv', index=None)
    df = pd.read_csv('data/training_metadata.csv')

    #EDA ON TRAINING DATA
    #instrument, source, pitch counts
    instrument_value_counts = df['instrument_family'].value_counts()
    instrument_series = df['instrument_family_str'].value_counts()
    source_series = df['instrument_source_str'].value_counts()
    pitch_series = df['pitch'].value_counts()
    velocity_series = df['velocity'].value_counts()

    #plot audio samples by source
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(source_series.index, source_series.values)
    ax.set_title('Audio Samples by Source Type', fontsize=16)
    plt.xticks(fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    plt.savefig('images/counts_by_source_type.png', bbox_inches='tight')

    #plot audio samples by instrument type
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(instrument_series.index, instrument_series.values)
    ax.set_title('Audio Samples by Instrument Type', fontsize=16)
    plt.xticks(fontsize=16, rotation=45)
    ax.set_ylabel('Count', fontsize=16)
    plt.savefig('images/counts_by_instrument_type.png', bbox_inches='tight')

    #electronic value counts by instrument
    instrument_electronic = df['instrument_family_str'][df['instrument_source_str']
                        =='electronic'].value_counts().reindex(df['instrument_family_str'].unique(),
                        fill_value=0).sort_index()
    #synthetic value counts by instrument
    instrument_synthetic = df['instrument_family_str'][df['instrument_source_str']==
                        'synthetic'].value_counts().reindex(df['instrument_family_str'].unique(),
                        fill_value=0).sort_index()
    #acoustic value counts by instrument
    instrument_acoustic = df['instrument_family_str'][df['instrument_source_str']==
                    'acoustic'].value_counts().reindex(df['instrument_family_str'].unique(),
                        fill_value=0).sort_index()

    #plot instrument family breakdown by source
    labels = instrument_acoustic.index
    acoustic = instrument_acoustic.values
    electronic = instrument_electronic.values
    synthetic = instrument_synthetic.values

    width = 0.7
    fig, ax = plt.subplots(figsize=(15,8))
    ax.bar(labels, acoustic, width, label='acoustic')
    ax.bar(labels, electronic, width, bottom=acoustic,
        label='electronic')
    ax.bar(labels, synthetic, width, bottom=electronic+acoustic,
        label='synthetic')
    plt.xticks(fontsize=16, rotation=45)
    plt.yticks(fontsize=14)
    ax.set_ylabel('Counts', fontsize=16)
    ax.set_title('Audio Sample Instrument Counts by Source Type', fontsize=16)
    ax.legend(fontsize=16)
    plt.savefig('images/instrument_by_source.png', bbox_inches='tight')

    #plot audio samples by pitch
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(pitch_series.index, pitch_series.values)
    ax.set_title('Audio Samples by Pitch, 0-based MIDI Pitch [0, 127]', fontsize=16)
    plt.xticks(fontsize=16, rotation=45)
    ax.set_ylabel('Count', fontsize=16)
    plt.savefig('images/counts_by_pitch.png', bbox_inches='tight')

    #plot audio samples by velocity
    fig, ax = plt.subplots(figsize=(12,8))
    ax.bar(velocity_series.index, velocity_series.values, width=15)
    ax.set_title('Audio Samples by Velocity, 0-based MIDI Velocity [0, 127]', fontsize=16)
    plt.xticks([25, 50, 75, 100, 127], fontsize=14)
    ax.set_xlabel('Velocity', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    plt.savefig('images/counts_by_velocity.png', bbox_inches='tight')

    #plot scatter plot of audio samples pitch counts by instrument
    groups = df.groupby('instrument_family_str')
    fig, ax = plt.subplots(figsize=(16,8))
    for name, group in groups:
        ax.scatter(group['pitch'].value_counts().index, group['pitch'].value_counts().values, label=name)
    ax.set_xlabel("Pitch", fontsize=16)
    ax.set_ylabel('Counts', fontsize=16)
    ax.set_title('Audio Samples Pitch Counts by Instrument, 0-based MIDI Pitch [0, 127]', fontsize=16)
    plt.legend(fontsize=14)
    plt.savefig('images/scatter_pitch_by_instrument.png', bbox_inches='tight')

    #box plot of pitch distributions by instrument
    fig, ax = plt.subplots(figsize=(15,8))
    list_of_lists = [list(df[df['instrument_family_str']=='bass']['pitch']),
                    list(df[df['instrument_family_str']=='brass']['pitch']),
                    list(df[df['instrument_family_str']=='flute']['pitch']),
                    list(df[df['instrument_family_str']=='guitar']['pitch']),
                    list(df[df['instrument_family_str']=='keyboard']['pitch']),
                    list(df[df['instrument_family_str']=='mallet']['pitch']),
                    list(df[df['instrument_family_str']=='organ']['pitch']),
                    list(df[df['instrument_family_str']=='reed']['pitch']),
                    list(df[df['instrument_family_str']=='string']['pitch']),
                    list(df[df['instrument_family_str']=='synth_lead']['pitch']),
                    list(df[df['instrument_family_str']=='vocal']['pitch'])
                    ]
    x_tick_label_list = ['Bass', 'Brass', 'Flute', 'Guitar', 'Keyboard',
                        'Mallet', 'Organ', 'Reed', 'String', 'Synth_lead', 'Vocal']
    box_plot(list_of_lists, 'Audio Samples Pitch Distribution by Instrument, 0-based MIDI Pitch [0, 127]',
            x_tick_label_list, 'Pitch', ax=ax)


    #box plot of pitch distributions by source
    fig, ax = plt.subplots(figsize=(8,8))
    list_of_lists = [list(df[df['instrument_source_str']=='acoustic']['pitch']),
                    list(df[df['instrument_source_str']=='electronic']['pitch']),
                    list(df[df['instrument_source_str']=='synthetic']['pitch'])
                    ]
    x_tick_label_list = ['Acoustic', 'Electronic', 'Synthetic']
    box_plot(list_of_lists, 'Audio Samples Pitch Distribution by Source, 0-based MIDI Pitch [0, 127]',
            x_tick_label_list, 'Pitch', ax=ax)
