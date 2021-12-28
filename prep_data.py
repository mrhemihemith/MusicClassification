import json
import librosa
import os

DATASET_PATH = "Voices"
JSON_PATH  = "data_sample.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec. of audio
DURATION = 30
SAMPLES_PER_TRACK = SAMPLES_TO_CONSIDER * DURATION

def prepare_dataset(dataset_path,json_path,n_mfcc=13,hop_length=512,n_fft=2048):

    #data dictionary
    data = {
        "mappings" : [],
        "Name" : [],
        "files" : [],
        "MFCCs" : []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensure that we are not at the root level
        if dirpath is not dataset_path:
            # update mapping
            category = dirpath.split("/")[-1]  # dataset/down => [dataset,down]
            data["mappings"].append(category)
            print(f"Processing{category}")

            # loop through all the filenames and extract MFCCs
            for f in filenames:
                # get file path
                file_path = os.path.join(dirpath, f)

                # load the audio file
                signal, sr = librosa.load(file_path)

                # ensure that the audio file is atleast 30 sec
                if len(signal) >= SAMPLES_PER_TRACK:
                    # enforce 30 sec , long signal
                    signal = signal[:SAMPLES_PER_TRACK]

                    # extract mfcc
                    MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                    # store data
                    data["Name"].append(i - 1)  # because dataset =(i=0)#
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["files"].append(file_path)
                    print(f"{file_path} : {i - 1}")

    # STORE IN A JSON FILE
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)



if __name__ == '__main__':
    prepare_dataset(DATASET_PATH,JSON_PATH)
