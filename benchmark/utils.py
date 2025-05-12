import os

def get_badSegment(folder_directory):
    """
    This function gets the filenames of all the bad segments in the given directory,
    Then saves them into a text file called bad_segments.txt.
    :param folder_directory:
    :return:
    """
    bad_segments = []
    for bad_files in os.listdir(folder_directory):
        if bad_files.lower().endswith('mp4'):
            bad_segments.append(bad_files)
    with open('bad_segments.txt', 'w') as f:
        for bad_file in bad_segments:
            f.write(bad_file + '\n')
    print(f"Bad segments saved to bad_segments.txt")

if __name__ == '__main__':
    folder_directory = '/home/samueladebayo/Documents/PhD/QUBPHEO/corrupted-segment'
    get_badSegment(folder_directory)


