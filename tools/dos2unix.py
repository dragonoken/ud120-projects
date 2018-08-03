#!/usr/bin/env python
"""\
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py <input> <output>
"""
def pkl_formatting():
    from os.path import exists as path_exists

    original = "../tools/word_data.pkl"
    destination = "../tools/word_data_unix.pkl"

    if not path_exists(destination):
        if path_exists(original):
            print("word_data_unix.pkl file does not exist.")
            print("Generating the pickle file using word_data.pkl...")
            content = ''
            outsize = 0
            with open(original, 'rb') as infile:
              content = infile.read()
            with open(destination, 'wb') as output:
              for line in content.splitlines():
                outsize += len(line) + 1
                output.write(line + str.encode('\n'))
            print("Done. Saved %s bytes." % (len(content)-outsize))
            print()
        else:
            print("Cannot find any required files")
