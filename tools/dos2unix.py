#!/usr/bin/env python
"""\
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py <input> <output>
"""
def pkl_formatting(pickle_original_file_path):
    from os.path import exists as path_exists

    original = pickle_original_file_path
    destination = pickle_original_file_path.rsplit('.pkl', 1)[0] + '_unix.pkl'

    if not path_exists(destination):
        if path_exists(original):
            print(destination, " file does not exist.")
            print("Generating the pickle file using %s..." % (original))
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
    else:
        pass
