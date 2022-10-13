"""
Module made to help identify errored trials and to document
exclusions based on some predetermined criteria.
"""
import os

def exc_file_exists(file = '_excluded.txt'):
    files = os.listdir()
    return any([file in x for x in files])

def make_exc_file(file = '_excluded.txt'):
    if not exc_file_exists(file):
        stream = open(file, 'w')
        stream.close()
        print(f"Created file {file}")

def is_excluded(*args, file = '_excluded.txt'):
    """
    Checks to see if the specific information is in
    the file tracking exclusions.
    """
    # checking to see if file exists, or doing nothing if found
    make_exc_file(file)
    
    # setting criteria
    crit = list(args)
    crit = [str(x) for x in crit]

    # empty string being appended to search file for
    # search criteria
    emp = str()
    for c in crit:
        emp += c + "\t"

    # conducting search and return whether or not the search was found
    stream = open(file, 'r')
    already = any([emp in line for line in stream])
    stream.close()
    return already


def exclude(*args, file = '_excluded.txt'):
    """
    Places criteria in the exclusion file, if not already there.
    """
    if not is_excluded(*args, file = file):
        crit = list(args)
        stream = open(file, 'a')
        for c in crit:
            stream.write(str(c) + "\t")
        stream.write("\n")
        stream.close()
        print(f"Wrote line to file: \n\t{args}")

        
def clear_exclusions(file = "_excluded.txt"):
    """
    Clears the exclusion file
    """
    stream = open(file, "w")
    stream.close()


