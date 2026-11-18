import csv
from .. import LinkStream

def from_csv(path, weighted=False):
	""" Read a link stream from a csv file 

	:param path: Absolute path of CSV file
	:param weighted: 'True' if triplets are weighted
	:type path: string
	:type weighted: Boolean, defaults to 'False'
	:type linkstream: class:'LinkStream', defaults to None
	:return: A link stream object of class class:'LinkStream'
	"""
	# Create link stream 
	linkstream = LinkStream()

	# Read csv file
	with open(path, 'r') as file: 
		reader = csv.reader(file, delimiter=',')
		for line in reader:
			# Add triplet (u, v,t)
			triplet = (line[0], line[1], int(line[2]))
			linkstream.add_triplet(triplet)
			# If weighted, add it to triplets
			if weighted: linkstream.set_triplet_weight(triplet, float(line[3]))

	linkstream.isweighted = weighted
	linkstream.set_stats()
	return linkstream


def to_csv(path: str, linkstream: LinkStream, separator: str = ","):
    """Write a link stream to a csv file.

    Format (no header):

        * Column 1: source
        * Column 2: destination
        * Column 3: time
        * Column 4: weight (if `linkstream.isweighted() == True`)

    :param path: absolute file path
    :param linkstream: linkstream object to save
    :param separator: field separator
    """
    if linkstream.isweighted:
        to_write = [
            (e[0], e[1], e[2], v) for e, v in linkstream.to_weightfn().items()
        ]
    else:
        to_write = [e for e in linkstream.to_triplets()]
    
    to_write = sorted(to_write, key=lambda x: x[2])
    
    # Write csv file
    with open(path, "w") as file:
        writer = csv.writer(file, delimiter=separator)
        for row in to_write:
            writer.writerow(row)

    file.close()
    return None
