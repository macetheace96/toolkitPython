from __future__ import (absolute_import, division, print_function, unicode_literals)

import re
import numpy as np

MISSING = float('inf')


def load_arff(filename):
	"""Load matrix from an ARFF file"""
	data = []
	attr_names = []
	str_to_enum = []
	enum_to_str = []
	reading_data = False

	rows = []  # we read data into array of rows, then convert into array of columns

	f = open(filename)
	for line in f.readlines():
		line = line.rstrip()
		if len(line) > 0 and line[0] != '%':
			if not reading_data:
				if line.lower().startswith("@relation"):
					dataset_name = line[9:].strip()
				elif line.lower().startswith("@attribute"):
					attr_def = line[10:].strip()
					if attr_def[0] == "'":
						attr_def = attr_def[1:]
						attr_name = attr_def[:attr_def.index("'")]
						attr_def = attr_def[attr_def.index("'") + 1:].strip()
					else:
						search = re.search(r'(\w*)\s*(.*)', attr_def)
						attr_name = search.group(1)
						attr_def = search.group(2)
						# Remove white space from atribute values
						attr_def = "".join(attr_def.split())

					attr_names += [attr_name]

					str_to_enum = {}
					enum_to_str = {}
					if not (
							attr_def.lower() == "real" or attr_def.lower() == "continuous" or attr_def.lower() == "integer"):
						# attribute is discrete
						assert attr_def[0] == '{' and attr_def[-1] == '}'
						attr_def = attr_def[1:-1]
						attr_vals = attr_def.split(",")
						val_idx = 0
						for val in attr_vals:
							val = val.strip()
							enum_to_str[val_idx] = val
							str_to_enum[val] = val_idx
							val_idx += 1

					enum_to_str.append(enum_to_str)
					str_to_enum.append(str_to_enum)

				elif line.lower().startswith("@data"):
					reading_data = True

			else:
				# reading data
				row = []
				val_idx = 0
				# print("{}".format(line))
				vals = line.split(",")
				for val in vals:
					val = val.strip()
					if not val:
						raise Exception("Missing data element in row with data '{}'".format(line))
					else:
						row += [float(MISSING if val == "?" else str_to_enum[val_idx].get(val, val))]

					val_idx += 1

				rows += [row]

	f.close()
	return rows