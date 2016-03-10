import json
import sys

in_loc = sys.argv[1]
out_loc = sys.argv[2]

data = json.load(open(in_loc))
json.dump(data, open(out_loc, 'w'), indent=2)
