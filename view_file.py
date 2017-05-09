import pprint 
import json


with open('train-v1.1.json') as f:
	raw_data = f.read()	

# with open('dev-v1.1.json', 'r', encoding='utf-8') as f:
# 	raw_data = f.read()

# with open('rms1atte128dev-prediction.json') as f:
# 	raw_data = f.read()

pprint.pprint(json.loads(raw_data))

