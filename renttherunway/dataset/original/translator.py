import csv
import pandas as pd
import json

df = pd.read_json('renttherunway_final_data.json', lines=True)

df.to_csv("renttherunway_final_data.csv", mode = 'w')