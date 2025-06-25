import pandas as pd

df = pd.read_xml("/Users/tommoore/Documents/GitHub/Research/P001/HealthApp/Raw/P001 export.xml")
df.to_csv("/Users/tommoore/Documents/GitHub/Research/P001/HealthApp/Labeled/LabeledTest.csv", index=False)