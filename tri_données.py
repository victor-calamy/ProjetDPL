import pandas as pd
import shutil

df = pd.read_csv("C:/Users/victo/Desktop/devoirsCentrale/DPL/projetDLE/train_curated.csv")
newDf = ["fname","labels"]
path = "C:/Users/victo/Desktop/devoirsCentrale/DPL/projetDLE/"
liste_labels = ["Bus","Motorcycle","Traffic_noise_and_roadway_noise","Accelerating_and_revving_and_vroom"]
for k in range (len(df)):
    if (df['labels'][k] in liste_labels):
        nom = df["fname"][k]
        newDf.append(df.values.tolist()[k])
        
        shutil.copy(path + "wav/" + nom, path + "5label")

newDf = pd.DataFrame(newDf)
newDf.to_csv(path + "out.csv")