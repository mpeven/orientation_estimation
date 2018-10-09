import glob
import re
import pandas as pd

print("Displaying test accuracies")
results = []
for csv in glob.glob("outputs/test*"):
    df = pd.read_csv(csv)
    close_matches = [
        df['orientation'] == df['predicted_orientation'],
        df['orientation'] == (df['predicted_orientation'] - 1),
        df['orientation'] == (df['predicted_orientation'] + 1),
        df['orientation'] == (df['predicted_orientation'] - 35),
        df['orientation'] == (df['predicted_orientation'] + 35),
        df['orientation'] == (df['predicted_orientation'] - 2),
        df['orientation'] == (df['predicted_orientation'] + 2),
        df['orientation'] == (df['predicted_orientation'] - 34),
        df['orientation'] == (df['predicted_orientation'] + 34),
    ]
    results.append({
        'epoch': int(re.search('epoch_(\d+)', csv).group(1)),
        'accuracy': sum(df['orientation'] == df['predicted_orientation'])/len(df) * 100,
        'accuracy_expanded': sum(sum(x) for x in close_matches)/len(df) * 100,
    })

print(pd.DataFrame(results).sort_values("accuracy_expanded", ascending=False).head(20))
