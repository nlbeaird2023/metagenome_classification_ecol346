"""
Recreates visual in Random Forest homework assignment.

Requires:
Seaborn, numpy, matplotlib, pandas
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    np.random.seed(42)

    # simulate 50 healthy samples
    methylation_healthy = np.random.normal(loc=0.4, scale=0.07, size=50)
    acetylation_healthy = np.random.normal(loc=0.6, scale=0.07, size=50)

    # simulate 50 diseased samples (less methylation, more acetylation)
    methylation_diseased = np.random.normal(loc=0.55, scale=0.07, size=50)
    acetylation_diseased = np.random.normal(loc=0.45, scale=0.07, size=50)

    data = pd.DataFrame({
        'methylation': np.concatenate([methylation_healthy, methylation_diseased]),
        'acetylation': np.concatenate([acetylation_healthy, acetylation_diseased]),
        'label': ['Healthy']*50 + ['Diseased']*50
                        })
    # trims to boundary
    data['methylation'] = data['methylation'].clip(0, 1)
    data['acetylation'] = data['acetylation'].clip(0, 1)

    # Visualize
    sns.scatterplot(data=data, x='methylation', y='acetylation', hue='label', palette='Set1')
    plt.xlabel("DNA Methylation Level")
    plt.ylabel("Histone Acetylation Level")
    plt.show()

if __name__ == "__main__":
    main()