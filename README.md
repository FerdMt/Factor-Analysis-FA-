# Factor-Analysis-FA-

#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from pandas import Series, DataFrame

df = pd.read_csv("merged_data.csv")
df = df.rename(columns={'Disposal': 'DSP','Drained':'DRN','Electricity':'ELC','Fertilizers':'FRT','Fires':'FRS','Forestland':'FRL','Household':'HSL','Packaging':'PCG','Pesticides':'PTC','Processing':'PRC','Retail':'RTL','Transport':'TRS','conversion':'CNV','energy':'ENG'})
print(df)

X = StandardScaler().fit_transform(df)
factors = 2

fas = [
    ("FA no rotation", FactorAnalysis(n_components=factors)),
    ("FA varimax", FactorAnalysis(n_components=factors, rotation="varimax"))
]

fig, axes = plt.subplots(ncols=len(fas),figsize=(10,8))

for ax, (title, fa) in zip(axes, fas):
    #  Fit the model to the standardized food data
    fa = fa.fit(X)
    #  and transpose the component (loading) matrix
    factor_matrix = fa.components_.T
    #  Plot the data as a heat map
    im = ax.imshow(factor_matrix, cmap="RdBu_r", vmax=1, vmin=-1)
    #  and add the corresponding value to the center of each cell
    for (i,j), z in np.ndenumerate(factor_matrix):
        ax.text(j, i, str(z.round(2)), ha="center", va="center")
    #  Tell matplotlib about the metadata of the plot
    ax.set_yticks(np.arange(len(df.columns)))
    if ax.get_subplotspec().is_first_col():
        ax.set_yticklabels(df.columns)
    else:
        ax.set_yticklabels([])
    ax.set_title(title)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Factor 1", "Factor 2"])
    #  and squeeze the axes tight, to save space
    plt.tight_layout()

#  and add a colorbar
cb = fig.colorbar(im, ax=axes, location='right', label="loadings")
plt.show()

#  show us the plot
fa = FactorAnalysis(n_components = 2, rotation="varimax")
fa.fit(X)
uniqueness = Series(fa.noise_variance_, index=df.columns)
uniqueness.plot(
    kind="bar",
    ylabel="Uniqueness"
)
plt.show()
lambda_ = fa.components_
psi = np.diag(uniqueness)
s = np.corrcoef(np.transpose(X))
sigma = np.matmul(lambda_.T, lambda_) + psi
residuals = (s - sigma)

ax = plt.axes()
im = ax.imshow(residuals, cmap="RdBu_r", vmin=-1, vmax=1)
ax.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
ax.set_xticks(range(14))
ax.set_xticklabels(df.columns)
ax.set_yticks(range(14))
ax.set_yticklabels(df.columns)
for (i,j), z in np.ndenumerate(residuals):
    ax.text(j, i, str(z.round(3)), ha="center", va="center")

fig.colorbar(im, ax=ax, location='right')
ax.set_title("FA residual matrix")
plt.tight_layout()
plt.show()

methods = [
    ("FA No rotation", FactorAnalysis(2,)),
    ("FA Varimax", FactorAnalysis(2, rotation="varimax")),
    ("FA Quartimax", FactorAnalysis(2, rotation="quartimax")),
]
fig, axes = plt.subplots(ncols=3, figsize=(10, 8), sharex=True, sharey=True)

for ax, (method, fa) in zip(axes, methods):
    fa = fa.fit(X)

    components = fa.components_

    vmax = np.abs(components).max()
    ax.scatter(components[0,:], components[1, :])
    ax.axhline(0, -1, 1, color='k')
    ax.axvline(0, -1, 1, color='k')
    for i,j, z in zip(components[0, :], components[1, :], df.columns):
        ax.text(i+.02, j+.02, str(z), ha="center")
    ax.set_title(str(method))
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Factor 1")
    ax.set_xlabel("Factor 2")
print(fa.get_covariance())
plt.tight_layout()
plt.show()
