import noisedata
import plotly.express as px
import pandas as pd

# NoiseData = noisedata.NoiseDataBin(dir='../../data')
# print(NoiseData.__len__())
# print(NoiseData.__getitem__(1))

# NoiseData = noisedata.NoiseDataBin(dir='../../data', use_type=True)
# print(NoiseData.__len__())
# print(NoiseData.__getitem__(1))

NoiseData = noisedata.NoiseDataFFT(dir='../../data', use_type=True)
print(NoiseData.__len__())
print(NoiseData.__getitem__(1))

# fft_df = pd.DataFrame()
# for i in [1,2,3,4]:
#     _, _, fft = NoiseData.__getitem__(i)
#     fft_df = fft_df._append(fft)
# fig = px.line(fft_df.transpose())
# fig.update_layout(yaxis=dict(tickformat=".2e"), xaxis=dict(tickformat="%d"), plot_bgcolor="#fff")
# fig.write_html('fft.html', full_html=False, include_plotlyjs='cdn')
