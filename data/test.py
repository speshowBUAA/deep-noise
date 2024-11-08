import noisedata

NoiseData = noisedata.NoiseData(dir='../../data')
print(NoiseData.__len__())
print(NoiseData.__getitem__(1))

NoiseData = noisedata.NoiseData(dir='../../data',use_type=True)
print(NoiseData.le)
print(NoiseData.__len__())
print(NoiseData.__getitem__(1))