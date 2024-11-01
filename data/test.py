import data

NoiseData = data.NoiseData(dir='../../data')
print(NoiseData.__len__())
print(NoiseData.__getitem__(1))

NoiseData = data.NoiseData(dir='../../data',use_type=True)
print(NoiseData.le)
print(NoiseData.__len__())
print(NoiseData.__getitem__(1))