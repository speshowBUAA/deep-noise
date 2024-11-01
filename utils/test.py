import transform

x = [542,150,3256]
transformations = transform.Normalizer(mean=[354.16, 32.17, 2649.37], std=[187.5, 647.17, 2045.62])

print(transformations.transform(x))