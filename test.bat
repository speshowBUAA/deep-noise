start /B python test_fft.py --snapshot .\snapshots\4_100_epoch_400.pth -—nc 100 >> test11.txt
start /B python test_fft.py --snapshot .\snapshots\4_200_epoch_400.pth -—nc 200 >> test12.txt
start /B python test_fft.py --snapshot .\snapshots\4_400_epoch_400.pth -—nc 400 >> test13.txt
start /B python test_fft.py --snapshot .\snapshots\4_800_epoch_400.pth -—nc 800 >> test14.txt
start /B python test_fft.py --snapshot .\snapshots\4_1600_epoch_400.pth -—nc 1600 >> test15.txt
start /B python test_fft.py --snapshot .\snapshots\4_1320_epoch_400.pth -—nc 3200 >> test16.txt