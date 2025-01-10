start /B python test.py --snapshot .\snapshots\1nc100_epoch_400.pth -—nc 100 >> test11.txt
start /B python test.py --snapshot .\snapshots\1nc200_epoch_400.pth -—nc 200 >> test12.txt
start /B python test.py --snapshot .\snapshots\1nc400_epoch_400.pth -—nc 400 >> test13.txt
start /B python test.py --snapshot .\snapshots\1nc800_epoch_400.pth -—nc 800 >> test14.txt
start /B python test.py --snapshot .\snapshots\1nc1600_epoch_400.pth -—nc 1600 >> test15.txt
start /B python test.py --snapshot .\snapshots\1nc1320_epoch_400.pth -—nc 3200 >> test16.txt