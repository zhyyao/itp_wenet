import time
try:
    while True:
        print("Start: "+str(time.ctime()))
        time.sleep(43200)
        print("End: "+str(time.ctime()))
except:
    print("close sleep")

