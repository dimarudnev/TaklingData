import pandas as pd
import os
from collections import defaultdict
import json
import time

datadir = 'D:\\Data\\TalkingData'

print("start")

gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'), index_col='device_id')
d = defaultdict(lambda: defaultdict(int))

appEventsIter = pd.read_csv(os.path.join(datadir,'app_events.csv'), index_col='event_id', chunksize = 100000)
events = pd.read_csv(os.path.join(datadir,'events.csv'), index_col='event_id')

chunkIndex=0;
print(chunkIndex)
for appEventChunk in appEventsIter:
    for eventId, appEventInfo in appEventChunk.iterrows():
        deviceId = events.ix[eventId]['device_id']
        if deviceId in gatrain.index:
            appId = appEventInfo['app_id']
            d[str(deviceId)][str(appId)] += 1
    print(chunkIndex)
    chunkIndex += 1
    if chunkIndex == 20:
        break;
with open(os.path.join(datadir,'event_count_per_device.json'), 'w') as f:
    json.dump(d, f);
print("ready")

    





