import pandas as pd
import numpy as np
import os
import shutil
from tqdm import tqdm
from glob import glob
import librosa
import warnings

warnings.filterwarnings("ignore")

sample_submission = pd.read_csv("A:\study\en_voice\sample_submission.csv")

africa_train_paths = glob("A:\study\en_voice/train/africa/*.wav")
australia_train_paths = glob("A:\study\en_voice/train/australia/*.wav")
canada_train_paths = glob("A:\study\en_voice/train/canada/*.wav")
england_train_paths = glob("A:\study\en_voice/train/england/*.wav")
hongkong_train_paths = glob("A:\study\en_voice/train/hongkong/*.wav")
us_train_paths = glob("A:\study\en_voice/train/us/*.wav")

path_list = [africa_train_paths, australia_train_paths, canada_train_paths,
             england_train_paths, hongkong_train_paths, us_train_paths]


# glob로 test data의 path를 불러올때 순서대로 로드되지 않을 경우를 주의해야 합니다.
# test_ 데이터 프레임을 만들어서 나중에 sample_submission과 id를 기준으로 merge시킬 준비를 합니다.

def get_id(data):
    return np.int(data.split("\\")[4].split(".")[0])
test = 'A:\\study\\en_voice\\test\\*.wav'
print(test.split("\\")[4].split(".")[0])

test_ = pd.DataFrame(index = range(0, 6100), columns = ["path", "id"])
test_["path"] = glob("A:\\study\\en_voice\\test\\*.wav")
print('path : ',test_["path"])
test_["id"] = test_["path"].apply(lambda x : get_id(x))
print('id : ',test_["id"])

test_.head()

def load_data(paths):
    
    result = []
    for path in tqdm(paths):
        # sr = 16000이 의미하는 것은 1초당 16000개의 데이터를 샘플링 한다는 것입니다.
        data, sr = librosa.load(path, sr = 18000)
        result.append(data)
    result = np.array(result) 
    # 메모리가 부족할 때는 데이터 타입을 변경해 주세요 ex) np.array(data, dtype = np.float32)

    return result

# train 데이터를 로드하기 위해서는 많은 시간이 소모 됩니다.
# 따라서 추출된 정보를 npy파일로 저장하여 필요 할 때마다 불러올 수 있게 준비합니다.

# os.mkdir("A:\study\en_voice/npy_data")

africa_train_data = load_data(africa_train_paths)
np.save("A:\study\en_voice/npy_data/africa18_npy", africa_train_data)

australia_train_data = load_data(australia_train_paths)
np.save("A:\study\en_voice/npy_data/australia18_npy", australia_train_data)

canada_train_data = load_data(canada_train_paths)
np.save("A:\study\en_voice/npy_data/canada18_npy", canada_train_data)

england_train_data = load_data(england_train_paths)
np.save("A:\study\en_voice/npy_data/england18_npy", england_train_data)

hongkong_train_data = load_data(hongkong_train_paths)
np.save("A:\study\en_voice/npy_data/hongkong18_npy", hongkong_train_data)

us_train_data = load_data(us_train_paths)
np.save("A:\study\en_voice/npy_data/us18_npy", us_train_data)

test_data = load_data(test_["path"])
np.save("A:\study\en_voice/npy_data/test18_npy", test_data)

print("done")

# sr = 18000 파일 끝에 18_npy