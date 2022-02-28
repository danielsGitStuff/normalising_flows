from common.prozess.Prozessor import Prozessor
from maf.mixlearn.dl3.MinibooneDL3 import MinibooneDL3

if __name__ == '__main__':
    # make sure the dataset is already in place before starting processes relying on it. Might cause race conditions otherwise
    p : Prozessor = Prozessor()
    dl3 = MinibooneDL3()
    dl3.run()
    print('donwloading miniboone done')