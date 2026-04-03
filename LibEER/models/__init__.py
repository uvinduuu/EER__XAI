from models.DGCNN import DGCNN
try:
    from models.RGNN_official import SymSimGCNNet as RGNN
except ModuleNotFoundError:
    RGNN = None   # torch_geometric not installed — RGNN unavailable
from models.EEGNet import EEGNet
from models.STRNN import STRNN
from models.GCBNet import GCBNet
from models.DBN import DBN
from models.TSception import TSception
from models.SVM import SVM
from models.CDCN import CDCN
from models.HSLT import HSLT
from models.ACRNN import ACRNN
from models.GCBNet_BLS import GCBNet_BLS
from models.MsMda import MSMDA