from model import Model
import torch
from utils.options import parser
args = parser.parse_args()
def build_model(args):
    model = None  # 默认返回 None
    if args.seq_len == 1:
        if args.network == 'model':
            model = Model()
    if model is None:
        print("Error: Model is None")
    return model
