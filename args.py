import argparse

def get_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset','-d', type=str, default="AD", \
        help='dataset choice')
    parser.add_argument('--modal','-m1', type=int, default=0, help='0:fmri,1:dti,2:both')
    parser.add_argument('--channel','-c', type=int, default=32, help='channel number')
    parser.add_argument('--fold','-f', type=int, default=-1, help='channel number')
    parser.add_argument('--layer','-l', type=int, default=2, help='layer number')
    parser.add_argument('--ab','-a', type=int, default=0, help='ablation study choice')
    parser.add_argument('--no', type=int, default=2, help='spit labels')
    parser.add_argument('-g','--gru', type=int, default=1, help='layer number')
    parser.add_argument('--alpha', type=float, default=0.8, help='layer number')


    args = parser.parse_args()
    args.dataset = args.dataset.upper()

    return args