import os
import argparse
import warnings
import sys


def get_args():
    parser = argparse.ArgumentParser(description='GAN algorithms')

    # gp model 
    parser.add_argument('--lr', default=.0002, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--id', default=1, type=int, help='unique id of every instance')
    parser.add_argument('--eval-only', action='store_true', help='will not save anything in this setting')
    parser.add_argument('--num-epochs', default=200, type=int)

    parser.add_argument('--save-dir', default='./save/', help='save directory')
    parser.add_argument('--log-dir', default='./logs/', help='save directory')

    args = parser.parse_args()
    args.log_dir = os.path.join(args.log_dir, str(args.id))
    args.save_dir = os.path.join(args.save_dir, str(args.id))
    if not args.eval_only:
        if os.path.exists(args.save_dir):
            warnings.warn('SAVE DIRECTORY ALREADY EXISTS!')
            ch = input('Press c to continue and s to stop: ')
            if ch == 's':
                sys.exit(0)
            elif ch == 'c':
                os.rename(args.save_dir, args.save_dir+'_old')
            elif ch != 'c':
                raise NotImplementedError 

        os.makedirs(args.save_dir)               
    return args